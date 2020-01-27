package main

import (
	"errors"
	"fmt"
	"math"
	"os"

	"gocv.io/x/gocv"
)

const (
	STEP_WEIGHT int = 256
	STEP_STREAM int = 4096 // STEP_STREAM must be a power of 2
	BUFFER_SIZE int = 1
)

func ConvertVideo(conf *Config) error {

	var err error

	if conf.Ffmpegtemp {
		// show ffmpeg template
		err = showFfmpegTemplate(conf)

	} else {
		// mapping
		err = convertVideo(conf)
	}

	return err
}

func showFfmpegTemplate(conf *Config) error {
	// try to open as a video file
	capture, err := gocv.OpenVideoCapture(conf.InputFileName)
	if err != nil {
		return err
	}
	defer capture.Close()

	// get projector size
	projH, projW, err := ReadProjectorHW(conf.PathToMappingTable)
	if err != nil {
		return err
	}

	// show template command
	base := fmt.Sprintf("%s -i %s -d %s -o - | ",
		os.Args[0], conf.InputFileName, conf.PathToMappingTable)
	size := fmt.Sprintf("-s %dx%d", projW, projH)
	vfmt := "-f rawvideo -pix_fmt bgr24"
	fps := fmt.Sprintf("-framerate %f", capture.Get(gocv.VideoCaptureFPS))
	ffmpeg := fmt.Sprintf("ffmpeg %s %s %s -i - output.mp4", vfmt, size, fps)
	cmd := base + ffmpeg

	fmt.Printf("template for ffmpeg:\n%s\n", cmd)
	return nil
}

func convertVideo(conf *Config) error {

	// try to open input file as a video
	capture, err := gocv.OpenVideoCapture(conf.InputFileName)
	if err != nil {
		return err
	}
	defer capture.Close()

	// try to open output file
	// if not filename is "-", gocv write video.
	// else, video stream output to stdout.
	var writer *gocv.VideoWriter
	if conf.OutputFileName != "-" {
		if conf.OutputFileName == "" {
			conf.OutputFileName = "output.mp4"
		}
		if !conf.Overwrite {
			exists := func() bool {
				_, err := os.Stat(conf.OutputFileName)
				return err == nil
			}()
			if exists {
				return errors.New(conf.OutputFileName + " already exists")
			}
		}
		fps := capture.Get(gocv.VideoCaptureFPS)
		// get projector size
		projH, projW, err := ReadProjectorHW(conf.PathToMappingTable)
		if err != nil {
			return err
		}
		writer, err = gocv.VideoWriterFile(conf.OutputFileName, "avc1",
			fps, int(projW), int(projH), true)
		if err != nil {
			return err
		}
		defer writer.Close()
	}

	// mapping indexes
	inputWH := []int{
		int(capture.Get(gocv.VideoCaptureFrameWidth)),
		int(capture.Get(gocv.VideoCaptureFrameHeight)),
	}
	mapper, err := PrepareMapping(conf, inputWH)
	if err != nil {
		return err
	}

	// ----------------------------------------------------------
	//                      MAPPING PROCESS
	// ----------------------------------------------------------
	// It's processed in the pipeline

	done := make(chan interface{})
	defer close(done)

	// read frames (pipeline generator)
	nframes := int(capture.Get(gocv.VideoCaptureFrameCount))
	if (0 < conf.NFrames) && (conf.NFrames < nframes) {
		nframes = conf.NFrames
	}
	inputStream := GenVideoFrame(done, capture, nframes)

	// pick out pixels used for the mapping process
	pickedStreamUint8 := PickupPixels(done, inputStream, mapper)

	// remove gamma corection to get linear image
	pickedStream := RemoveGamma(done, pickedStreamUint8, conf, mapper)

	// mapping
	mappedStream1 := BilinearInterpolate(done, pickedStream, mapper)

	// gamma correction --> edge-blur --> overlap
	mappedStream2 := GammaCorrect(done, mappedStream1, conf, mapper)
	mappedStream3 := EdgeBlur(done, mappedStream2, mapper)
	mappedStream4 := Overlap(done, mappedStream3, mapper)

	// store pixels to output frame
	outputStream := StorePixels(done, mappedStream4, mapper)

	if conf.OutputFileName == "-" {
		// write output stream to stdout (pipeline consumer)
		for output := range outputStream {
			os.Stdout.Write(output)
		}
	} else {
		// write output stream to file (pipeline consumer)
		frameNumberStream := WriteVideo(done, outputStream, writer, mapper)

		fmt.Println(conf.InputFileName)
		fmt.Printf("   size: %dx%d\n", inputWH[0], inputWH[1])
		fmt.Println(conf.OutputFileName)
		fmt.Printf("   size: %dx%d\n\n", mapper.ProjW, mapper.ProjH)

		percent := 0.0
		for n := range frameNumberStream {
			percent = float64(n) / float64(nframes) * 100.0
			fmt.Printf("\rMapping... %d/%d %.0f%%", n, nframes, percent)
		}
		fmt.Printf("\n")
	}

	return nil
}

// generator of pipeline
func GenVideoFrame(
	done <-chan interface{}, vc *gocv.VideoCapture, nframes int,
) <-chan []uint8 {
	stream := make(chan []uint8, BUFFER_SIZE)
	go func() {
		defer close(stream)
		frame := gocv.NewMat()
		defer frame.Close()
		var ok bool
		for i := 0; i < nframes; i++ {
			ok = vc.Read(&frame)
			if !ok {
				if i == 0 {
					count := 0
					for !ok {
						ok = vc.Read(&frame)
						count++
						if count > 1000 {
							fmt.Println("too many empty frame at the beginning")
							return
						}
					}
				} else {
					return
				}
			}
			select {
			case <-done:
				return
			case stream <- frame.ToBytes(): // ToBytes "copies" Mat data to byte array.
			}
		}
	}()
	return stream
}

// pick out pixels used for the mapping process
func PickupPixels(
	done <-chan interface{}, inputStream <-chan []uint8, mapper *Mapper,
) <-chan []uint8 {
	pickupIndex := mapper.PickupIndex
	L := len(pickupIndex) * 3
	outputStream := make(chan []uint8, BUFFER_SIZE)
	go func() {
		defer close(outputStream)
		for input := range inputStream {
			pickedPixels := make([]uint8, L)
			i := 0
			for _, iPick := range pickupIndex {
				for j := 0; j < 3; j++ {
					pickedPixels[i] = input[iPick+j]
					i++
				}
			}
			select {
			case <-done:
				return
			case outputStream <- pickedPixels:
			}
		}
	}()
	return outputStream
}

// remove gamma correction
func RemoveGamma(
	done <-chan interface{}, inputStream <-chan []uint8,
	conf *Config, mapper *Mapper,
) <-chan []uint16 {
	gamma := conf.Gamma
	L := len(mapper.PickupIndex) * 3
	// lookup table
	var LUT [256]uint16 // inputSteram is []uint8 type. so, LUT is 256 step.
	maxStream := float64(STEP_STREAM - 1)
	for i := 0; i < 256; i++ {
		LUT[i] = uint16(math.Pow(float64(i)/255, gamma) * maxStream)
	}
	// process
	outputStream := make(chan []uint16, BUFFER_SIZE)
	go func() {
		defer close(outputStream)
		for input := range inputStream {
			y := make([]uint16, L)
			for i, v := range input {
				y[i] = LUT[v]
			}
			select {
			case <-done:
				return
			case outputStream <- y:
			}
		}
	}()
	return outputStream
}

// mapping
func BilinearInterpolate(
	done <-chan interface{}, inputStream <-chan []uint16, mapper *Mapper,
) <-chan []uint16 {
	index := mapper.BilinearIndex
	weight := mapper.BilinearWeight
	L := len(index[0])

	// lookup table
	maxWeight := float64(STEP_WEIGHT - 1)
	var LUT [STEP_WEIGHT][STEP_STREAM]uint16
	for i := 0; i < STEP_WEIGHT; i++ {
		for j := 0; j < STEP_STREAM; j++ {
			LUT[i][j] = uint16(float64(j) * float64(i) / maxWeight)
		}
	}

	// quantize weight
	// this is used as index of LUT
	W := make([][]int, 4)
	for i := 0; i < 4; i++ {
		W[i] = make([]int, L)
		for j, v := range weight[i] {
			W[i][j] = int(v * maxWeight)
		}
	}

	// ----- process -----
	// distribute
	distribute := func(
		done <-chan interface{}, in <-chan []uint16,
	) (_, _, _, _ <-chan []uint16) {
		out1 := make(chan []uint16, BUFFER_SIZE)
		out2 := make(chan []uint16, BUFFER_SIZE)
		out3 := make(chan []uint16, BUFFER_SIZE)
		out4 := make(chan []uint16, BUFFER_SIZE)
		go func() {
			defer close(out1)
			defer close(out2)
			defer close(out3)
			defer close(out4)
			for v := range in {
				// copy channel
				var out1, out2, out3, out4 = out1, out2, out3, out4
				for i := 0; i < 4; i++ {
					select {
					case <-done:
						return
					case out1 <- v:
						out1 = nil
					case out2 <- v:
						out2 = nil
					case out3 <- v:
						out3 = nil
					case out4 <- v:
						out4 = nil
					}
				}
			}
		}()
		return out1, out2, out3, out4
	}

	// weight
	weighting := func(
		done <-chan interface{}, in <-chan []uint16, index []int, weight []int,
	) <-chan []uint16 {
		out := make(chan []uint16, BUFFER_SIZE)
		go func() {
			defer close(out)
			cnt := 0
			for x := range in {
				y := make([]uint16, L*3)
				cnt = 0
				for j := 0; j < L; j++ {
					for BGR := 0; BGR < 3; BGR++ {
						y[cnt] += LUT[weight[j]][x[index[j]+BGR]]
						cnt++
					}
				}
				select {
				case <-done:
					return
				case out <- y:
				}
			}
		}()
		return out
	}

	// sum
	sum := func(
		done <-chan interface{}, in1, in2, in3, in4 <-chan []uint16,
	) <-chan []uint16 {
		out := make(chan []uint16, BUFFER_SIZE)
		go func() {
			defer close(out)
			var val []uint16
			var ok bool
			for {
				var in1, in2, in3, in4 = in1, in2, in3, in4
				vsum := make([]uint16, L*3)
				for i := 0; i < 4; i++ {
					select {
					case <-done:
						return
					case val, ok = <-in1:
						in1 = nil
					case val, ok = <-in2:
						in2 = nil
					case val, ok = <-in3:
						in3 = nil
					case val, ok = <-in4:
						in4 = nil
					}
					if ok == false {
						return
					}
					for j, v := range val {
						vsum[j] += v
					}
				}
				select {
				case <-done:
					return
				case out <- vsum:
				}
			}
		}()
		return out
	}

	// pipe
	in0, in1, in2, in3 := distribute(done, inputStream)
	out0 := weighting(done, in0, index[0], W[0])
	out1 := weighting(done, in1, index[1], W[1])
	out2 := weighting(done, in2, index[2], W[2])
	out3 := weighting(done, in3, index[3], W[3])
	out := sum(done, out0, out1, out2, out3)
	return out
}

// gamma correction
func GammaCorrect(
	done <-chan interface{}, inputStream <-chan []uint16,
	conf *Config, mapper *Mapper,
) <-chan []uint16 {
	gamma := conf.Contrast / conf.Gamma
	L := len(mapper.StoreIndex) * 3

	// lookup table
	var LUT [STEP_STREAM]uint16
	maxStream := float64(STEP_STREAM - 1)
	for i := 0; i < STEP_STREAM; i++ {
		LUT[i] = uint16(math.Pow(float64(i)/maxStream, gamma) * maxStream)
	}

	// process
	outputStream := make(chan []uint16, BUFFER_SIZE)
	go func() {
		defer close(outputStream)
		for input := range inputStream {
			y := make([]uint16, L)
			for i := range input {
				y[i] = LUT[input[i]]
			}
			select {
			case <-done:
				return
			case outputStream <- y:
			}
		}
	}()
	return outputStream
}

// edge blur
func EdgeBlur(
	done <-chan interface{}, inputStream <-chan []uint16, mapper *Mapper,
) <-chan []uint16 {

	index := mapper.EdgeBlurIndex
	weight := mapper.EdgeBlurWeight

	// lookup table
	maxWeight := float64(STEP_WEIGHT - 1)

	var LUT [STEP_WEIGHT][STEP_STREAM]uint16
	for i := 0; i < STEP_WEIGHT; i++ {
		for j := 0; j < STEP_STREAM; j++ {
			LUT[i][j] = uint16(float64(j) * float64(i) / maxWeight)
		}
	}

	// quantize weight
	// this is used as index of LUT
	W := make([]int, len(weight))
	for i, v := range weight {
		W[i] = int(v * maxWeight)
	}

	// process
	outputStream := make(chan []uint16, BUFFER_SIZE)
	go func() {
		defer close(outputStream)
		for input := range inputStream {
			y := make([]uint16, len(input))
			copy(y, input) // --------------------------------------------- cost
			for i := 0; i < len(index); i++ {
				for BGR := 0; BGR < 3; BGR++ {
					y[index[i]+BGR] = LUT[W[i]][input[index[i]+BGR]]
				}
			}
			select {
			case <-done:
				return
			case outputStream <- y:
			}
		}
	}()
	return outputStream
}

// overlap
func Overlap(
	done <-chan interface{}, inputStream <-chan []uint16, mapper *Mapper,
) <-chan []uint16 {

	index := mapper.OverlapIndex
	weight := mapper.OverlapWeight

	// lookup table
	maxWeight := float64(STEP_WEIGHT - 1)
	maxStream := float64(STEP_STREAM - 1)
	var LUT [STEP_WEIGHT][STEP_STREAM]uint16
	xi := make([]float64, STEP_STREAM)
	for i := 0; i < STEP_STREAM; i++ {
		xi[i] = float64(i) / maxStream
	}
	yi := InterpolateOverlapWeight(mapper.ToneInput, mapper.ToneOutput, xi)
	yiWeighted := make([]float64, len(yi))
	xiWeighted := make([]float64, len(yi))
	w := 0.0
	for i := 0; i < STEP_WEIGHT; i++ {
		w = float64(i) / maxWeight
		for j, v := range yi {
			yiWeighted[j] = v * w
		}
		xiWeighted = InterpolateOverlapWeight(mapper.ToneOutput,
			mapper.ToneInput, yiWeighted)
		for j, v := range xiWeighted {
			LUT[i][j] = uint16(v * maxStream)
		}
	}

	// quantize weight
	W := make([]int, len(weight))
	for i, v := range weight {
		W[i] = int(v * maxWeight)
	}

	// process
	outputStream := make(chan []uint16, BUFFER_SIZE)
	go func() {
		defer close(outputStream)
		for input := range inputStream {
			y := make([]uint16, len(input))
			copy(y, input) // --------------------------------------------- cost
			for i := 0; i < len(index); i++ {
				for BGR := 0; BGR < 3; BGR++ {
					y[index[i]+BGR] = LUT[W[i]][input[index[i]+BGR]]
				}
			}
			select {
			case <-done:
				return
			case outputStream <- y:
			}
		}
	}()
	return outputStream
}

func InterpolateOverlapWeight(x, y, xi []float64) []float64 {
	yi := make([]float64, len(xi))
	for i, v := range xi {
		for j := 0; j < (len(x) - 1); j++ {
			if (x[j] <= v) && (v <= x[j+1]) {
				yi[i] = y[j] - (y[j]-y[j+1])/(x[j]-x[j+1])*(x[j]-v)
				break
			}
		}
	}
	return yi
}

// store pixels to output frame
func StorePixels(
	done <-chan interface{}, inputStream <-chan []uint16, mapper *Mapper,
) <-chan []uint8 {
	storeIndex := mapper.StoreIndex
	L := mapper.ProjW * mapper.ProjH * 3
	nBitShift := int(math.Log2(float64(STEP_STREAM / 256)))

	// process
	outputStream := make(chan []uint8, BUFFER_SIZE)
	go func() {
		defer close(outputStream)
		for input := range inputStream {
			y := make([]uint8, L)
			cnt := 0
			for _, v := range storeIndex {
				for k := 0; k < 3; k++ {
					y[v+k] = uint8(input[cnt] >> nBitShift) // uint16 to uint8
					cnt++
				}
			}

			select {
			case <-done:
				return
			case outputStream <- y:
			}
		}
	}()
	return outputStream
}

// write frame to file
func WriteVideo(
	done <-chan interface{}, inputStream <-chan []uint8,
	writer *gocv.VideoWriter, mapper *Mapper,
) <-chan int {
	frameNumber := make(chan int)
	go func() {
		defer close(frameNumber)
		cnt := 0
		mat := gocv.NewMat()
		defer mat.Close()

		for input := range inputStream {
			mat, _ = gocv.NewMatFromBytes(
				mapper.ProjH, mapper.ProjW, gocv.MatTypeCV8UC3, input)
			writer.Write(mat)
			cnt++

			select {
			case <-done:
				return
			case frameNumber <- cnt:
			}
		}
	}()
	return frameNumber
}
