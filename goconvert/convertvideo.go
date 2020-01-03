package main

import (
	"errors"
	"fmt"
	"math"
	"os"

	"gocv.io/x/gocv"
)

const (
	STEP_WEIGHT int = 256  // max. 256
	STEP_STREAM int = 1024 // max. 65536
)

func ConvertVideo(conf *Config) error {

	var err error

	if conf.Ffmpegtemp {
		// show ffmpeg template
		err = showFfmpegTemplate(conf)

	} else if conf.OutputFileName == "-" {
		// stdout mode
		err = convertVideoStdout(conf)

	} else {
		// normal mode
		err = convertVideoNormal(conf)
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

func convertVideoStdout(conf *Config) error {
	return nil

}

func convertVideoNormal(conf *Config) error {

	// try to open as a video file
	capture, err := gocv.OpenVideoCapture(conf.InputFileName)
	if err != nil {
		return err
	}
	defer capture.Close()

	// mapping indexes
	inputWH := []int{
		int(capture.Get(gocv.VideoCaptureFrameWidth)),
		int(capture.Get(gocv.VideoCaptureFrameHeight)),
	}
	mapper, err := PrepareMapping(conf, inputWH)
	if err != nil {
		return err
	}

	// try to open output file
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
	writer, err := gocv.VideoWriterFile(conf.OutputFileName, "avc1", fps,
		mapper.ProjW, mapper.ProjH, true)
	if err != nil {
		return err
	}
	defer writer.Close()

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
	pickedStreamUint8 := PickupPixels(done, inputStream, mapper.PickupIndex)

	// remove gamma corection to get linear image
	pickedStream := RemoveGamma(done, pickedStreamUint8, conf.Gamma, mapper.NPickup)

	// mapping
	mappedStream := BilinearInterpolate(done, pickedStream, mapper.BilinearIndex, mapper.BilinearWeight)

	// gamma correction --> edge-blur --> overlap
	mappedStreamUint8 := GammaCorrect(done, mappedStream, conf.Gamma, conf.Contrast, mapper.NStore)

	// store pixels to output frame
	outputStream := StorePixels(done, mappedStreamUint8, mapper.StoreIndex, mapper.NProj)

	// write output stream to file (pipeline consumer)
	fmt.Println(conf.InputFileName)
	fmt.Printf("   size: %dx%d\n", inputWH[0], inputWH[1])
	fmt.Println(conf.OutputFileName)
	fmt.Printf("   size: %dx%d\n\n", mapper.ProjW, mapper.ProjH)

	imgOutput := gocv.NewMat()
	defer imgOutput.Close()
	cnt := 0
	percent := 0.0
	for buff := range outputStream {
		percent = float64(cnt) / float64(nframes) * 100.0
		fmt.Printf("\rMapping... %d/%d %.0f%%", cnt+1, nframes, percent)
		cnt++

		// write frame
		imgOutput, _ = gocv.NewMatFromBytes(mapper.ProjH, mapper.ProjW, gocv.MatTypeCV8UC3, buff)
		writer.Write(imgOutput)
	}
	fmt.Printf("\rMapping... %d/%d %.0f%%\n", cnt+1, nframes, percent)

	return nil
}

// generator of pipeline
func GenVideoFrame(
	done <-chan interface{}, vc *gocv.VideoCapture, nframes int,
) <-chan []uint8 {
	stream := make(chan []uint8)
	go func() {
		defer close(stream)
		frame := gocv.NewMat()
		defer frame.Close()
		for i := 0; i < nframes; i++ {
			_ = vc.Read(&frame)
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
	done <-chan interface{}, inputStream <-chan []uint8, pickupIndex []int,
) <-chan []uint8 {
	outputStream := make(chan []uint8)
	go func() {
		defer close(outputStream)
		for input := range inputStream {
			pickedPixels := make([]uint8, len(pickupIndex)*3)
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
	gamma float64, length int,
) <-chan []uint16 {

	// lookup table
	// inputSteram is []uint8 type. so, LUT is 256 step.
	var LUT [256]uint16
	maxStream := float64(STEP_STREAM - 1)
	for i := 0; i < 256; i++ {
		LUT[i] = uint16(math.Pow(float64(i)/255, gamma) * maxStream)
	}
	// process
	outputStream := make(chan []uint16)
	go func() {
		defer close(outputStream)
		for input := range inputStream {
			y := make([]uint16, length*3)
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
	done <-chan interface{}, inputStream <-chan []uint16,
	bilinearIndex [][]int, bilinearWeight [][]float64,
) <-chan []uint16 {

	maxWeight := float64(STEP_WEIGHT - 1)
	lenOutPixels := len(bilinearIndex[0])

	// lookup table
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
		W[i] = make([]int, lenOutPixels)
		for j, v := range bilinearWeight[i] {
			W[i][j] = int(v * maxWeight)
		}
	}

	// process
	outputStream := make(chan []uint16)
	go func() {
		defer close(outputStream)
		cnt := 0
		for input := range inputStream {
			// --------------------------------------------- takes time!!!!
			y := make([]uint16, lenOutPixels*3)
			cnt = 0
			for i := 0; i < 4; i++ {
				cnt = 0
				for j := 0; j < lenOutPixels; j++ {
					for k := 0; k < 3; k++ {
						y[cnt] += LUT[W[i][j]][input[bilinearIndex[i][j]+k]]
						cnt++
					}
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

// gamma correction
func GammaCorrect(
	done <-chan interface{}, inputStream <-chan []uint16,
	gamma float64, contrast float64, length int,
) <-chan []uint8 {

	// lookup table
	var LUT [STEP_STREAM]uint8
	maxStream := float64(STEP_STREAM - 1)
	for i := 0; i < STEP_STREAM; i++ {
		LUT[i] = uint8(math.Pow(float64(i)/maxStream, contrast/gamma) * 255)
	}

	// process
	outputStream := make(chan []uint8)
	go func() {
		defer close(outputStream)
		for input := range inputStream {
			y := make([]uint8, length*3)
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

// store pixels to output frame
func StorePixels(
	done <-chan interface{}, inputStream <-chan []uint8,
	storeIndex []int, nProjPixels int,
) <-chan []uint8 {

	// process
	outputStream := make(chan []uint8)
	go func() {
		defer close(outputStream)
		for input := range inputStream {
			y := make([]uint8, nProjPixels*3)
			cnt := 0
			for _, v := range storeIndex {
				for k := 0; k < 3; k++ {
					y[v+k] = input[cnt]
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
