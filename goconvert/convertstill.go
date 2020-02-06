package main

import (
	"errors"
	"math"
	"os"

	"gocv.io/x/gocv"
)

func ConvertStill(conf *Config) error {

	// try to open input file as a still image
	inImg := gocv.IMRead(conf.InputFileName, gocv.IMReadColor)
	defer inImg.Close()
	if inImg.Empty() {
		return errors.New("invalid input")
	}

	// --ffmpeg-template option is invalid for still image
	if conf.Ffmpegtemp {
		msg := "ffmpeg-template flag is not supported for still image"
		return errors.New(msg)
	}

	// try to open output file
	if conf.OutputFileName == "-" {
		return errors.New("stdout mode is not supported for still image")
	}
	if conf.OutputFileName == "output.mp4/png" {
		conf.OutputFileName = "output.png"
	}
	if !conf.Overwrite {
		exists := func() bool {
			_, err := os.Stat(conf.OutputFileName)
			return err == nil
		}()
		if exists {
			return errors.New("output file already exists")
		}
	}

	// mapping indexes
	inputWH := []int{inImg.Cols(), inImg.Rows()}
	mapper, err := PrepareMapping(conf, inputWH)
	if err != nil {
		return err
	}

	// convert Mat to uint8
	buffer := inImg.ToBytes()

	// pickup pixels and remove gamma
	pickedPixels := make([]float64, len(mapper.PickupIndex)*3)
	i := 0
	var v uint8
	for _, iPick := range mapper.PickupIndex {
		for BGR := 0; BGR < 3; BGR++ {
			v = buffer[iPick+BGR]
			pickedPixels[i] = math.Pow(float64(v)/255, conf.Gamma)
			i++
		}
	}

	// bilinear interpolation
	L := len(mapper.BilinearIndex[0])
	mappedPixels := make([]float64, L*3)
	for i := 0; i < 4; i++ {
		cnt := 0
		index := mapper.BilinearIndex[i]
		weight := mapper.BilinearWeight[i]
		for j := 0; j < L; j++ {
			for BGR := 0; BGR < 3; BGR++ {
				mappedPixels[cnt] += pickedPixels[index[j]+BGR] * weight[j]
				cnt++
			}
		}
	}

	// gamma correction
	gamma := conf.Contrast / conf.Gamma
	for i, v := range mappedPixels {
		mappedPixels[i] = math.Pow(v, gamma)
	}

	// edge blur
	for cnt, i := range mapper.EdgeBlurIndex {
		for BGR := 0; BGR < 3; BGR++ {
			mappedPixels[i+BGR] = mappedPixels[i+BGR] * mapper.EdgeBlurWeight[cnt]
		}
	}

	// overlap
	i2oInterp := func(xi float64) float64 {
		x := mapper.ToneInput
		y := mapper.ToneOutput
		var yi float64
		for j := 0; j < (len(x) - 1); j++ {
			if (x[j] <= xi) && (xi <= x[j+1]) {
				yi = y[j] - (y[j]-y[j+1])/(x[j]-x[j+1])*(x[j]-xi)
				break
			}
		}
		return yi
	}
	o2iInterp := func(xi float64) float64 {
		x := mapper.ToneOutput
		y := mapper.ToneInput
		var yi float64
		for j := 0; j < (len(x) - 1); j++ {
			if (x[j] <= xi) && (xi <= x[j+1]) {
				yi = y[j] - (y[j]-y[j+1])/(x[j]-x[j+1])*(x[j]-xi)
				break
			}
		}
		return yi
	}
    
	for cnt, i := range mapper.OverlapIndex {
		for BGR := 0; BGR < 3; BGR++ {
			mappedPixels[i+BGR] = o2iInterp(
                i2oInterp(mappedPixels[i+BGR]) * mapper.OverlapWeight[cnt])
		}
	}
    
	// store pixels to frame
	outBuffer := make([]uint8, mapper.ProjW*mapper.ProjH*3)
	cnt := 0
	var val float64
	for _, i := range mapper.StoreIndex {
		for BGR := 0; BGR < 3; BGR++ {
			val = mappedPixels[cnt]
			if val > 1.0 {
				val = 1.0
			}
			outBuffer[i+BGR] = uint8(val * 255)
			cnt++
		}
	}

	// write frame to file
	outImg := gocv.NewMat()
	defer outImg.Close()
	outImg, _ = gocv.NewMatFromBytes(mapper.ProjH, mapper.ProjW,
		gocv.MatTypeCV8UC3, outBuffer)
	ok := gocv.IMWrite(conf.OutputFileName, outImg)
	if !ok {
		msg := "failed to write image"
		return errors.New(msg)
	}

	return nil
}


