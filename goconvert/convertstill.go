package main

import (
    "os"
	"errors"

	"gocv.io/x/gocv"
)

func ConvertStill(conf *Config) error {
    
	// try to open as a image file
	img := gocv.IMRead(conf.InputFileName, gocv.IMReadColor)
	if img.Empty() {
		return errors.New("invalid input")
	}

	// still 画像入力時の --ffmpeg-template フラグは無効
	if conf.Ffmpegtemp {
		msg := "ffmpeg-template flag is not supported for still image"
		return errors.New(msg)
	}
    
    // try to open output file
    if conf.OutputFileName == "-" {
        return errors.New("stdout mode is not supported for still image")
    }
    if !conf.Overwrite {
        exists := func () bool {
            _, err := os.Stat(conf.OutputFileName)
            return err == nil
        } ()
        if exists {
            return errors.New("output file already exists")
        }
    }


	return nil
}
