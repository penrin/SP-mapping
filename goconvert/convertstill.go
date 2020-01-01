package main

import (
	"errors"

	"gocv.io/x/gocv"
)

func ConvertStill(conf *Config) error {

	// still 画像入力時の --ffmpeg-template フラグは無効
	if conf.Ffmpegtemp {
		msg := "ffmpeg-template flag is not supported for still image"
		return errors.New(msg)
	}

	// try to open as a image file
	img := gocv.IMRead(conf.InputFileName, gocv.IMReadColor)
	if img.Empty() {
		return errors.New("invalid input")
	}

	return nil
}
