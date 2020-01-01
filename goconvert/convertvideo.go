package main

import (
	//"errors"

	"gocv.io/x/gocv"
)

func ConvertVideo(conf *Config) error {

	// try to open as a video file
	capture, err := gocv.OpenVideoCapture(conf.InputFileName)
	if err != nil {
		return err
	}
	defer capture.Close()

	// read mapping table

	// ffmpeg template or mapping
	if conf.Ffmpegtemp {
		return nil
	}

	return nil
}
