package main

import (
	"errors"
	"fmt"
	"os"

	"gocv.io/x/gocv"
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

	// read mapping table
	/*
		mapTable, err := ReadMap(conf.PathToMappingTable)
		if err != nil {
			return err
		}
	*/
	// get projector size
	projH, projW, err := ReadProjectorHW(conf.PathToMappingTable)
	if err != nil {
		return err
	}

	// show template command
	base := fmt.Sprintf("%s -i %s -d %s - | ",
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

	// try to open as a video file
	capture, err := gocv.OpenVideoCapture(conf.InputFileName)
	if err != nil {
		return err
	}
	defer capture.Close()

	// read mapping table
	/*
		mapTable, err := ReadMap(conf.PathToMappingTable)
		if err != nil {
			return err
		}
	*/
	return nil
}

func convertVideoNormal(conf *Config) error {

	// try to open as a video file
	capture, err := gocv.OpenVideoCapture(conf.InputFileName)
	if err != nil {
		return err
	}
	defer capture.Close()

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

	// read mapping table
	/*
		    mapTable, err := ReadMap(conf.PathToMappingTable)
			if err != nil {
				return err
			}
	*/
	return nil
}
