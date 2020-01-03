package main

import (
	"fmt"
	"os"

	"github.com/akamensky/argparse"
)

func main() {

	// parse arguments
	conf, err := ParseArg(os.Args)
	if err != nil {
		return
	}

	// try mapping as still image
	errStill := ConvertStill(conf)
	if errStill == nil {
		return
	}
	if errStill.Error() != "invalid input" {
		fmt.Println(errStill)
		return
	}

	// try mapping as still video () only if
	// ConcertStill() failed with invalid input
	errVideo := ConvertVideo(conf)
	if errVideo != nil {
		fmt.Println(errVideo)
	}
	return
}

type Config struct {
	InputFileName      string
	OutputFileName     string
	PathToMappingTable string
	Offset             float64
	EdgeBlur           float64
	Contrast           float64
	Gamma              float64
	NFrames            int
	Ffmpegtemp         bool
	Overwrite          bool
}

func ParseArg(args []string) (*Config, error) {

	// Create new parser object
	parser := argparse.NewParser("goconvert", "a tool for mapping spherical image")

	// Create string flag
	i := parser.String("i", "in", &argparse.Options{Required: true, Help: "input image or movie filename"})
	d := parser.String("d", "map", &argparse.Options{Required: true, Help: "path to directly of mapping table"})
	o := parser.String("o", "out", &argparse.Options{Required: false, Default: "output.mp4", Help: "output filename (default: output.mp4)"})
	offset := parser.Float("", "offset", &argparse.Options{Required: false, Default: 0.0, Help: "Horizontal offset (default: 0.0, unit: degree)"})
	edgeblur := parser.Float("", "edgeblur", &argparse.Options{Required: false, Default: 0.5, Help: "Edge blur (default: 0.5, unit:degree)"})
	contrast := parser.Float("", "contrast", &argparse.Options{Required: false, Default: 1.0, Help: "Contrast (default: Gamma 1.0)"})
	gamma := parser.Float("", "gamma", &argparse.Options{Required: false, Default: 2.2, Help: "Gamma (default: 2.2)"})
	nframes := parser.Int("", "nframes", &argparse.Options{Required: false, Default: 0, Help: "number of frames to video convert"})
	ffmpegtemp := parser.Flag("f", "ffmpeg-template", &argparse.Options{Required: false, Help: "show template options for ffmpeg"})
	overwrite := parser.Flag("y", "overwrite", &argparse.Options{Required: false, Help: "overwrite output files"})

	// Parse input
	err := parser.Parse(os.Args)
	if err != nil {
		// In case of error print error and print usage
		// This can also be done by passing -h or --help flags
		fmt.Print(parser.Usage(err))
		return nil, err
	}

	// set config
	conf := &Config{
		InputFileName:      *i,
		OutputFileName:     *o,
		PathToMappingTable: *d,
		Offset:             *offset,
		EdgeBlur:           *edgeblur,
		Contrast:           *contrast,
		Gamma:              *gamma,
		NFrames:            *nframes,
		Ffmpegtemp:         *ffmpegtemp,
		Overwrite:          *overwrite,
	}
	return conf, nil
}
