package main

import (
	"errors"
	"os"
	"path/filepath"

	"github.com/penrin/gonpy"
)

type MappingTable struct {

	// Projection direction
	Y       []int64
	X       []int64
	Polar   []float64
	Azimuth []float64

	// overlap information
	OvlpY      []int64
	OvlpX      []int64
	OvlpWeight []float64

	// projector tone
	ToneInput  []float64
	ToneOutput []float64

	// projector size
	ProjH int64
	ProjW int64
}

func ReadMap(f string) (*MappingTable, error) {
	npz, err := openMap(f)
	if err != nil {
		return nil, err
	}
	defer npz.Close()

	mappingTable, err := fetchMap(npz)
	if err != nil {
		return nil, err
	}
	return mappingTable, nil
}

// read projector size only
func ReadProjectorHW(f string) (int64, int64, error) {

	var projH int64
	var projW int64

	npz, err := openMap(f)
	if err != nil {
		return projH, projW, err
	}
	defer npz.Close()

	// read
	npy, _ := npz.Get("proj_HW")
	if npy.Dtype != "i8" {
		return projH, projW, errors.New("Invalid dtype mapping table")
	}
	projHW, err := npy.GetInt64()
	if err != nil {
		return projH, projW, err
	}
	projH = projHW[0]
	projW = projHW[1]
	return projH, projW, err
}

func openMap(f string) (*gonpy.NpzReader, error) {

	filename := f

	// add "/mapping_table.npz" to f is directory
	isDir := func() bool {
		stat, err := os.Stat(filename)
		return (err == nil) && stat.IsDir()
	}()
	if isDir {
		filename = filepath.Join(filename, "mapping_table.npz")
	}

	// try to open
	npz, err := gonpy.OpenNpzReader(filename)
	if err != nil {
		return nil, err
	}

	// check .npz contents
	keyList := []string{
		"y", "x", "polar", "azimuth",
		"ovlp_y", "ovlp_x", "ovlp_weight",
		"tone_input", "tone_output", "proj_HW",
	}
	var npy *gonpy.NpyReader
	for _, key := range keyList {
		npy, _ = npz.Get(key)
		if npy == nil {
			msg := "invalid mapping table. \"" + key + "\" is not included."
			err := errors.New(msg)
			npz.Close()
			return nil, err
		}
	}
	return npz, nil
}

func fetchMap(npz *gonpy.NpzReader) (*MappingTable, error) {

	var npy *gonpy.NpyReader

	// read
	errDtype := errors.New("Invalid dtype mapping table")
	npy, _ = npz.Get("y")
	if npy.Dtype != "i8" {
		return nil, errDtype
	}
	y, err := npy.GetInt64()
	if err != nil {
		return nil, err
	}

	npy, _ = npz.Get("x")
	if npy.Dtype != "i8" {
		return nil, errDtype
	}
	x, err := npy.GetInt64()
	if err != nil {
		return nil, err
	}

	npy, _ = npz.Get("polar")
	if npy.Dtype != "f8" {
		return nil, errDtype
	}
	polar, err := npy.GetFloat64()
	if err != nil {
		return nil, err
	}

	npy, _ = npz.Get("azimuth")
	if npy.Dtype != "f8" {
		return nil, errDtype
	}
	azimuth, err := npy.GetFloat64()
	if err != nil {
		return nil, err
	}

	npy, _ = npz.Get("ovlp_y")
	if npy.Dtype != "i8" {
		return nil, errDtype
	}
	ovlpY, err := npy.GetInt64()
	if err != nil {
		return nil, err
	}

	npy, _ = npz.Get("ovlp_x")
	if npy.Dtype != "i8" {
		return nil, errDtype
	}
	ovlpX, err := npy.GetInt64()
	if err != nil {
		return nil, err
	}

	npy, _ = npz.Get("ovlp_weight")
	if npy.Dtype != "f8" {
		return nil, errDtype
	}
	ovlpWeight, err := npy.GetFloat64()
	if err != nil {
		return nil, err
	}

	npy, _ = npz.Get("tone_input")
	if npy.Dtype != "f8" {
		return nil, errDtype
	}
	toneInput, err := npy.GetFloat64()
	if err != nil {
		return nil, err
	}

	npy, _ = npz.Get("tone_output")
	if npy.Dtype != "f8" {
		return nil, errDtype
	}
	toneOutput, err := npy.GetFloat64()
	if err != nil {
		return nil, err
	}

	npy, _ = npz.Get("proj_HW")
	if npy.Dtype != "i8" {
		return nil, errDtype
	}
	projHW, err := npy.GetInt64()
	if err != nil {
		return nil, err
	}

	mappingTable := &MappingTable{
		Y:          y,
		X:          x,
		Polar:      polar,
		Azimuth:    azimuth,
		OvlpY:      ovlpY,
		OvlpX:      ovlpX,
		OvlpWeight: ovlpWeight,
		ToneInput:  toneInput,
		ToneOutput: toneOutput,
		ProjH:      projHW[0],
		ProjW:      projHW[1],
	}
	return mappingTable, nil
}
