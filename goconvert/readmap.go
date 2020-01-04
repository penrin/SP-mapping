package main

import (
	"errors"
	"math"
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

// PrepareMapping() calculate various indexes and weights required
// for actual processing based on mapping-table
type Mapper struct {

	//
	PickupIndex []int

	//
	BilinearIndex  [][]int
	BilinearWeight [][]float64

	//
	EdgeBlurIndex  []int
	EdgeBlurWeight []float64

	//
	StoreIndex []int
	ProjH      int
	ProjW      int

	//
	NPickup int
	NStore  int
	NProj   int
}

func PrepareMapping(conf *Config, inputWH []int) (*Mapper, error) {

	// try to read mapping-table
	mapTable, err := ReadMap(conf.PathToMappingTable)
	if err != nil {
		return nil, err
	}

	NUMPIXEL := 3 // BGR colors

	// ----------------------------------------------------
	//      PREPARATION FOR "BILINEAR INTERPOLATION"
	// ----------------------------------------------------
	//
	//          x1    x2
	//          |     |
	//    y1 --(1)---(2)--   <= (#) is data points.
	//          |     |
	//          | *   |      <= * is sampling point (xSub, ySub).
	//    y2 --(3)---(4)--
	//          |     |
	//
	// sampling point data (proector RGB value) is determined
	//  from the surrounding four points (input RGB data).
	//

	L := len(mapTable.X)
	xSub := make([]float64, L)
	x1 := make([]int, L)
	x2 := make([]int, L)
	xPixPerDeg := float64(inputWH[0]) / 360
	for i := 0; i < L; i++ {
		xSub[i] = (mapTable.Azimuth[i] - conf.Offset) * xPixPerDeg
		x1[i] = int(math.Floor(xSub[i])) // round down to the nearest decimal
		x2[i] = x1[i] + 1
	}
	ySub := make([]float64, L)
	y1 := make([]int, L)
	y2 := make([]int, L)
	yPixPerDeg := float64(inputWH[1]) / 180
	for i := 0; i < L; i++ {
		ySub[i] = mapTable.Polar[i] * yPixPerDeg
		y1[i] = int(math.Floor(ySub[i])) // round down to the nearest decimal
		y2[i] = y1[i] + 1
	}

	// weight for bilinear interpolation
	w1 := make([]float64, L)
	w2 := make([]float64, L)
	w3 := make([]float64, L)
	w4 := make([]float64, L)
	var dx, dy float64
	for i := 0; i < L; i++ {
		dx = xSub[i] - float64(x1[i])
		dy = ySub[i] - float64(y1[i])
		w1[i] = (1 - dx) * (1 - dy)
		w2[i] = dx * (1 - dy)
		w3[i] = (1 - dx) * dy
		w4[i] = dx * dy
	}

	// roll (Must be processed after calculating the bilinear weight!!)
	for i := 0; i < L; i++ {
		x1[i] %= inputWH[0]
		x2[i] %= inputWH[0]
		y1[i] %= inputWH[1]
		y2[i] %= inputWH[1]
	}

	// calcurate index that selects the pixels
	// to be used for processing from input
	isUsedPixel := make([]bool, inputWH[0]*inputWH[1])
	for i := 0; i < L; i++ {
		isUsedPixel[y1[i]*inputWH[0]+x1[i]] = true
		isUsedPixel[y1[i]*inputWH[0]+x2[i]] = true
		isUsedPixel[y2[i]*inputWH[0]+x1[i]] = true
		isUsedPixel[y2[i]*inputWH[0]+x2[i]] = true
	}
	numUsedPixel := 0
	usedPixelNumbering := make([]int, inputWH[0]*inputWH[1])
	for i, yes := range isUsedPixel {
		if yes {
			usedPixelNumbering[i] = numUsedPixel
			numUsedPixel++
		}
	}
	pickupIndex := make([]int, numUsedPixel)
	j := 0
	for i, yes := range isUsedPixel {
		if yes {
			pickupIndex[j] = i * NUMPIXEL
			j++
		}
	}

	// index for bilinear interpolation
	i1 := make([]int, L)
	i2 := make([]int, L)
	i3 := make([]int, L)
	i4 := make([]int, L)
	for i := 0; i < L; i++ {
		i1[i] = usedPixelNumbering[y1[i]*inputWH[0]+x1[i]] * NUMPIXEL
		i2[i] = usedPixelNumbering[y1[i]*inputWH[0]+x2[i]] * NUMPIXEL
		i3[i] = usedPixelNumbering[y2[i]*inputWH[0]+x1[i]] * NUMPIXEL
		i4[i] = usedPixelNumbering[y2[i]*inputWH[0]+x2[i]] * NUMPIXEL
	}

	// ----------------------------------------------------
	//      PREPARATION FOR EDGE-BLUR
	// ----------------------------------------------------
	// calculte index and weight for Edge-blur
	edgeBlurIndex := []int{}
	edgeBlurWeight := []float64{}

	if conf.EdgeBlur > 0 {

		// bottom and upper edge
		bottomEdge := 0.0
		for _, v := range mapTable.Polar {
			if v > bottomEdge {
				bottomEdge = v
			}
		}
		upperEdge := bottomEdge
		for _, v := range mapTable.Polar {
			if v < upperEdge {
				upperEdge = v
			}
		}

		// count target pixel
		upperStart := upperEdge + conf.EdgeBlur
		bottomStart := bottomEdge - conf.EdgeBlur
		count := 0
		for _, v := range mapTable.Polar {
			if (v < upperStart) || (bottomStart < v) {
				count++
			}
		}
		L := count

		// calculate index and weight
		edgeBlurIndex = make([]int, L)
		edgeBlurWeight = make([]float64, L)
		count = 0
		for i, v := range mapTable.Polar {
			if v < upperStart {
				edgeBlurIndex[count] = i * NUMPIXEL
				edgeBlurWeight[count] = (v - upperEdge) / conf.EdgeBlur
				count++
			} else if bottomStart < v {
				edgeBlurIndex[count] = i * NUMPIXEL
				edgeBlurWeight[count] = (bottomEdge - v) / conf.EdgeBlur
				count++
			}
		}
	}

	// ----------------------------------------------------
	//      PREPARATION FOR OUTPUT
	// ----------------------------------------------------
	// generate index for storing in output video frame.
	// it's called storeIndex here.
	storeIndex := make([]int, L)
	for i := 0; i < L; i++ {
		storeIndex[i] = int(mapTable.Y[i]*mapTable.ProjW+mapTable.X[i]) * NUMPIXEL
	}

	// bind indexes for mapping
	mapper := &Mapper{
		PickupIndex:    pickupIndex,
		BilinearIndex:  [][]int{i1, i2, i3, i4},
		BilinearWeight: [][]float64{w1, w2, w3, w4},
		EdgeBlurIndex:  edgeBlurIndex,
		EdgeBlurWeight: edgeBlurWeight,
		StoreIndex:     storeIndex,
		ProjH:          int(mapTable.ProjH),
		ProjW:          int(mapTable.ProjW),
		NPickup:        len(pickupIndex),
		NStore:         len(storeIndex),
		NProj:          int(mapTable.ProjH) * int(mapTable.ProjW),
	}
	return mapper, nil
}
