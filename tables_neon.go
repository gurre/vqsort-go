//go:build goexperiment.simd && arm64

package vqsort

// permTable32x4 drives the partition compress for 4-lane 32-bit vectors.
//
// arm64 has no lane-granularity permute and no compress, only the byte-wise
// table lookup TBL (archsimd's Uint8x16.LookupOrZero). Row m, indexed by the
// bitmask of lanes that belong on the right of the pivot, holds the byte
// indices that gather the left lanes into the low end of the vector and the
// right lanes into the high end, preserving the relative order of each group.
//
// Each of the four lanes contributes four consecutive bytes, so a lane index
// l becomes bytes 4l, 4l+1, 4l+2, 4l+3.
var permTable32x4 [16][16]uint8

// laneWeights32x4 converts a 4-lane comparison mask to a bitmask: masking these
// weights by the comparison and summing across lanes yields one bit per lane.
var laneWeights32x4 = [4]uint32{1, 2, 4, 8}

func init() {
	for m := range permTable32x4 {
		var order [4]uint8
		n := 0
		for lane := 0; lane < 4; lane++ {
			if m>>lane&1 == 0 { // left: key <= pivot
				order[n] = uint8(lane)
				n++
			}
		}
		for lane := 0; lane < 4; lane++ {
			if m>>lane&1 == 1 { // right: key > pivot
				order[n] = uint8(lane)
				n++
			}
		}
		for j, lane := range order {
			for b := 0; b < 4; b++ {
				permTable32x4[m][4*j+b] = lane*4 + uint8(b)
			}
		}
	}
}

// permTable64x2 is the same idea as permTable32x4 for 2-lane 64-bit vectors,
// where a lane spans eight bytes and there are only four possible masks.
var permTable64x2 [4][16]uint8

func init() {
	for m := range permTable64x2 {
		var order [2]uint8
		n := 0
		for lane := 0; lane < 2; lane++ {
			if m>>lane&1 == 0 {
				order[n] = uint8(lane)
				n++
			}
		}
		for lane := 0; lane < 2; lane++ {
			if m>>lane&1 == 1 {
				order[n] = uint8(lane)
				n++
			}
		}
		for j, lane := range order {
			for b := 0; b < 8; b++ {
				permTable64x2[m][8*j+b] = lane*8 + uint8(b)
			}
		}
	}
}
