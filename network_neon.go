//go:build goexperiment.simd && arm64

package vqsort

import (
	"math"
	"simd/archsimd"
)

// Constants for the generated sorting networks in network_*_neon.go.
//
// These are hand-written rather than generated because the type rewrite that
// derives the float kernels from the integer ones must not touch them: the
// direction masks stay integer vectors whatever the key type, since a mask is
// built from a bit pattern, and Mask32x4 is the mask type for every 32-bit lane
// vector regardless of whether the keys are integers or floats.
//
// They are returned by functions rather than held in package-level vector
// variables because archsimd documents that keeping a vector in a global, or
// taking its address, defeats register allocation. Each of these is a load from
// read-only data that the compiler hoists out of the network.

// padI32 and padF32 fill a base-case buffer beyond the keys actually present,
// so the padding sorts to the end and is discarded. For floats this is +Inf,
// which is the largest value once the NaN prepass has run - not the largest
// finite value, which real keys could equal.
const padI32 int32 = math.MaxInt32

var padF32 = float32(math.Inf(1))

// Lane orders for the two shuffle distances a 4-lane bitonic merge needs,
// expressed as byte indices because arm64's only data-dependent permute is the
// byte-wise table lookup.
var (
	laneSwap1Bytes = [16]uint8{4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11}
	laneSwap2Bytes = [16]uint8{8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7}
)

func laneSwap32D1() archsimd.Uint8x16 { return archsimd.LoadUint8x16Array(&laneSwap1Bytes) }
func laneSwap32D2() archsimd.Uint8x16 { return archsimd.LoadUint8x16Array(&laneSwap2Bytes) }

// Direction masks for the stages whose sort direction is decided by the lane
// rather than by the row: true where that lane's comparison is ascending.
var (
	dirK16Bits = [4]int32{-1, 0, -1, 0}
	dirK32Bits = [4]int32{-1, -1, 0, 0}
)

func dirMask32K16() archsimd.Mask32x4 { return archsimd.LoadInt32x4Array(&dirK16Bits).ToMask() }
func dirMask32K32() archsimd.Mask32x4 { return archsimd.LoadInt32x4Array(&dirK32Bits).ToMask() }

// Selection masks for the lane-distance stages: true where the lane keeps the
// minimum of its pair, which is where it is both the lower lane of the pair and
// sorting ascending, or neither.
var (
	keepK32D1Bits = [4]int32{-1, 0, 0, -1}
	keepK64D1Bits = [4]int32{-1, 0, -1, 0}
	keepK64D2Bits = [4]int32{-1, -1, 0, 0}
)

func keepMin32K32D1() archsimd.Mask32x4 { return archsimd.LoadInt32x4Array(&keepK32D1Bits).ToMask() }
func keepMin32K64D1() archsimd.Mask32x4 { return archsimd.LoadInt32x4Array(&keepK64D1Bits).ToMask() }
func keepMin32K64D2() archsimd.Mask32x4 { return archsimd.LoadInt32x4Array(&keepK64D2Bits).ToMask() }

// The 2-lane network needs a single shuffle distance, one direction mask, and
// one selection mask. With only two lanes per vector the logical index of lane
// j row r is j*16+r, so the direction at k=16 is decided by the lane and at
// k=32 every comparison is ascending.

const padI64 int64 = math.MaxInt64

var padF64 = math.Inf(1)

var laneSwap64D1Bytes = [16]uint8{8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7}

func laneSwap64D1() archsimd.Uint8x16 { return archsimd.LoadUint8x16Array(&laneSwap64D1Bytes) }

var (
	dir64K16Bits    = [2]int64{-1, 0}
	keep64K32D1Bits = [2]int64{-1, 0}
)

func dirMask64K16() archsimd.Mask64x2 { return archsimd.LoadInt64x2Array(&dir64K16Bits).ToMask() }
func keepMin64K32D1() archsimd.Mask64x2 {
	return archsimd.LoadInt64x2Array(&keep64K32D1Bits).ToMask()
}

// Compare-exchange primitives for the networks.
//
// Int64x2 has no Min or Max on arm64 - not emulated, simply absent - so the
// 64-bit integer pair is synthesized from the native 64-bit compare. Note that
// IfElse keeps the receiver where the mask is true, which reads backwards:
// getting the receiver and argument the wrong way round silently swaps min for
// max, and since every comparator in the network uses these, that would corrupt
// only certain inputs. TestMinMaxKeys checks them exhaustively.
func minKeyI32(a, b archsimd.Int32x4) archsimd.Int32x4 { return a.Min(b) }
func maxKeyI32(a, b archsimd.Int32x4) archsimd.Int32x4 { return a.Max(b) }

func minKeyF32(a, b archsimd.Float32x4) archsimd.Float32x4 { return a.Min(b) }
func maxKeyF32(a, b archsimd.Float32x4) archsimd.Float32x4 { return a.Max(b) }

func minKeyI64(a, b archsimd.Int64x2) archsimd.Int64x2 { return b.IfElse(a.Greater(b), a) }
func maxKeyI64(a, b archsimd.Int64x2) archsimd.Int64x2 { return a.IfElse(a.Greater(b), b) }

func minKeyF64(a, b archsimd.Float64x2) archsimd.Float64x2 { return a.Min(b) }
func maxKeyF64(a, b archsimd.Float64x2) archsimd.Float64x2 { return a.Max(b) }
