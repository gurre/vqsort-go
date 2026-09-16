//go:build goexperiment.simd && arm64

package vqsort

import (
	"math/bits"
	"simd/archsimd"
)

// The 64-bit kernel is the 32-bit kernel at half the width: two lanes per
// vector instead of four. That halves the lanes moved per instruction, but most
// of the win over a scalar quicksort comes from the partition being branchless
// rather than from lane count, so the narrower kernel still pays.
//
// Two differences from the 32-bit kernel are forced by the instruction set:
// arm64 has no cross-lane reduction for 64-bit elements, so the comparison mask
// is read out lane by lane; and Int64x2 has no Min or Max at all, so the
// min/max scan is scalar rather than vectorized.

const (
	lanes64  = 2
	unroll64 = 4 * lanes64

	// networkKeys64 is the width of the generated sorting network.
	networkKeys64 = 16 * lanes64

	// baseCase64 is the largest range sorted without partitioning: the
	// network's width, which here coincides with the partition loop's floor.
	baseCase64 = networkKeys64

	// insertionFloor64 is the range below which the scalar loop still wins.
	insertionFloor64 = 8

	bufLen64 = 2 * unroll64
)

// maskBitsI64 reduces a 2-lane comparison mask to a 2-bit index.
//
// There is no ReduceSum for 64-bit elements on arm64 and no ToBits on any arm64
// mask, so the mask vector is read out one lane at a time. Each lane is all
// ones or all zeros, so the low bit carries the answer. Two independent UMOVs
// are cheaper here than manufacturing a 32-bit mask to reduce.
func maskBitsI64(m archsimd.Mask64x2) int {
	b := m.ToInt64x2().ToBits()
	return int(b.GetElem(0)&1) | int(b.GetElem(1)&1)<<1
}

func partitionVecI64(v, pivot archsimd.Int64x2) (archsimd.Int64x2, int) {
	m := maskBitsI64(pivot.Less(v))
	idx := archsimd.LoadUint8x16Array(&permTable64x2[m])
	lr := v.ToBits().ReshapeToUint8s().LookupOrZero(idx).ReshapeToUint64s().BitsToInt64()
	return lr, lanes64 - bits.OnesCount(uint(m))
}

func storeBothI64(keys []int64, wl, wr int, v archsimd.Int64x2) {
	v.StoreArray((*[lanes64]int64)(at(keys, wl, lanes64)))
	v.StoreArray((*[lanes64]int64)(at(keys, wr-lanes64, lanes64)))
}

func storeOneI64(keys []int64, wl, wr int, pivot, v archsimd.Int64x2) (int, int) {
	p, n := partitionVecI64(v, pivot)
	storeBothI64(keys, wl, wr, p)
	return wl + n, wr - (lanes64 - n)
}

// storeLastI64 places the drain's final vector by exact copy; see storeLastI32
// for why the paired stores cannot be used there.
func storeLastI64(keys []int64, wl, wr int, pivot, v archsimd.Int64x2) (int, int) {
	p, n := partitionVecI64(v, pivot)
	var packed [lanes64]int64
	p.StoreArray(&packed)
	copy(keys[wl:wl+n], packed[:n])
	copy(keys[wr-(lanes64-n):wr], packed[n:])
	return wl + n, wr - (lanes64 - n)
}

func storeFourI64(keys []int64, wl, wr int, pivot, v0, v1, v2, v3 archsimd.Int64x2) (int, int) {
	p0, n0 := partitionVecI64(v0, pivot)
	p1, n1 := partitionVecI64(v1, pivot)
	p2, n2 := partitionVecI64(v2, pivot)
	p3, n3 := partitionVecI64(v3, pivot)

	storeBothI64(keys, wl, wr, p0)
	wl, wr = wl+n0, wr-(lanes64-n0)
	storeBothI64(keys, wl, wr, p1)
	wl, wr = wl+n1, wr-(lanes64-n1)
	storeBothI64(keys, wl, wr, p2)
	wl, wr = wl+n2, wr-(lanes64-n2)
	storeBothI64(keys, wl, wr, p3)
	return wl + n3, wr - (lanes64 - n3)
}

func loadFourI64(keys []int64, i int) (a, b, c, d archsimd.Int64x2) {
	p := (*[unroll64]int64)(at(keys, i, unroll64))
	return archsimd.LoadInt64x2Array((*[lanes64]int64)(p[0:lanes64])),
		archsimd.LoadInt64x2Array((*[lanes64]int64)(p[lanes64 : 2*lanes64])),
		archsimd.LoadInt64x2Array((*[lanes64]int64)(p[2*lanes64 : 3*lanes64])),
		archsimd.LoadInt64x2Array((*[lanes64]int64)(p[3*lanes64 : 4*lanes64]))
}

func partitionRemainderI64(keys []int64, pivot int64, buf []int64) (wl, nBuf int) {
	for _, k := range keys {
		right := 0
		if k > pivot {
			right = 1
		}
		keys[wl] = k
		buf[nBuf] = k
		wl += 1 - right
		nBuf += right
	}
	return wl, nBuf
}

// partitionI64 is partitionI32 at two lanes; the capacity invariants are stated
// there and hold for any lane count, since they depend only on the unroll
// factor being four vectors.
func partitionI64(keys []int64, pivot int64, pv archsimd.Int64x2, buf *[bufLen64]int64) int {
	n := len(keys)
	rem := n % unroll64
	wl, nBuf := partitionRemainderI64(keys[:rem], pivot, buf[:])
	readL, readR, wr := rem, n, n

	l0, l1, l2, l3 := loadFourI64(keys, readL)
	readL += unroll64
	readR -= unroll64
	r0, r1, r2, r3 := loadFourI64(keys, readR)

	for readL < readR {
		var v0, v1, v2, v3 archsimd.Int64x2
		if wr-readR < readL-wl {
			readR -= unroll64
			v0, v1, v2, v3 = loadFourI64(keys, readR)
		} else {
			v0, v1, v2, v3 = loadFourI64(keys, readL)
			readL += unroll64
		}
		wl, wr = storeFourI64(keys, wl, wr, pv, v0, v1, v2, v3)
	}

	wl, wr = storeFourI64(keys, wl, wr, pv, l0, l1, l2, l3)
	wl, wr = storeOneI64(keys, wl, wr, pv, r0)
	wl, wr = storeOneI64(keys, wl, wr, pv, r1)
	wl, wr = storeOneI64(keys, wl, wr, pv, r2)
	wl, wr = storeLastI64(keys, wl, wr, pv, r3)

	copy(keys[wl:wr], buf[:nBuf])
	return wl
}

// sortBaseI64 sorts a range small enough to need no partitioning; see
// sortBaseI32 for why the network is used above a handful of keys.
func sortBaseI64(s []int64) {
	n := len(s)
	if n <= insertionFloor64 {
		insertionSort(s)
		return
	}
	var buf [networkKeys64]int64
	copy(buf[:], s)
	for i := n; i < networkKeys64; i++ {
		buf[i] = padI64
	}
	sortNetworkI64(&buf)
	copy(s, buf[:n])
}

// minMaxI64 is scalar: arm64 has no 64-bit integer Min or Max, and this runs
// only on the degenerate path where a partition put every key on the left.
func minMaxI64(s []int64) (int64, int64) {
	lo, hi := s[0], s[0]
	for _, v := range s[1:] {
		if v < lo {
			lo = v
		}
		if v > hi {
			hi = v
		}
	}
	return lo, hi
}

func choosePivotI64(s []int64, rng *sfc64) int64 {
	n := uint32(len(s))
	if len(s) < nintherFloor {
		return medianOf3(s[rng.bounded(n)], s[rng.bounded(n)], s[rng.bounded(n)])
	}
	var m [3]int64
	for i := range m {
		m[i] = medianOf3(s[rng.bounded(n)], s[rng.bounded(n)], s[rng.bounded(n)])
	}
	return medianOf3(m[0], m[1], m[2])
}

func recurseI64(keys []int64, pivot int64, buf *[bufLen64]int64, rng *sfc64, depth int) {
	for {
		if len(keys) <= baseCase64 {
			sortBaseI64(keys)
			return
		}
		if depth <= 0 {
			guardFired()
			heapSort(keys)
			return
		}
		depth--

		bound := partitionI64(keys, pivot, archsimd.BroadcastInt64x2(pivot), buf)
		left, right := keys[:bound], keys[bound:]

		if len(right) == 0 {
			lo, hi := minMaxI64(keys)
			if lo == hi {
				return
			}
			pivot = lo
			continue
		}

		if len(left) <= len(right) {
			recurseI64(left, choosePivotI64(left, rng), buf, rng, depth)
			keys, pivot = right, choosePivotI64(right, rng)
		} else {
			recurseI64(right, choosePivotI64(right, rng), buf, rng, depth)
			keys, pivot = left, choosePivotI64(left, rng)
		}
	}
}

func sortInt64NEON(s []int64) {
	if len(s) <= baseCase64 {
		sortBaseI64(s)
		return
	}
	var buf [bufLen64]int64
	rng := newSFC64(seedCounter.Add(1))
	recurseI64(s, choosePivotI64(s, &rng), &buf, &rng, maxDepth(len(s)))
}
