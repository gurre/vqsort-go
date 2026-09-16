//go:build goexperiment.simd && arm64

package vqsort

import (
	"math/bits"
	"simd/archsimd"
)

const (
	lanes32 = 4 // int32 lanes in a 128-bit vector

	// unroll32 is the number of keys the partition loop consumes per step.
	// The Go compiler does not unroll loops, so the four vector steps are
	// written out.
	unroll32 = 4 * lanes32

	// networkKeys32 is the width of the generated sorting network, which sorts
	// a fixed number of keys entirely in registers.
	networkKeys32 = 16 * lanes32

	// baseCase32 is the largest range sorted without partitioning.
	//
	// It is the network's width: the network costs the same whatever the range
	// actually holds, so the threshold wants to be as high as the network goes
	// and no higher. Partitioning further down instead - with a single-vector
	// loop whose floor is 2*lanes32 - was tried and is slower at every
	// threshold, because a short range pays a whole partition call's fixed cost
	// (pivot sampling, preload, drain) for too few keys.
	baseCase32 = networkKeys32

	// insertionFloor32 is the range below which the scalar loop still wins: the
	// network's cost is flat, so for a handful of keys it is pure overhead.
	insertionFloor32 = 8

	// bufLen32 holds the right-hand spill from the remainder loop, which is
	// at most unroll32-1 keys.
	bufLen32 = 2 * unroll32
)

// partitionVecI32 returns v with the lanes that are <= pivot packed into the low
// end and the lanes that are > pivot packed into the high end, preserving the
// relative order within each group, together with the number of low lanes.
//
// arm64 has neither a compress instruction nor a lane-granularity permute, so
// the comparison mask is reduced to a 4-bit index and the packing is done by a
// byte-wise table lookup.
func partitionVecI32(v, pivot archsimd.Int32x4) (archsimd.Int32x4, int) {
	right := pivot.Less(v) // lanes strictly greater than the pivot
	m := archsimd.LoadUint32x4Array(&laneWeights32x4).Masked(right).ReduceSum()
	idx := archsimd.LoadUint8x16Array(&permTable32x4[m])
	lr := v.ToBits().ReshapeToUint8s().LookupOrZero(idx).ReshapeToUint32s().BitsToInt32()
	return lr, lanes32 - bits.OnesCount32(m)
}

// storeBothI32 writes the whole packed vector to both ends of the free region.
// The low lanes land correctly at wl and the high lanes land correctly at
// wr-lanes32; the remaining lanes of each store are garbage that falls inside
// the free region and is overwritten by a later store.
//
// This requires the free region wr-wl to hold either exactly one vector, where
// both stores address the same slots, or at least two, where the second store
// cannot reach the slots the first one filled. Every store but the last of the
// drain satisfies that; see storeLastI32.
func storeBothI32(keys []int32, wl, wr int, v archsimd.Int32x4) {
	v.StoreArray((*[lanes32]int32)(at(keys, wl, lanes32)))
	v.StoreArray((*[lanes32]int32)(at(keys, wr-lanes32, lanes32)))
}

// storeOneI32 partitions and stores a single vector.
func storeOneI32(keys []int32, wl, wr int, pivot, v archsimd.Int32x4) (int, int) {
	p, n := partitionVecI32(v, pivot)
	storeBothI32(keys, wl, wr, p)
	return wl + n, wr - (lanes32 - n)
}

// storeLastI32 places the final vector of the drain by exact copy.
//
// When the drain reaches its last vector the free region holds
// lanes32 + nBuf keys, where nBuf is what the remainder loop spilled. If that
// is strictly between one and two vectors wide - nBuf from 1 to lanes32-1 - the
// right-hand full-width store reaches back over slots the left-hand store has
// already filled and duplicates keys. Copying only the valid lanes into their
// two destinations is correct for any free width of at least one vector, and it
// runs once per partition call rather than once per vector.
func storeLastI32(keys []int32, wl, wr int, pivot, v archsimd.Int32x4) (int, int) {
	p, n := partitionVecI32(v, pivot)
	var packed [lanes32]int32
	p.StoreArray(&packed)
	copy(keys[wl:wl+n], packed[:n])
	copy(keys[wr-(lanes32-n):wr], packed[n:])
	return wl + n, wr - (lanes32 - n)
}

// storeFourI32 partitions and stores four vectors.
//
// All four packed vectors are computed before the first store. Each lane count
// comes from a cross-lane reduction, and it gates the next store's address, so
// computing them interleaved with the stores would put that reduction on a
// serial dependency chain four times per loop step. Computed up front they are
// mutually independent and only the cheap scalar counts are threaded through.
func storeFourI32(keys []int32, wl, wr int, pivot, v0, v1, v2, v3 archsimd.Int32x4) (int, int) {
	p0, n0 := partitionVecI32(v0, pivot)
	p1, n1 := partitionVecI32(v1, pivot)
	p2, n2 := partitionVecI32(v2, pivot)
	p3, n3 := partitionVecI32(v3, pivot)

	storeBothI32(keys, wl, wr, p0)
	wl, wr = wl+n0, wr-(lanes32-n0)
	storeBothI32(keys, wl, wr, p1)
	wl, wr = wl+n1, wr-(lanes32-n1)
	storeBothI32(keys, wl, wr, p2)
	wl, wr = wl+n2, wr-(lanes32-n2)
	storeBothI32(keys, wl, wr, p3)
	return wl + n3, wr - (lanes32 - n3)
}

// loadFourI32 reads one unrolled block. The three-index slice and the array
// pointer conversion concentrate the bounds check to one for all 16 keys.
func loadFourI32(keys []int32, i int) (a, b, c, d archsimd.Int32x4) {
	p := (*[unroll32]int32)(at(keys, i, unroll32))
	return archsimd.LoadInt32x4Array((*[lanes32]int32)(p[0:lanes32])),
		archsimd.LoadInt32x4Array((*[lanes32]int32)(p[lanes32 : 2*lanes32])),
		archsimd.LoadInt32x4Array((*[lanes32]int32)(p[2*lanes32 : 3*lanes32])),
		archsimd.LoadInt32x4Array((*[lanes32]int32)(p[3*lanes32 : 4*lanes32]))
}

// partitionRemainderI32 partitions the keys that do not fill a whole unrolled
// block, compacting the left keys in place and spilling the right keys to buf.
// It is branchless: both stores always happen and only the cursors move.
func partitionRemainderI32(keys []int32, pivot int32, buf []int32) (wl, nBuf int) {
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

// partitionI32 reorders keys so that every key <= pivot precedes every key
// > pivot, and returns the number of keys in the left part. pv must be pivot
// broadcast to all lanes. len(keys) must be greater than baseCase32.
//
// The loop reads from whichever side has less free space and writes to both
// ends, so the written region never overtakes the unread region. With
// capL = readL-wl and capR = wr-readR, the preload establishes
// capL+capR = 8*lanes32+nBuf, and every step both consumes 4*lanes32 and reads
// 4*lanes32, preserving it. Reading from the smaller side leaves both sides at
// least 4*lanes32 free, so each of the four stores that follows has room.
func partitionI32(keys []int32, pivot int32, pv archsimd.Int32x4, buf *[bufLen32]int32) int {
	n := len(keys)
	rem := n % unroll32
	wl, nBuf := partitionRemainderI32(keys[:rem], pivot, buf[:])
	readL, readR, wr := rem, n, n

	l0, l1, l2, l3 := loadFourI32(keys, readL)
	readL += unroll32
	readR -= unroll32
	r0, r1, r2, r3 := loadFourI32(keys, readR)

	for readL < readR {
		var v0, v1, v2, v3 archsimd.Int32x4
		if wr-readR < readL-wl {
			readR -= unroll32
			v0, v1, v2, v3 = loadFourI32(keys, readR)
		} else {
			v0, v1, v2, v3 = loadFourI32(keys, readL)
			readL += unroll32
		}
		wl, wr = storeFourI32(keys, wl, wr, pv, v0, v1, v2, v3)
	}

	// readL == readR, so the free region is one contiguous interval wide
	// enough for the eight preloaded vectors. It shrinks to nBuf, so only the
	// final store can meet a free region too narrow for the paired stores.
	wl, wr = storeFourI32(keys, wl, wr, pv, l0, l1, l2, l3)
	wl, wr = storeOneI32(keys, wl, wr, pv, r0)
	wl, wr = storeOneI32(keys, wl, wr, pv, r1)
	wl, wr = storeOneI32(keys, wl, wr, pv, r2)
	wl, wr = storeLastI32(keys, wl, wr, pv, r3)

	// What is left of the free region is exactly the remainder spill, all of
	// which belongs on the right.
	copy(keys[wl:wr], buf[:nBuf])
	return wl
}

// sortBaseI32 sorts a range small enough to need no partitioning.
//
// Above a handful of keys this is the branch-free sorting network, which sorts
// its full width whatever the range holds; the keys are copied into a padded
// buffer so the padding sorts to the end and is discarded. Insertion sort, whose
// cost is quadratic, is left to handle only the shortest ranges, where the
// network's flat cost would not be repaid.
func sortBaseI32(s []int32) {
	n := len(s)
	if n <= insertionFloor32 {
		insertionSort(s)
		return
	}
	var buf [networkKeys32]int32
	copy(buf[:], s)
	for i := n; i < networkKeys32; i++ {
		buf[i] = padI32
	}
	sortNetworkI32(&buf)
	copy(s, buf[:n])
}

// minMaxI32 returns the smallest and largest key in s, which must not be empty.
func minMaxI32(s []int32) (int32, int32) {
	lo, hi := s[0], s[0]
	i := 0
	if len(s) >= lanes32 {
		vlo := archsimd.LoadInt32x4Array((*[lanes32]int32)(s[0:lanes32:lanes32]))
		vhi := vlo
		for i = lanes32; i+lanes32 <= len(s); i += lanes32 {
			v := archsimd.LoadInt32x4Array((*[lanes32]int32)(s[i : i+lanes32 : i+lanes32]))
			vlo = vlo.Min(v)
			vhi = vhi.Max(v)
		}
		lo, hi = vlo.ReduceMin(), vhi.ReduceMax()
	}
	for ; i < len(s); i++ {
		if s[i] < lo {
			lo = s[i]
		}
		if s[i] > hi {
			hi = s[i]
		}
	}
	return lo, hi
}

// choosePivotI32 returns the median of three medians of three sampled keys. The
// result is always a key present in s, which the degenerate-partition handling
// in recurseI32 relies on.
func choosePivotI32(s []int32, rng *sfc64) int32 {
	n := uint32(len(s))
	if len(s) < nintherFloor {
		return medianOf3(s[rng.bounded(n)], s[rng.bounded(n)], s[rng.bounded(n)])
	}
	var m [3]int32
	for i := range m {
		m[i] = medianOf3(s[rng.bounded(n)], s[rng.bounded(n)], s[rng.bounded(n)])
	}
	return medianOf3(m[0], m[1], m[2])
}

func recurseI32(keys []int32, pivot int32, buf *[bufLen32]int32, rng *sfc64, depth int) {
	for {
		if len(keys) <= baseCase32 {
			sortBaseI32(keys)
			return
		}
		if depth <= 0 {
			guardFired()
			heapSort(keys)
			return
		}
		depth--

		bound := partitionI32(keys, pivot, archsimd.BroadcastInt32x4(pivot), buf)
		left, right := keys[:bound], keys[bound:]

		if len(right) == 0 {
			// Every key is <= pivot and the pivot is one of them, so the
			// range is sorted exactly when all keys are equal. Otherwise
			// the minimum splits off at least one key.
			lo, hi := minMaxI32(keys)
			if lo == hi {
				return
			}
			pivot = lo
			continue
		}

		// Recurse into the smaller side and loop on the larger, so stack
		// depth stays logarithmic even when partitions are lopsided.
		if len(left) <= len(right) {
			recurseI32(left, choosePivotI32(left, rng), buf, rng, depth)
			keys, pivot = right, choosePivotI32(right, rng)
		} else {
			recurseI32(right, choosePivotI32(right, rng), buf, rng, depth)
			keys, pivot = left, choosePivotI32(left, rng)
		}
	}
}

func sortInt32NEON(s []int32) {
	if len(s) <= baseCase32 {
		sortBaseI32(s)
		return
	}
	var buf [bufLen32]int32
	rng := newSFC64(seedCounter.Add(1))
	recurseI32(s, choosePivotI32(s, &rng), &buf, &rng, maxDepth(len(s)))
}
