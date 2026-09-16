//go:build goexperiment.simd && arm64

package vqsort

import (
	"math/rand"
	"slices"
	"testing"

	"simd/archsimd"
)

// TestPermTable32x4 checks the table drives a permutation, not just a plausible
// set of bytes. An out-of-range index is silently turned into a zero byte by
// LookupOrZero, so a malformed row corrupts data without faulting.
func TestPermTable32x4(t *testing.T) {
	for m, row := range permTable32x4 {
		var seen [lanes32]bool
		for j := 0; j < lanes32; j++ {
			lane := row[4*j] / 4
			if row[4*j]%4 != 0 || lane >= lanes32 {
				t.Fatalf("mask %d lane %d: byte index %d is not a lane start", m, j, row[4*j])
			}
			for b := 0; b < 4; b++ {
				if got, want := row[4*j+b], lane*4+uint8(b); got != want {
					t.Fatalf("mask %d lane %d byte %d: got %d, want %d", m, j, b, got, want)
				}
			}
			if seen[lane] {
				t.Fatalf("mask %d: lane %d appears twice", m, lane)
			}
			seen[lane] = true
		}
	}
}

// TestPartitionVec32Exhaustive covers every one of the 16 possible comparison
// masks, asserting both the packing and the order preservation within each
// group that the partition loop's correctness depends on.
func TestPartitionVec32Exhaustive(t *testing.T) {
	const pivot = 50
	for m := 0; m < 16; m++ {
		var in [lanes32]int32
		var wantLeft, wantRight []int32
		for lane := 0; lane < lanes32; lane++ {
			if m>>lane&1 == 1 {
				in[lane] = int32(100 + lane) // > pivot
				wantRight = append(wantRight, in[lane])
			} else {
				in[lane] = int32(lane) // <= pivot
				wantLeft = append(wantLeft, in[lane])
			}
		}

		v := archsimd.LoadInt32x4Array(&in)
		packed, n := partitionVecI32(v, archsimd.BroadcastInt32x4(pivot))
		var got [lanes32]int32
		packed.StoreArray(&got)

		if n != len(wantLeft) {
			t.Fatalf("mask %04b: got %d left lanes, want %d", m, n, len(wantLeft))
		}
		if !slices.Equal(got[:n], wantLeft) {
			t.Fatalf("mask %04b: left lanes %v, want %v", m, got[:n], wantLeft)
		}
		if !slices.Equal(got[n:], wantRight) {
			t.Fatalf("mask %04b: right lanes %v, want %v", m, got[n:], wantRight)
		}
	}
}

// checkPartition asserts the postcondition of partitionI32.
func checkPartition(t *testing.T, before []int32, keys []int32, pivot int32, bound int) {
	t.Helper()
	if bound < 1 || bound > len(keys) {
		t.Fatalf("bound %d out of range for %d keys", bound, len(keys))
	}
	for i, k := range keys[:bound] {
		if k > pivot {
			t.Fatalf("key %d at %d is greater than pivot %d but sits on the left", k, i, pivot)
		}
	}
	for i, k := range keys[bound:] {
		if k <= pivot {
			t.Fatalf("key %d at %d is not greater than pivot %d but sits on the right", k, bound+i, pivot)
		}
	}
	wantSorted, gotSorted := slices.Clone(before), slices.Clone(keys)
	slices.Sort(wantSorted)
	slices.Sort(gotSorted)
	if !slices.Equal(wantSorted, gotSorted) {
		t.Fatalf("partition did not preserve the multiset of keys")
	}
}

func runPartition(t *testing.T, keys []int32, pivot int32) int {
	t.Helper()
	var buf [bufLen32]int32
	return partitionI32(keys, pivot, archsimd.BroadcastInt32x4(pivot), &buf)
}

// TestPartition32Sizes sweeps every length and every remainder residue the
// partition loop can see, with pivots drawn from the data so the left side is
// never empty.
func TestPartition32Sizes(t *testing.T) {
	r := rand.New(rand.NewSource(7))
	for n := baseCase32 + 1; n <= 600; n++ {
		for trial := 0; trial < 4; trial++ {
			keys := make([]int32, n)
			for i := range keys {
				keys[i] = int32(r.Intn(1000)) - 500
			}
			pivot := keys[r.Intn(n)]
			before := slices.Clone(keys)
			bound := runPartition(t, keys, pivot)
			checkPartition(t, before, keys, pivot, bound)
		}
	}
}

// TestPartition32RemainderSpill targets the case that the paired full-width
// stores get wrong: the remainder loop spills between 1 and lanes32-1 keys, so
// the drain's last store faces a free region wider than one vector but narrower
// than two.
func TestPartition32RemainderSpill(t *testing.T) {
	const pivot = 0
	for spill := 1; spill < lanes32; spill++ {
		for _, blocks := range []int{2, 3, 5} {
			rem := spill + 2 // a few left keys ahead of the spilled ones
			n := blocks*unroll32 + rem
			keys := make([]int32, n)
			// The first rem keys decide the spill: exactly `spill` of them
			// are greater than the pivot.
			for i := 0; i < rem; i++ {
				if i < spill {
					keys[i] = 10
				} else {
					keys[i] = -10
				}
			}
			for i := rem; i < n; i++ {
				if i%3 == 0 {
					keys[i] = 10
				} else {
					keys[i] = -10
				}
			}
			before := slices.Clone(keys)
			bound := runPartition(t, keys, pivot)
			checkPartition(t, before, keys, pivot, bound)
		}
	}
}

// TestPartition32Degenerate covers the pivots that make one side as large as
// possible, which is where the capacity invariants are tightest.
func TestPartition32Degenerate(t *testing.T) {
	for _, n := range []int{33, 48, 64, 65, 79, 128, 129, 1000} {
		t.Run("allleft", func(t *testing.T) {
			keys := make([]int32, n)
			for i := range keys {
				keys[i] = 5
			}
			before := slices.Clone(keys)
			bound := runPartition(t, keys, 5)
			if bound != n {
				t.Fatalf("n=%d: all keys equal the pivot, want bound %d, got %d", n, n, bound)
			}
			checkPartition(t, before, keys, 5, bound)
		})
		t.Run("oneleft", func(t *testing.T) {
			keys := make([]int32, n)
			for i := range keys {
				keys[i] = 100
			}
			keys[n/2] = 1
			before := slices.Clone(keys)
			bound := runPartition(t, keys, 1)
			if bound != 1 {
				t.Fatalf("n=%d: one key equals the pivot, want bound 1, got %d", n, bound)
			}
			checkPartition(t, before, keys, 1, bound)
		})
	}
}

// TestMinMax32 checks the scan used by the degenerate-partition path, over
// every length class of its vector body and scalar tail.
func TestMinMax32(t *testing.T) {
	r := rand.New(rand.NewSource(11))
	for n := 1; n <= 200; n++ {
		s := make([]int32, n)
		for i := range s {
			s[i] = int32(r.Intn(2000)) - 1000
		}
		lo, hi := minMaxI32(s)
		wantLo, wantHi := slices.Min(s), slices.Max(s)
		if lo != wantLo || hi != wantHi {
			t.Fatalf("n=%d: got (%d,%d), want (%d,%d)", n, lo, hi, wantLo, wantHi)
		}
	}
}

// TestChoosePivot32IsPresentKey checks the property the degenerate-partition
// logic depends on: the pivot is always a key from the range, so the left side
// of a partition is never empty.
func TestChoosePivot32IsPresentKey(t *testing.T) {
	r := rand.New(rand.NewSource(13))
	rng := newSFC64(99)
	for trial := 0; trial < 2000; trial++ {
		n := 1 + r.Intn(300)
		s := make([]int32, n)
		for i := range s {
			s[i] = int32(r.Intn(50))
		}
		p := choosePivotI32(s, &rng)
		if !slices.Contains(s, p) {
			t.Fatalf("pivot %d is not one of the keys", p)
		}
	}
}

// TestDepthGuardSorts checks the fallback itself, by entering the recursion with
// the budget already spent.
func TestDepthGuardSorts(t *testing.T) {
	r := rand.New(rand.NewSource(3))
	for _, n := range []int{1, 2, 33, 100, 5000} {
		s := make([]int32, n)
		for i := range s {
			s[i] = int32(r.Intn(1000))
		}
		before := slices.Clone(s)
		trips := depthGuardTrips.Load()

		var buf [bufLen32]int32
		rng := newSFC64(1)
		recurseI32(s, 0, &buf, &rng, 0)

		checkSortedPermutation(t, before, s)
		if n > baseCase32 && depthGuardTrips.Load() == trips {
			t.Errorf("n=%d: exhausted budget did not register a depth guard trip", n)
		}
	}
}

// The sorting networks are generated straight-line code with no branches, so a
// wrong comparator or a wrong mask corrupts only some inputs and leaves others
// perfectly sorted. These tests lean on that: ties and 0/1 inputs are where a
// miswired network shows up, per the 0-1 principle, which says a data-oblivious
// network sorts every input exactly when it sorts every 0/1 input.

func TestSortNetworkI32(t *testing.T) {
	r := rand.New(rand.NewSource(21))
	var buf [networkKeys32]int32
	for trial := 0; trial < 30000; trial++ {
		fillNetworkCase(r, buf[:], trial, func(v int) int32 { return int32(v) })
		want := buf
		slices.Sort(want[:])
		got := buf
		sortNetworkI32(&got)
		if got != want {
			t.Fatalf("trial %d\n in:   %v\n got:  %v\n want: %v", trial, buf, got, want)
		}
	}
}

func TestSortNetworkI64(t *testing.T) {
	r := rand.New(rand.NewSource(22))
	var buf [networkKeys64]int64
	for trial := 0; trial < 30000; trial++ {
		fillNetworkCase(r, buf[:], trial, func(v int) int64 { return int64(v) })
		want := buf
		slices.Sort(want[:])
		got := buf
		sortNetworkI64(&got)
		if got != want {
			t.Fatalf("trial %d\n in:   %v\n got:  %v\n want: %v", trial, buf, got, want)
		}
	}
}

func fillNetworkCase[T key](r *rand.Rand, s []T, trial int, conv func(int) T) {
	switch trial % 5 {
	case 0: // distinct random
		for i := range s {
			s[i] = conv(r.Intn(1 << 20))
		}
	case 1: // heavy ties
		for i := range s {
			s[i] = conv(r.Intn(3))
		}
	case 2: // sorted, then reversed half the time
		for i := range s {
			s[i] = conv(i)
		}
		if trial%10 == 2 {
			slices.Reverse(s)
		}
	case 3: // 0/1 inputs, the 0-1 principle's own test set
		for i := range s {
			s[i] = conv(r.Intn(2))
		}
	case 4: // a single one among zeros, walked across every position
		for i := range s {
			s[i] = conv(0)
		}
		s[trial/5%len(s)] = conv(1)
	}
}

// TestMinMaxKeys checks the compare-exchange primitives the networks are built
// from. IfElse keeps the receiver where the mask is true, which reads backwards,
// and swapping the operands silently turns min into max.
func TestMinMaxKeys(t *testing.T) {
	vals := []int64{-1 << 62, -7, -1, 0, 1, 7, 1 << 62}
	for _, a0 := range vals {
		for _, b0 := range vals {
			va := archsimd.LoadInt64x2Array(&[2]int64{a0, b0})
			vb := archsimd.LoadInt64x2Array(&[2]int64{b0, a0})
			var gotMin, gotMax [2]int64
			minKeyI64(va, vb).StoreArray(&gotMin)
			maxKeyI64(va, vb).StoreArray(&gotMax)
			if gotMin != [2]int64{min(a0, b0), min(b0, a0)} {
				t.Fatalf("minKeyI64(%d,%d) = %v", a0, b0, gotMin)
			}
			if gotMax != [2]int64{max(a0, b0), max(b0, a0)} {
				t.Fatalf("maxKeyI64(%d,%d) = %v", a0, b0, gotMax)
			}
		}
	}
}

// TestSortBasePadding checks that the padding used to fill the network buffer
// never survives into the output, at every length the base case can see.
func TestSortBasePadding(t *testing.T) {
	r := rand.New(rand.NewSource(23))
	for n := 0; n <= baseCase32; n++ {
		s := make([]int32, n)
		for i := range s {
			// Include the pad value itself as a real key: it must be kept.
			if r.Intn(6) == 0 {
				s[i] = padI32
			} else {
				s[i] = int32(r.Intn(100))
			}
		}
		before := slices.Clone(s)
		sortBaseI32(s)
		checkSortedPermutation(t, before, s)
	}
	for n := 0; n <= baseCase64; n++ {
		s := make([]int64, n)
		for i := range s {
			if r.Intn(6) == 0 {
				s[i] = padI64
			} else {
				s[i] = int64(r.Intn(100))
			}
		}
		before := slices.Clone(s)
		sortBaseI64(s)
		checkSortedPermutation(t, before, s)
	}
}
