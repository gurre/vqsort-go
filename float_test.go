package vqsort

import (
	"math"
	"math/rand"
	"slices"
	"testing"
)

// Float keys are the only place vqsort's ordering could diverge from
// slices.Sort, because Less is not a total order once NaNs are present and the
// vector Min and Max do not agree with it. The kernel moves NaNs aside first;
// these tests check that the result is indistinguishable from slices.Sort,
// down to preserved NaN payloads.

func specialFloat64s() []float64 {
	return []float64{
		math.NaN(),
		math.Float64frombits(0x7FF8000000000001), // NaN, distinct payload
		math.Float64frombits(0xFFF8000000000001), // negative NaN
		math.Inf(-1), math.Inf(1),
		0, math.Copysign(0, -1),
		math.SmallestNonzeroFloat64, -math.SmallestNonzeroFloat64,
		math.MaxFloat64, -math.MaxFloat64,
		1, -1, 0.5, -0.5, 1e300, -1e300,
	}
}

func specialFloat32s() []float32 {
	return []float32{
		float32(math.NaN()),
		math.Float32frombits(0x7FC00001),
		math.Float32frombits(0xFFC00001),
		float32(math.Inf(-1)), float32(math.Inf(1)),
		0, float32(math.Copysign(0, -1)),
		math.SmallestNonzeroFloat32, -math.SmallestNonzeroFloat32,
		math.MaxFloat32, -math.MaxFloat32,
		1, -1, 0.5, -0.5,
	}
}

// checkFloatOrder asserts the ordering slices.Sort documents: NaNs first, then
// everything else ascending.
func checkFloatOrder[T float32 | float64](t *testing.T, s []T) {
	t.Helper()
	nans := 0
	for nans < len(s) && s[nans] != s[nans] {
		nans++
	}
	for i := nans; i < len(s); i++ {
		if s[i] != s[i] {
			t.Fatalf("NaN at index %d follows a non-NaN", i)
		}
		if i > nans && s[i] < s[i-1] {
			t.Fatalf("not sorted at %d: %v > %v", i, s[i-1], s[i])
		}
	}
}

// bitCounts fingerprints a float slice by raw bit pattern, so that a lost NaN
// payload or a swapped -0.0 for +0.0 is visible. Comparing values would not
// catch either: NaN != NaN, and -0.0 == +0.0.
func bitCounts32(s []float32) map[uint32]int {
	m := make(map[uint32]int, len(s))
	for _, v := range s {
		m[math.Float32bits(v)]++
	}
	return m
}

func bitCounts64(s []float64) map[uint64]int {
	m := make(map[uint64]int, len(s))
	for _, v := range s {
		m[math.Float64bits(v)]++
	}
	return m
}

func TestSortFloat32Specials(t *testing.T) {
	for _, target := range availableTargets() {
		t.Run(target.String(), func(t *testing.T) {
			defer forceTarget(t, target)()
			r := rand.New(rand.NewSource(5))
			specials := specialFloat32s()
			for _, n := range sizes(testing.Short()) {
				for _, density := range []int{1, 3, 20} {
					s := make([]float32, n)
					for i := range s {
						if r.Intn(density) == 0 {
							s[i] = specials[r.Intn(len(specials))]
						} else {
							s[i] = float32(r.NormFloat64() * 1e3)
						}
					}
					want := slices.Clone(s)
					slices.Sort(want)
					before := bitCounts32(s)

					Sort(s)
					checkFloatOrder(t, s)
					got := bitCounts32(s)
					for k, v := range before {
						if got[k] != v {
							t.Fatalf("n=%d: bit pattern %#x count %d, want %d", n, k, got[k], v)
						}
					}
					// NaNs are equal under cmp.Less, so their order among
					// themselves is unspecified; compare the rest exactly.
					nans := 0
					for nans < len(s) && s[nans] != s[nans] {
						nans++
					}
					if !slices.Equal(s[nans:], want[nans:]) {
						t.Fatalf("n=%d: non-NaN part differs from slices.Sort", n)
					}
				}
			}
		})
	}
}

func TestSortFloat64Specials(t *testing.T) {
	for _, target := range availableTargets() {
		t.Run(target.String(), func(t *testing.T) {
			defer forceTarget(t, target)()
			r := rand.New(rand.NewSource(5))
			specials := specialFloat64s()
			for _, n := range sizes(testing.Short()) {
				s := make([]float64, n)
				for i := range s {
					if r.Intn(3) == 0 {
						s[i] = specials[r.Intn(len(specials))]
					} else {
						s[i] = r.NormFloat64() * 1e3
					}
				}
				want := slices.Clone(s)
				slices.Sort(want)
				before := bitCounts64(s)

				Sort(s)
				checkFloatOrder(t, s)
				after := bitCounts64(s)
				for k, v := range before {
					if after[k] != v {
						t.Fatalf("n=%d: bit pattern %#x count changed", n, k)
					}
				}
			}
		})
	}
}

// TestSortAllNaN covers the degenerate input where the prepass consumes the
// whole slice and the kernel is handed nothing.
func TestSortAllNaN(t *testing.T) {
	for _, n := range []int{1, 2, 33, 100, 5000} {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32(math.NaN())
		}
		Sort(s)
		for i, v := range s {
			if v == v {
				t.Fatalf("n=%d: index %d is not NaN", n, i)
			}
		}
	}
}

// TestSignedZeroCount checks that the sort moves signed zeros around rather
// than synthesizing or dropping them. Their relative order is unspecified, so
// only the counts are asserted.
func TestSignedZeroCount(t *testing.T) {
	const n = 5000
	s := make([]float64, n)
	r := rand.New(rand.NewSource(9))
	negs := 0
	for i := range s {
		if r.Intn(2) == 0 {
			s[i] = math.Copysign(0, -1)
			negs++
		} else {
			s[i] = 0
		}
	}
	Sort(s)
	got := 0
	for _, v := range s {
		if math.Signbit(v) {
			got++
		}
	}
	if got != negs {
		t.Errorf("got %d negative zeros, want %d", got, negs)
	}
}

func FuzzSortInt32(f *testing.F) {
	f.Add([]byte{3, 1, 2})
	f.Fuzz(func(t *testing.T, b []byte) {
		s := make([]int32, len(b)/4)
		for i := range s {
			s[i] = int32(b[4*i]) | int32(b[4*i+1])<<8 | int32(b[4*i+2])<<16 | int32(b[4*i+3])<<24
		}
		before := slices.Clone(s)
		Sort(s)
		checkSortedPermutation(t, before, s)
	})
}

func FuzzSortFloat32(f *testing.F) {
	f.Add([]byte{0, 0, 0, 0, 1, 2, 3, 4})
	f.Fuzz(func(t *testing.T, b []byte) {
		s := make([]float32, len(b)/4)
		for i := range s {
			u := uint32(b[4*i]) | uint32(b[4*i+1])<<8 | uint32(b[4*i+2])<<16 | uint32(b[4*i+3])<<24
			s[i] = math.Float32frombits(u)
		}
		before := bitCounts32(s)
		Sort(s)
		checkFloatOrder(t, s)
		after := bitCounts32(s)
		for k, v := range before {
			if after[k] != v {
				t.Fatalf("bit pattern %#x count changed", k)
			}
		}
	})
}
