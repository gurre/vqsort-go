package vqsort

import (
	"math/rand"
	"slices"
	"testing"
)

// TestDepthGuardStaysQuiet is the assertion no output check can make.
//
// Heapsort is correct, so if the depth guard tripped on every range the sort
// would still return sorted output and every other test would still pass, while
// the whole point of the kernel was lost. The only way to see that is to watch
// the counter.
func TestDepthGuardStaysQuiet(t *testing.T) {
	for _, target := range availableTargets() {
		if target == Scalar {
			continue // slices.Sort has its own guard and does not report here
		}
		t.Run(target.String(), func(t *testing.T) {
			defer forceTarget(t, target)()
			for _, f := range families {
				for _, n := range []int{100, 1000, 20000} {
					before := depthGuardTrips.Load()
					in := makeInput[int32](f, n, 1)
					Sort(in)
					if got := depthGuardTrips.Load() - before; got != 0 {
						t.Errorf("%s n=%d: depth guard fired %d times on ordinary input", f.name, n, got)
					}
				}
			}
		})
	}
}

func TestHeapSort(t *testing.T) {
	r := rand.New(rand.NewSource(4))
	for n := 0; n < 200; n++ {
		s := make([]int32, n)
		for i := range s {
			s[i] = int32(r.Intn(100))
		}
		before := slices.Clone(s)
		heapSort(s)
		checkSortedPermutation(t, before, s)
	}
}

func TestInsertionSort(t *testing.T) {
	r := rand.New(rand.NewSource(6))
	for n := 0; n < 200; n++ {
		s := make([]float64, n)
		for i := range s {
			s[i] = float64(r.Intn(100))
		}
		before := slices.Clone(s)
		insertionSort(s)
		checkSortedPermutation(t, before, s)
	}
}

func TestMedianOf3(t *testing.T) {
	for a := 0; a < 4; a++ {
		for b := 0; b < 4; b++ {
			for c := 0; c < 4; c++ {
				got := medianOf3(int32(a), int32(b), int32(c))
				want := []int32{int32(a), int32(b), int32(c)}
				slices.Sort(want)
				if got != want[1] {
					t.Fatalf("medianOf3(%d,%d,%d) = %d, want %d", a, b, c, got, want[1])
				}
			}
		}
	}
}

// TestSortDoesNotAllocate holds the package doc to its word. Nothing in the type
// system enforces it, so a future refactor that moves the scratch buffer or
// swaps the RNG for one that escapes would otherwise go unnoticed.
func TestSortDoesNotAllocate(t *testing.T) {
	for _, target := range availableTargets() {
		t.Run(target.String(), func(t *testing.T) {
			defer forceTarget(t, target)()

			i32 := makeInput[int32](families[0], 20000, 1)
			i64 := makeInput[int64](families[0], 20000, 1)
			f32 := makeInput[float32](families[0], 20000, 1)
			f64 := makeInput[float64](families[0], 20000, 1)

			for _, c := range []struct {
				name string
				run  func()
			}{
				{"int32", func() { Sort(i32) }},
				{"int64", func() { Sort(i64) }},
				{"float32", func() { Sort(f32) }},
				{"float64", func() { Sort(f64) }},
			} {
				if got := testing.AllocsPerRun(5, c.run); got != 0 {
					t.Errorf("Sort[%s] allocated %v times per call, want 0", c.name, got)
				}
			}
		})
	}
}

func TestSFC64IsDeterministic(t *testing.T) {
	a, b := newSFC64(42), newSFC64(42)
	for i := 0; i < 100; i++ {
		if x, y := a.next(), b.next(); x != y {
			t.Fatalf("draw %d: %d != %d for the same seed", i, x, y)
		}
	}
	c := newSFC64(43)
	same := 0
	d := newSFC64(42)
	for i := 0; i < 100; i++ {
		if c.next() == d.next() {
			same++
		}
	}
	if same > 2 {
		t.Errorf("different seeds produced %d identical draws in 100", same)
	}
}

func TestBoundedStaysInRange(t *testing.T) {
	rng := newSFC64(77)
	for _, n := range []uint32{1, 2, 3, 17, 1000, 1 << 20} {
		for i := 0; i < 5000; i++ {
			if v := rng.bounded(n); v >= n {
				t.Fatalf("bounded(%d) returned %d", n, v)
			}
		}
	}
}

// TestMaxDepthBound pins the depth guard's budget to exact values.
//
// Nothing about a sort's output reveals this number, so a guard set far too
// high - or removed - leaves every other test passing while the worst case
// quietly becomes quadratic. The budget is 2*ceil(log2(n)) + 4, the bound that
// makes quicksort's worst case O(n log n).
func TestMaxDepthBound(t *testing.T) {
	for _, c := range []struct{ n, want int }{
		{1, 6}, {2, 8}, {3, 8}, {4, 10}, {16, 14}, {1024, 26}, {1 << 20, 46}, {1 << 30, 66},
	} {
		if got := maxDepth(c.n); got != c.want {
			t.Errorf("maxDepth(%d) = %d, want %d", c.n, got, c.want)
		}
	}
	// A budget that grows faster than logarithmically is not a guard at all.
	if maxDepth(1<<30)-maxDepth(1<<20) != 20 {
		t.Errorf("budget does not grow by 2 per doubling: %d vs %d", maxDepth(1<<30), maxDepth(1<<20))
	}
}
