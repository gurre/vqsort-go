package vqsort

import (
	"math"
	"math/rand"
	"slices"
	"strconv"
	"testing"
	"time"
	"unsafe"
)

// Benchmarks report throughput as MB/s of key bytes sorted, which is the unit
// the vqsort paper uses and the only one comparable across key widths. Two
// custom metrics are added:
//
//	ns/key    time per key, independent of key width
//	x_stdlib  throughput as a multiple of slices.Sort on the same input
//
// x_stdlib is measured inside the benchmark rather than left to benchstat, so a
// single run answers the question the project exists to answer.

var benchSizes = []struct {
	name string
	n    int
}{
	{"1K", 1 << 10},
	{"64K", 1 << 16},
	{"1M", 1 << 20},
	{"16M", 1 << 24},
}

func benchInput[T key](n int, seed int64) []T {
	r := rand.New(rand.NewSource(seed))
	s := make([]T, n)
	for i := range s {
		s[i] = T(r.Int31() - 1<<30)
	}
	return s
}

// nsPerKey returns the best of several batches, which is less noisy than the
// mean for a baseline that only needs to be accurate to a few percent.
//
// Each batch is sized so it runs long enough to measure: sorting a few hundred
// keys takes less time than the clock's own resolution, and a baseline taken
// from a single such sort is noise, which then shows up as a meaningless
// x_stdlib on every small benchmark.
func nsPerKey[T key](src, work []T, sort func([]T)) float64 {
	const batchTarget = 200 * time.Microsecond
	reps := 1
	for {
		start := time.Now()
		for i := 0; i < reps; i++ {
			copy(work, src)
			sort(work)
		}
		if elapsed := time.Since(start); elapsed >= batchTarget || reps >= 1<<20 {
			break
		}
		reps *= 4
	}

	best := math.MaxFloat64
	for round := 0; round < 5; round++ {
		start := time.Now()
		for i := 0; i < reps; i++ {
			copy(work, src)
			sort(work)
		}
		// The copy is inside the timed region for both the baseline and the
		// benchmark's own pool refill, so it does not bias the ratio.
		ns := float64(time.Since(start).Nanoseconds()) / float64(reps) / float64(len(src))
		if ns < best {
			best = ns
		}
	}
	return best
}

// benchmarkSort times sort over independent copies of src.
//
// The obvious harness - copy the input back between iterations with the timer
// stopped - is not usable for small inputs: testing.B.StopTimer reads memory
// statistics, which stops the world and evicts the caches, so every iteration
// would sort from cold and the overhead would swamp a sort that takes only
// microseconds. Instead a pool of ready copies is prepared up front and the
// timer is only stopped when the pool runs out, which for small inputs is
// almost never and for large ones is amortized over a sort that dwarfs it.
func benchmarkSort[T key](b *testing.B, src []T, sort func([]T)) {
	n := len(src)
	var zero T
	width := int(unsafe.Sizeof(zero))

	const poolBudget = 64 << 20
	copies := poolBudget / (n * width)
	copies = min(max(copies, 1), 1024)

	pool := make([]T, copies*n)
	refill := func() {
		for i := 0; i < copies; i++ {
			copy(pool[i*n:(i+1)*n], src)
		}
	}
	refill()

	baseline := nsPerKey(src, make([]T, n), slices.Sort[[]T, T])

	b.SetBytes(int64(n) * int64(width))
	i := 0
	b.ResetTimer()
	for b.Loop() {
		if i == copies {
			b.StopTimer()
			refill()
			b.StartTimer()
			i = 0
		}
		sort(pool[i*n : (i+1)*n])
		i++
	}
	b.StopTimer()

	if !slices.IsSorted(pool[0:n]) {
		b.Fatal("benchmark produced unsorted output")
	}
	perKey := float64(b.Elapsed().Nanoseconds()) / float64(b.N) / float64(n)
	b.ReportMetric(perKey, "ns/key")
	b.ReportMetric(baseline/perKey, "x_stdlib")
}

func BenchmarkSortInt32(b *testing.B)   { benchSortKind[int32](b) }
func BenchmarkSortInt64(b *testing.B)   { benchSortKind[int64](b) }
func BenchmarkSortFloat32(b *testing.B) { benchSortKind[float32](b) }
func BenchmarkSortFloat64(b *testing.B) { benchSortKind[float64](b) }

func benchSortKind[T key](b *testing.B) {
	for _, size := range benchSizes {
		src := benchInput[T](size.n, 42)
		b.Run(size.name+"/vqsort", func(b *testing.B) {
			benchmarkSort(b, src, Sort[[]T, T])
		})
		b.Run(size.name+"/stdlib", func(b *testing.B) {
			benchmarkSort(b, src, slices.Sort[[]T, T])
		})
	}
}

// BenchmarkSortPattern covers the input shapes that decide whether pivot
// selection and the degenerate-partition path hold up, not just the uniform
// random case that flatters every quicksort.
func BenchmarkSortPattern(b *testing.B) {
	const n = 1 << 20
	for _, f := range []string{"lowentropy", "allequal", "sorted", "reverse", "distinct16", "organpipe", "pivotbait"} {
		i := slices.IndexFunc(families, func(x family) bool { return x.name == f })
		src := makeInput[int32](families[i], n, 1)
		b.Run(f, func(b *testing.B) {
			benchmarkSort(b, src, Sort[[]int32, int32])
		})
	}
}

// partitionScalar32 is the reference the vector partition has to beat: Hoare's
// in-place partition, the same loop a scalar quicksort would run. Comparing
// against it isolates the question of whether the compiler's SIMD codegen is
// worth the complexity, with no dependence on an external figure.
func partitionScalar32(keys []int32, pivot int32) int {
	i, j := 0, len(keys)-1
	for {
		for i <= j && keys[i] <= pivot {
			i++
		}
		for i <= j && keys[j] > pivot {
			j--
		}
		if i > j {
			break
		}
		keys[i], keys[j] = keys[j], keys[i]
		i++
		j--
	}
	return i
}

// BenchmarkCrossover locates the length at which each vector kernel overtakes
// slices.Sort, which is what the small-input floor in the dispatch is set from.
//
// Both implementations run through the same harness so the comparison is
// apples-to-apples: read the vqsort and stdlib ns/key for a size and divide.
// The x_stdlib metric is not reliable here, because its baseline is timed
// differently from the benchmark body and the gap matters at these sizes.
func BenchmarkCrossover(b *testing.B) {
	// Bypass the floor this benchmark exists to calibrate.
	prev32, prev64 := vectorFloor32, vectorFloor64
	vectorFloor32, vectorFloor64 = 0, 0
	defer func() { vectorFloor32, vectorFloor64 = prev32, prev64 }()

	for _, n := range []int{64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384} {
		crossoverPair[int32](b, "int32", n)
		crossoverPair[float32](b, "float32", n)
		crossoverPair[int64](b, "int64", n)
		crossoverPair[float64](b, "float64", n)
	}
}

func crossoverPair[T key](b *testing.B, name string, n int) {
	src := benchInput[T](n, 3)
	b.Run(name+"/"+itoa(n)+"/vqsort", func(b *testing.B) {
		benchmarkSort(b, src, Sort[[]T, T])
	})
	b.Run(name+"/"+itoa(n)+"/stdlib", func(b *testing.B) {
		benchmarkSort(b, src, slices.Sort[[]T, T])
	})
}

func itoa(n int) string { return strconv.Itoa(n) }
