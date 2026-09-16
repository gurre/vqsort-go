//go:build goexperiment.simd && arm64

package vqsort

import (
	"testing"
	"time"
	"unsafe"

	"simd/archsimd"
)

// BenchmarkPartition measures the partition loop alone, which is where a
// vectorized quicksort spends most of its time: every key passes through it at
// every level of the recursion, while the base case sees each key once and
// pivot selection is amortized to nothing.
//
// The x_scalar metric is the acceptance criterion for the whole approach. With
// 4 lanes per vector the ceiling is 4x; the bar is half of that.
func BenchmarkPartition(b *testing.B) {
	const n = 1 << 24
	src := benchInput[int32](n, 42)
	pivot := int32(0) // a median-ish pivot for uniform keys centred on zero
	work := make([]int32, n)
	var buf [bufLen32]int32

	scalarNs := func() float64 {
		best := 0.0
		for i := 0; i < 3; i++ {
			copy(work, src)
			start := time.Now()
			partitionScalar32(work, pivot)
			ns := float64(time.Since(start).Nanoseconds()) / float64(n)
			if best == 0 || ns < best {
				best = ns
			}
		}
		return best
	}()

	b.SetBytes(int64(n) * int64(unsafe.Sizeof(int32(0))))
	b.ResetTimer()
	for b.Loop() {
		b.StopTimer()
		copy(work, src)
		b.StartTimer()
		partitionI32(work, pivot, archsimd.BroadcastInt32x4(pivot), &buf)
	}
	b.StopTimer()

	perKey := float64(b.Elapsed().Nanoseconds()) / float64(b.N) / float64(n)
	b.ReportMetric(perKey, "ns/key")
	b.ReportMetric(scalarNs/perKey, "x_scalar")
}

// BenchmarkPartitionScalar is the same work through the scalar reference, so
// the two can also be compared with benchstat.
func BenchmarkPartitionScalar(b *testing.B) {
	const n = 1 << 24
	src := benchInput[int32](n, 42)
	work := make([]int32, n)

	b.SetBytes(int64(n) * int64(unsafe.Sizeof(int32(0))))
	b.ResetTimer()
	for b.Loop() {
		b.StopTimer()
		copy(work, src)
		b.StartTimer()
		partitionScalar32(work, 0)
	}
	b.StopTimer()
	b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N)/float64(n), "ns/key")
}

// BenchmarkMemcpy gives the partition numbers a ceiling to be read against:
// partition at this size is bound by memory, not arithmetic.
func BenchmarkMemcpy(b *testing.B) {
	const n = 1 << 24
	src := make([]int32, n)
	dst := make([]int32, n)
	b.SetBytes(int64(n) * int64(unsafe.Sizeof(int32(0))))
	for b.Loop() {
		copy(dst, src)
	}
	b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N)/float64(n), "ns/key")
}

// BenchmarkPartitionVec isolates the per-vector cost - compare, mask reduce,
// table lookup - from the loop's memory traffic.
func BenchmarkPartitionVec(b *testing.B) {
	in := [lanes32]int32{5, -1, 9, -3}
	v := archsimd.LoadInt32x4Array(&in)
	pivot := archsimd.BroadcastInt32x4(0)
	var sink int
	for b.Loop() {
		_, n := partitionVecI32(v, pivot)
		sink += n
	}
	b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N)/lanes32, "ns/key")
	if sink == 0 {
		b.Fatal("optimized away")
	}
}

// BenchmarkNetwork measures the base-case sorting networks on their own.
func BenchmarkNetwork32(b *testing.B) {
	var src [networkKeys32]int32
	for i := range src {
		src[i] = int32(i*2654435761) >> 7
	}
	work := src
	for b.Loop() {
		work = src
		sortNetworkI32(&work)
	}
	b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N)/networkKeys32, "ns/key")
}

func BenchmarkNetwork64(b *testing.B) {
	var src [networkKeys64]int64
	for i := range src {
		src[i] = int64(i*2654435761) >> 7
	}
	work := src
	for b.Loop() {
		work = src
		sortNetworkI64(&work)
	}
	b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N)/networkKeys64, "ns/key")
}
