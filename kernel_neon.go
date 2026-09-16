//go:build goexperiment.simd && arm64

package vqsort

import "slices"

// Kernel selection for arm64. Every arm64 CPU Go supports implements NEON, so
// availability is decided at build time by the simd experiment alone and there
// is no runtime feature check.

// vectorFloor32 and vectorFloor64 are the shortest inputs the vector kernels
// are used for; below them slices.Sort is faster and Sort defers to it.
//
// The two differ by an order of magnitude because the kernels do. At four lanes
// the sorting network alone beats pdqsort on a single base case, so the floor
// is the network's width. At two lanes each partition call moves half as many
// keys for the same fixed cost - pivot sampling, preload, drain - and the 64-bit
// compare-exchange costs three instructions where the 32-bit one costs one, so
// the kernel does not pull ahead until the ranges are long enough to amortize
// all of that.
//
// Both are measured, not derived; see BenchmarkCrossover, which runs vqsort and
// slices.Sort through one harness so the two are directly comparable.
//
// They are variables rather than constants only so tests can drop them to zero
// and exercise the kernels at every length, which is where the partition loop's
// remainder and drain cases live.
var (
	vectorFloor32 = 64
	vectorFloor64 = 8192
)

func bestTarget() Target { return neon }

func targetAvailable(t Target) bool { return t == Scalar || t == neon }

// kernelReady reports whether element kind k has a vector kernel under target
// t. Kinds without one still sort correctly; they forward to slices.Sort.
func kernelReady(k elemKind, t Target) bool {
	return t == neon && k != kindOther
}

func sortInt32(s []int32) {
	if len(s) >= vectorFloor32 && Target(current.Load()) == neon {
		sortInt32NEON(s)
		return
	}
	slices.Sort(s)
}

func sortInt64(s []int64) {
	if len(s) >= vectorFloor64 && Target(current.Load()) == neon {
		sortInt64NEON(s)
		return
	}
	slices.Sort(s)
}
func sortFloat32(s []float32) {
	if len(s) >= vectorFloor32 && Target(current.Load()) == neon {
		sortFloat32NEON(s)
		return
	}
	slices.Sort(s)
}
func sortFloat64(s []float64) {
	if len(s) >= vectorFloor64 && Target(current.Load()) == neon {
		sortFloat64NEON(s)
		return
	}
	slices.Sort(s)
}
