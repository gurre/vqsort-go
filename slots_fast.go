//go:build goexperiment.simd && arm64 && !vqsortchecked

package vqsort

import "unsafe"

// at returns a pointer to keys[i], for a window of n elements that the caller
// has already proved is in range.
//
// The partition loop's capacity invariants (see partitionI32) guarantee every
// window it forms lies inside the slice, so the compiler's bounds check is
// provably redundant - and measurably expensive, since it sits on the innermost
// load and store of the hot loop. Removing it is worth about nine percent.
//
// Building with -tags vqsortchecked swaps in a version that bounds-checks every
// window, turning an invariant violation back into a panic instead of memory
// corruption. The test suite passes under both, and CI runs both.
func at[T key](keys []T, i, n int) unsafe.Pointer {
	var zero T
	return unsafe.Add(unsafe.Pointer(unsafe.SliceData(keys)), uintptr(i)*unsafe.Sizeof(zero))
}
