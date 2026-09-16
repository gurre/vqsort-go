//go:build goexperiment.simd && arm64 && vqsortchecked

package vqsort

import "unsafe"

// at is the checked counterpart of the version in slots_fast.go: it re-slices
// through the bounds check, so any window the partition loop forms outside the
// slice panics rather than corrupting memory. Build with -tags vqsortchecked.
func at[T key](keys []T, i, n int) unsafe.Pointer {
	w := keys[i : i+n : i+n]
	return unsafe.Pointer(unsafe.SliceData(w))
}
