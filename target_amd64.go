package vqsort

const (
	// AVX2 is the 256-bit kernel. It is the floor for vectorized sorting on
	// amd64: 128-bit kernels move too few keys per instruction to beat
	// slices.Sort.
	AVX2 = avx2

	// AVX512 is the 512-bit kernel, requiring AVX512F+CD+BW+DQ+VL.
	AVX512 = avx512
)
