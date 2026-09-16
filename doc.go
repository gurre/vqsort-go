//go:generate go run ./gen
/*
Package vqsort sorts slices of ordered values using vectorized quicksort.

Sort is a drop-in replacement for [slices.Sort]. It has the same signature and
produces the same result for every input, so migrating a call site is a rename:

	slices.Sort(keys)  ->  vqsort.Sort(keys)

The algorithm is the vectorized quicksort of Blacher, Giesen, Wassenberg and
Sanders (arXiv:2205.05982): a bidirectional in-place partition driven by SIMD
compare-and-compress, with a sorting-network base case and a heapsort fallback
that bounds the worst case at O(n log n).

# You must opt in at build time

The vector kernels are compiled only when the toolchain is built with the simd
experiment enabled:

	GOEXPERIMENT=simd go build ./...

That setting is process-wide: it affects the whole binary, not just this
package. Without it every call is correct and simply forwards to [slices.Sort],
so a default build loses the speedup rather than failing. Use [Accelerated] to
find out which one you got.

# Which element types are accelerated

A kernel exists for the element kinds below. Every other element type permitted
by [cmp.Ordered] - string, the 8- and 16-bit integers, and the unsigned types -
forwards to [slices.Sort]. Named types are accelerated too: an element type
whose underlying type is one of these, such as "type UserID int64", takes the
same kernel as its underlying type.

	int32, float32           4 lanes on arm64 NEON, 8 on AVX2, 16 on AVX-512
	int64, float64, int      2 lanes on arm64 NEON, 4 on AVX2, 8 on AVX-512

Short slices go to [slices.Sort] for every element type, because a short sort
spends most of its time on per-partition overhead rather than on moving keys,
and the scalar pdqsort wins there. The threshold is 64 keys for 32-bit elements
and 8192 for 64-bit ones.

# Ordering

Sort matches [slices.Sort] exactly, which means it matches [cmp.Less]:

  - Floating-point NaNs are ordered before all other values.
  - Negative and positive zero compare equal, and their relative order in the
    output is unspecified.
  - The sort is not stable. Use [slices.SortStableFunc] if you need stability.

# Allocation

The current implementation performs no heap allocation, for any element type or
length. This is covered by test rather than promised by the API.

# Choosing a kernel

[SetTarget] forces a particular kernel and is intended for tests, benchmarks and
diagnostics. It changes a process-global setting, so call it from main or from a
test, never from a reusable library: overriding it silently changes the
performance - never the results - of every other caller in the process.

The Go runtime's own switch also works and needs no code change, because kernel
selection reads the standard CPU feature detection:

	GODEBUG=cpu.avx512=off  # force the AVX2 kernel
	GODEBUG=cpu.all=off     # force the scalar path
*/
package vqsort
