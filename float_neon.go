//go:build goexperiment.simd && arm64

package vqsort

// The float entry points are hand-written rather than derived from the integer
// kernels, because they have to deal with NaN before anything else runs.

// sortFloat32NEON sorts s, ordering NaNs before every other value to match
// cmp.Less and therefore slices.Sort.
//
// The NaNs are moved to the front first and excluded from the sort. That is not
// only about output order: with NaNs present, Less is not a total order and
// Min and Max do not agree with it, so the partition and the pivot reduction
// would both be operating on a comparison that is not transitive.
func sortFloat32NEON(s []float32) {
	nan := 0
	for i, v := range s {
		if v != v {
			s[i], s[nan] = s[nan], v
			nan++
		}
	}
	s = s[nan:]

	if len(s) <= baseCase32 {
		sortBaseF32(s)
		return
	}
	var buf [bufLen32]float32
	rng := newSFC64(seedCounter.Add(1))
	recurseF32(s, choosePivotF32(s, &rng), &buf, &rng, maxDepth(len(s)))
}

// sortFloat64NEON sorts s, ordering NaNs before every other value to match
// cmp.Less and therefore slices.Sort. See sortFloat32NEON for why the prepass
// has to run before anything compares keys.
func sortFloat64NEON(s []float64) {
	nan := 0
	for i, v := range s {
		if v != v {
			s[i], s[nan] = s[nan], v
			nan++
		}
	}
	s = s[nan:]

	if len(s) <= baseCase64 {
		sortBaseF64(s)
		return
	}
	var buf [bufLen64]float64
	rng := newSFC64(seedCounter.Add(1))
	recurseF64(s, choosePivotF64(s, &rng), &buf, &rng, maxDepth(len(s)))
}
