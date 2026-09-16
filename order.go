package vqsort

import "sync/atomic"

// key is the set of element types the vector kernels sort. Ordering here uses
// the < operator directly, which is a total order for the integer types and,
// once NaNs have been moved aside by the float prepass, for the float types
// too.
type key interface {
	~int32 | ~int64 | ~float32 | ~float64
}

// insertionSort sorts s. It is the base case for small ranges, where the
// branch-predictable scalar loop beats setting up vectors.
//
// The cursor is written as j in [0, i] rather than the more usual j in [-1,
// i-1], because that is the form the compiler's bounds-check prover can follow:
// j only ever decreases from i, and i is already known to be inside s, so both
// indices in the shifting loop are provably in range and neither carries a
// check. The same loop written with j+1 keeps a bounds check on the hottest
// instruction in the base case.
func insertionSort[T key](s []T) {
	for i := 1; i < len(s); i++ {
		k := s[i]
		j := i
		for j > 0 {
			prev := s[j-1]
			if !(k < prev) {
				break
			}
			s[j] = prev
			j--
		}
		s[j] = k
	}
}

// heapSort is the depth-guard fallback. Reaching it means pivot selection has
// been unlucky (or adversarial) often enough that quicksort's expected depth
// has been exceeded; heapsort bounds the remainder at O(n log n).
func heapSort[T key](s []T) {
	for i := len(s)/2 - 1; i >= 0; i-- {
		siftDown(s, i, len(s))
	}
	for i := len(s) - 1; i > 0; i-- {
		s[0], s[i] = s[i], s[0]
		siftDown(s, 0, i)
	}
}

func siftDown[T key](s []T, root, n int) {
	for {
		child := 2*root + 1
		if child >= n {
			return
		}
		if child+1 < n && s[child] < s[child+1] {
			child++
		}
		if !(s[root] < s[child]) {
			return
		}
		s[root], s[child] = s[child], s[root]
		root = child
	}
}

// medianOf3 returns the middle of three values.
func medianOf3[T key](a, b, c T) T {
	if b < a {
		a, b = b, a
	}
	if c < b {
		b = c
		if b < a {
			b = a
		}
	}
	return b
}

// sfc64 is Chris Doty-Humphrey's small fast counting generator. Pivot sampling
// needs an RNG that is cheap enough to call on every partition and independent
// of math/rand's global state, so that a sort is reproducible under test.
type sfc64 struct {
	a, b, c, counter uint64
}

// seedCounter makes each top-level sort draw a distinct but reproducible
// stream: runs are deterministic for a fixed sequence of calls, while an
// adversary cannot predict pivots for a single input from a previous run.
var seedCounter atomic.Uint64

func newSFC64(seed uint64) sfc64 {
	r := sfc64{a: seed, b: seed, c: seed, counter: 1}
	for i := 0; i < 12; i++ {
		r.next()
	}
	return r
}

func (r *sfc64) next() uint64 {
	tmp := r.a + r.b + r.counter
	r.counter++
	r.a = r.b ^ (r.b >> 11)
	r.b = r.c + (r.c << 3)
	r.c = (r.c<<24 | r.c>>40) + tmp
	return tmp
}

// bounded returns a value in [0, n) by multiply-shift. The slight bias is
// irrelevant for choosing which keys to sample as pivot candidates.
func (r *sfc64) bounded(n uint32) uint32 {
	return uint32((uint64(uint32(r.next()>>32)) * uint64(n)) >> 32)
}

// depthGuardTrips counts the ranges that exhausted the depth guard and fell
// back to heapsort.
//
// No assertion on a sort's output can detect this: heapsort is correct, so a
// threshold set too low would silently turn every sort into a heapsort and the
// only symptom would be an unexplained benchmark. Tests therefore assert on
// this counter directly, in both directions - that it trips on adversarial
// input, and that it does not trip on ordinary input.
var depthGuardTrips atomic.Uint64

func guardFired() { depthGuardTrips.Add(1) }

// maxDepth bounds how many partitions a range may take before the sort falls
// back to heapsort.
func maxDepth(n int) int {
	d := 0
	for i := n; i > 0; i >>= 1 {
		d++
	}
	return 2*d + 4
}

// nintherFloor is the shortest range that samples nine keys for its pivot
// rather than three.
//
// Each sample is a random gather, and short ranges run many partitions per key,
// so on small inputs the nine-sample median costs more than the better pivot
// saves. Above this length the balance reverses: a bad split there costs a full
// extra pass over a long range.
const nintherFloor = 128
