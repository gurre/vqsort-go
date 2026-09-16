package vqsort

import (
	"fmt"
	"math/rand"
	"slices"
	"testing"
)

// The oracle is slices.Sort on a copy. Both it and vqsort are unstable, and for
// float keys equal values can be distinguishable (-0.0 versus +0.0, NaN
// payloads), so the general assertion is "sorted, and a permutation of the
// input" rather than a positional diff against the oracle. For the integer
// kinds, where equal keys are indistinguishable, those two properties are
// exactly equivalent to matching the oracle element for element, and the
// stronger check is asserted as well.

// values produces one input family. Values stay inside the int32 range so the
// same generator feeds every key type.
type family struct {
	name string
	fill func(s []int64, r *rand.Rand)
}

var families = []family{
	{"uniform", func(s []int64, r *rand.Rand) {
		for i := range s {
			s[i] = int64(r.Int31()) - 1<<30
		}
	}},
	{"lowentropy", func(s []int64, r *rand.Rand) {
		// 16-bit values in wider keys: the paper's trigger for degenerate
		// partitions, where many keys tie with the pivot.
		for i := range s {
			s[i] = int64(r.Intn(1 << 16))
		}
	}},
	{"allequal", func(s []int64, r *rand.Rand) {
		v := int64(r.Int31())
		for i := range s {
			s[i] = v
		}
	}},
	{"sorted", func(s []int64, r *rand.Rand) {
		for i := range s {
			s[i] = int64(i)
		}
	}},
	{"reverse", func(s []int64, r *rand.Rand) {
		for i := range s {
			s[i] = int64(len(s) - i)
		}
	}},
	{"sawtooth2", sawtooth(2)},
	{"sawtooth3", sawtooth(3)},
	{"sawtooth17", sawtooth(17)},
	{"sawtooth256", sawtooth(256)},
	{"organpipe", func(s []int64, r *rand.Rand) {
		for i := range s {
			if i < len(s)/2 {
				s[i] = int64(i)
			} else {
				s[i] = int64(len(s) - i)
			}
		}
	}},
	{"distinct2", distinct(2)},
	{"distinct3", distinct(3)},
	{"distinct16", distinct(16)},
	{"distinct256", distinct(256)},
	{"minmax", func(s []int64, r *rand.Rand) {
		for i := range s {
			if i%2 == 0 {
				s[i] = -1 << 31
			} else {
				s[i] = 1<<31 - 1
			}
		}
	}},
	{"pivotbait", func(s []int64, r *rand.Rand) {
		// Almost all keys equal, with a few outliers: partitions split off
		// one key at a time unless degenerate handling works.
		for i := range s {
			s[i] = 1 << 20
		}
		for i := 0; i < len(s)/64+1 && i < len(s); i++ {
			s[r.Intn(len(s))] = int64(r.Int31())
		}
	}},
}

func sawtooth(period int) func([]int64, *rand.Rand) {
	return func(s []int64, r *rand.Rand) {
		for i := range s {
			s[i] = int64(i % period)
		}
	}
}

func distinct(k int) func([]int64, *rand.Rand) {
	return func(s []int64, r *rand.Rand) {
		for i := range s {
			s[i] = int64(r.Intn(k))
		}
	}
}

func makeInput[T key](f family, n int, seed int64) []T {
	raw := make([]int64, n)
	f.fill(raw, rand.New(rand.NewSource(seed)))
	s := make([]T, n)
	for i, v := range raw {
		s[i] = T(v)
	}
	return s
}

// checkSortedPermutation asserts the two properties that define a correct sort.
// The permutation check is an exact multiset comparison: a checksum such as
// sum-and-xor cancels precisely on the low-entropy families above, where a
// dropped key and a duplicated key are often the same value.
func checkSortedPermutation[T key](t *testing.T, before, after []T) {
	t.Helper()
	if len(before) != len(after) {
		t.Fatalf("length changed: %d -> %d", len(before), len(after))
	}
	for i := 1; i < len(after); i++ {
		if after[i] < after[i-1] {
			t.Fatalf("not sorted at %d: %v > %v", i, after[i-1], after[i])
		}
	}
	counts := make(map[T]int, len(before))
	for _, v := range before {
		counts[v]++
	}
	for _, v := range after {
		counts[v]--
		if counts[v] < 0 {
			t.Fatalf("key %v appears more often in the output", v)
		}
	}
	for v, n := range counts {
		if n != 0 {
			t.Fatalf("key %v lost %d times", v, n)
		}
	}
}

// sizes returns the lengths to test. The exhaustive sweep covers every base
// case width and every remainder residue of the partition loop's unroll
// factor; the short sweep keeps the boundaries that actually break things.
func sizes(short bool) []int {
	if !short {
		s := make([]int, 0, 2049)
		for n := 0; n <= 2048; n++ {
			s = append(s, n)
		}
		return append(s, 4093, 65536, 1<<20)
	}
	return []int{
		0, 1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 33, 47, 48, 49,
		63, 64, 65, 95, 96, 127, 128, 129, 255, 257, 1000, 4093, 65536,
	}
}

func seeds(short bool) []int64 {
	if short {
		return []int64{1}
	}
	return []int64{1, 2, 3}
}

func TestSortInt32(t *testing.T)   { testSortKind[int32](t) }
func TestSortInt64(t *testing.T)   { testSortKind[int64](t) }
func TestSortFloat32(t *testing.T) { testSortKind[float32](t) }
func TestSortFloat64(t *testing.T) { testSortKind[float64](t) }

func testSortKind[T key](t *testing.T) {
	for _, target := range availableTargets() {
		t.Run(target.String(), func(t *testing.T) {
			restore := forceTarget(t, target)
			defer restore()
			for _, f := range families {
				for _, n := range sizes(testing.Short()) {
					for _, seed := range seeds(testing.Short()) {
						in := makeInput[T](f, n, seed)
						before := slices.Clone(in)
						want := slices.Clone(in)
						slices.Sort(want)

						Sort(in)
						checkSortedPermutation(t, before, in)
						if !slices.Equal(in, want) {
							t.Fatalf("%s n=%d seed=%d: output differs from slices.Sort", f.name, n, seed)
						}
					}
				}
			}
		})
	}
}

// availableTargets returns every kernel this binary and CPU can run, so one
// test binary exercises all of them.
func availableTargets() []Target {
	var ts []Target
	for _, t := range []Target{Scalar, neon, avx2, avx512} {
		if targetAvailable(t) {
			ts = append(ts, t)
		}
	}
	return ts
}

// forceTarget pins the kernel for the duration of a test. Forcing anything
// other than Scalar also drops the short-input floor, so that the kernels are
// tested at every length rather than only above the length where they are
// faster than slices.Sort.
func forceTarget(t *testing.T, target Target) func() {
	t.Helper()
	prev, err := SetTarget(target)
	if err != nil {
		t.Skipf("target %s unavailable: %v", target, err)
	}
	prev32, prev64 := vectorFloor32, vectorFloor64
	if target != Scalar {
		vectorFloor32, vectorFloor64 = 0, 0
	}
	return func() {
		vectorFloor32, vectorFloor64 = prev32, prev64
		SetTarget(prev)
	}
}

// TestNamedTypes covers the case a type switch on any(x) silently drops: a
// defined type whose underlying type has a kernel.
func TestNamedTypes(t *testing.T) {
	type UserID int64
	type Score float32
	type IDs []UserID

	ids := IDs{9, 3, 7, 1, 3}
	Sort(ids)
	if !slices.IsSorted(ids) {
		t.Errorf("named slice of named element type not sorted: %v", ids)
	}
	if kindOf[UserID]() != kindInt64 {
		t.Errorf("UserID classified as %v, want kindInt64", kindOf[UserID]())
	}
	if kindOf[Score]() != kindFloat32 {
		t.Errorf("Score classified as %v, want kindFloat32", kindOf[Score]())
	}

	// A larger case, to make sure the reinterpreted slice really reaches the
	// kernel rather than only the base case.
	big := make([]UserID, 5000)
	for i := range big {
		big[i] = UserID(rand.Int63())
	}
	before := slices.Clone(big)
	Sort(big)
	checkSortedPermutation(t, before, big)
}

// TestStringNeverReinterpreted guards a memory-safety boundary: string is part
// of cmp.Ordered and is pointer-backed, so it must classify as kindOther and
// reach slices.Sort without ever passing through the unsafe reinterpretation
// used for the numeric kinds.
func TestStringNeverReinterpreted(t *testing.T) {
	if got := kindOf[string](); got != kindOther {
		t.Fatalf("string classified as %v, want kindOther", got)
	}
	type Name string
	if got := kindOf[Name](); got != kindOther {
		t.Fatalf("named string classified as %v, want kindOther", got)
	}
	s := []string{"pear", "apple", "fig", "date"}
	Sort(s)
	if !slices.IsSorted(s) {
		t.Errorf("strings not sorted: %v", s)
	}
}

// TestUnkerneledKindsStillSort covers the element types that are inside
// cmp.Ordered but deliberately have no kernel in v1.
func TestUnkerneledKindsStillSort(t *testing.T) {
	i8 := []int8{5, -3, 0, 127, -128}
	Sort(i8)
	if !slices.IsSorted(i8) {
		t.Errorf("int8 not sorted: %v", i8)
	}
	u64 := []uint64{9, 1, 1 << 63, 0}
	Sort(u64)
	if !slices.IsSorted(u64) {
		t.Errorf("uint64 not sorted: %v", u64)
	}
	if Accelerated[uint64]() {
		t.Error("Accelerated[uint64] reports true, but uint64 has no kernel in v1")
	}
	if Accelerated[string]() {
		t.Error("Accelerated[string] reports true")
	}
}

func TestSortIsIdempotentOnSortedInput(t *testing.T) {
	for _, n := range []int{33, 64, 65, 1000, 20000} {
		s := make([]int32, n)
		for i := range s {
			s[i] = int32(i)
		}
		Sort(s)
		for i := range s {
			if s[i] != int32(i) {
				t.Fatalf("n=%d: sorted input perturbed at %d: %d", n, i, s[i])
			}
		}
	}
}

func ExampleSort() {
	keys := []int64{5, 2, 9, 2, -7}
	Sort(keys)
	fmt.Println(keys)
	// Output: [-7 2 2 5 9]
}
