package vqsort

import (
	"cmp"
	"reflect"
	"slices"
	"unsafe"
)

// Sort sorts x in ascending order.
//
// Sort produces the same result as [slices.Sort] for every input, including
// floating-point NaNs, which are ordered before all other values. The sort is
// not stable.
//
// Element types with a vector kernel are sorted by it; all others forward to
// [slices.Sort]. See the package documentation for the table, and [Accelerated]
// to test a particular element type at run time.
func Sort[S ~[]E, E cmp.Ordered](x S) {
	s := []E(x)
	if len(s) < 2 {
		return
	}
	switch kindOf[E]() {
	case kindInt32:
		sortInt32(reinterpret[int32](s))
	case kindInt64:
		sortInt64(reinterpret[int64](s))
	case kindFloat32:
		sortFloat32(reinterpret[float32](s))
	case kindFloat64:
		sortFloat64(reinterpret[float64](s))
	default:
		slices.Sort(s)
	}
}

// Accelerated reports whether Sort can use a vector kernel for element type E in
// this binary, on this CPU, under the current [Target].
//
// It answers the two questions the design creates: whether the binary was built
// with GOEXPERIMENT=simd, and whether E is one of the element kinds a kernel
// covers. It says nothing about a particular call: short slices go to
// [slices.Sort] whatever Accelerated reports, because below a few thousand keys
// that is the faster path.
func Accelerated[E cmp.Ordered]() bool {
	return kernelReady(kindOf[E](), CurrentTarget())
}

// An elemKind is the family of kernel that sorts a given element type. Types
// that differ only in name share a kind, so "type UserID int64" sorts through
// the same kernel as int64.
type elemKind uint8

const (
	kindOther elemKind = iota
	kindInt32
	kindInt64
	kindFloat32
	kindFloat64
)

// kindOf classifies E. A type switch on any(x) cannot be used here: for a
// defined type such as "type UserID int64" the dynamic type of any([]UserID{})
// is []UserID, which matches no case for []int64, so every named type would
// silently lose its kernel.
func kindOf[E cmp.Ordered]() elemKind {
	switch reflect.TypeFor[E]().Kind() {
	case reflect.Int32:
		return kindInt32
	case reflect.Int64:
		return kindInt64
	case reflect.Float32:
		return kindFloat32
	case reflect.Float64:
		return kindFloat64
	case reflect.Int:
		if unsafe.Sizeof(int(0)) == 8 {
			return kindInt64
		}
		return kindInt32
	}
	return kindOther
}

// reinterpret views s as a slice of T.
//
// This is sound only for the element kinds kindOf recognizes: each is a
// pointer-free numeric type with the same width and alignment as its T, so the
// two slices have identical memory layout and the garbage collector sees no
// pointers through either. String, the one pointer-bearing kind in cmp.Ordered,
// classifies as kindOther and never reaches here.
func reinterpret[T any, E cmp.Ordered](s []E) []T {
	return unsafe.Slice((*T)(unsafe.Pointer(unsafe.SliceData(s))), len(s))
}
