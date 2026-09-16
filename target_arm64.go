package vqsort

// NEON is the 128-bit Advanced SIMD kernel. It is available when the binary is
// built with GOEXPERIMENT=simd; every arm64 CPU Go supports implements NEON, so
// there is no runtime feature check.
const NEON = neon
