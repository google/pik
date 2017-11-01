// Includes a given .inc file for every enabled SIMD_TARGET. This is used in
// attr mode to generate template instantiations to be called by dispatch::Run.

#ifndef SIMD_ATTR_IMPL
#error "Must set SIMD_ATTR_IMPL to name of include file"
#endif

#if SIMD_ENABLE_AVX2
#undef SIMD_TARGET
#define SIMD_TARGET AVX2
#include SIMD_ATTR_IMPL
#endif

#if SIMD_ENABLE_SSE4
#undef SIMD_TARGET
#define SIMD_TARGET SSE4
#include SIMD_ATTR_IMPL
#endif

#if SIMD_ENABLE_ARM8
#undef SIMD_TARGET
#define SIMD_TARGET ARM8
#include SIMD_ATTR_IMPL
#endif

#undef SIMD_TARGET
#define SIMD_TARGET NONE
#include SIMD_ATTR_IMPL

#undef SIMD_ATTR_IMPL
