// No effect if included "normally" (see attr_test.cc).
#ifdef SIMD_TARGET

// Generic implementation, "instantiated" for all supported instruction sets.
template <>
SIMD_ATTR SIMD_NOINLINE void AttrTest::operator()<SIMD_TARGET>() {
  const SIMD_FULL(int32_t) d;
  SIMD_ALIGN int32_t lanes[d.N];
  std::iota(lanes, lanes + d.N, 1);
  auto v = load(d, lanes);
  SIMD_ALIGN int32_t lanes2[d.N];
  store(v, d, lanes2);
  for (size_t i = 0; i < d.N; ++i) {
    if (lanes[i] != lanes2[i]) {
      printf("Mismatch in lane %zu\n", i);
      abort();
    }
  }
  printf("OK: %s\n", vec_name<SIMD_FULL(uint8_t)>());
}

#endif
