## Efficient and portable SIMD wrapper

This library provides type-safe and source-code portable wrappers over existing
platform-specific intrinsics. Its design aims for simplicity, efficiency and
immediate usability with current compilers.

## Current status

Implemented for SSE4, AVX2 and scalar (portable) targets, each with unit tests.

`blaze build -c opt simd:all &&
blaze-bin/simd/simd_test`

## Design philosophy

*   Performance is important but not the sole consideration. Anyone who goes to
    the trouble of using SIMD clearly cares about speed. However, portability,
    maintainability and readability also matter, otherwise we would write in
    assembly. We aim for performance within 10-20% of a hand-written assembly
    implementation on the development platform.

*   The guiding principles of C++ are "pay only for what you use" and "leave no
    room for a lower-level language below C++". We apply these by defining a
    SIMD API that ensures operation costs are visible, predictable and minimal.

*   Performance portability is important, i.e. the API should be efficient on
    all target platforms. Unfortunately, common idioms for one platform can be
    inefficient on others. For example: summing lanes horizontally versus
    shuffling. Documenting which operations are expensive does not prevent their
    use, as evidenced by widespread use of `HADDPS`. Performance acceptance
    tests may detect large regressions, but do not help choose the approach
    during initial development. Analysis tools can warn about some potential
    inefficiencies, but likely not all. We instead provide [a carefully chosen
    set of vector types and operations that are efficient on all target
    platforms][instmtx] (PPC8, SSE4/AVX2+, ARMv8), plus some useful but less
    performance-portable operations in an `ext` namespace to make their cost
    visible.

*   Future SIMD hardware features are difficult to predict. For example, AVX2
    came with surprising semantics (almost no interaction between 128-bit
    halves) and AVX-512 added two kinds of predicates (writemask and zeromask).
    To ensure the API reflects hardware realities, we suggest a flexible
    approach that adds new operations as they become commonly available.

*   Masking is not yet widely supported on current CPUs. It is difficult to
    define an interface that provides access to all platform features while
    retaining performance portability. The P0214R5 proposal lacks support for
    AVX-512/ARM SVE zeromasks. We suggest standardizing masking only after the
    community has gained more experience with it.

*   "Width-agnostic" SIMD is more future-proof than user-specified fixed sizes.
    For example, valarray-like code can iterate over a 1D array with a
    library-specified vector width. This will result in better code when vector
    sizes increase, and matches the direction taken by ARM SVE and RiscV
    hardware. However, some applications may require fixed sizes, so we also
    support vectors of 128-bit size on all platforms, and 256-bit vectors in
    AVX2-specific applications.

*   The API and its implementation should be usable and efficient with commonly
    used compilers. Some of our open-source users cannot upgrade, so we need to
    support 4-6 year old compilers (e.g. GCC 4.8). However, we take advantage of
    newer features such as function-specific target attributes when available.

*   Efficient and safe runtime dispatch is important. Modules such as image or
    video codecs are typically embedded into larger applications such as
    browsers, so they cannot require separate binaries for each CPU. Using only
    the lowest-common denominator instructions sacrifices too much performance.
    Therefore, we need to provide code paths for multiple instruction sets and
    choose the best one at runtime. To reduce overhead, dispatch should be
    hoisted to higher layers instead of checking inside every low-level
    function. Generating each code path from the same source reduces
    implementation and debugging cost.

*   Not every CPU need be supported. For example, pre-SSE4 CPUs are increasingly
    rare and the AVX instruction set is limited to floating-point operations.
    To reduce code size and compile time, we provide specializations for SSE4,
    AVX2 and AVX-512 instruction sets on x86.

*   Access to platform-specific intrinsics is necessary for acceptance in
    performance-critical projects. We provide conversions to and from intrinsics
    to allow utilizing specialized platform-specific functionality such as
    `MPSADBW`, and simplify incremental porting of existing code.

*   The core API should be compact and easy to learn. We provide only the few
    dozen operations which are necessary and sufficient for most of the 150+
    SIMD applications we examined. As a result, the quick_reference card in
    `g3doc/` is only 6 pages long.

## Differences versus [P0214R5 proposal](https://goo.gl/zKW4SA)

1.  Adding widely used and portable operations such as `average`, `mul_even`,
    and `shuffle`.

1.  Adding the concept of half-vectors, which are often used in existing ARM
    and x86 code.

1.  Avoiding the need for non-native vectors. By contrast, P0214R5's `simd_cast`
    returns `fixed_size<>` vectors which are more expensive to access because
    they reside on the stack. We can avoid this plus additional overhead on
    ARM/AVX2 by defining width-expanding operations as functions of a vector
    subset, e.g. promoting half a vector of `uint8_t` lanes to one full vector
    of `uint16_t`, or demoting a full vector to a half vector with half-width
    lanes.

1.  Guaranteeing access to the underlying intrinsic vector type. P0214R5 only
    'encourages' implementations to allow a static_cast. We provide conversions
    to ensure all platform-specific features can be used.

1.  Enabling safe runtime dispatch and inlining in the same binary. P0214R5 is
    based on the Vc library, which does not provide assistance for linking
    multiple instruction sets into the same binary. The Vc documentation
    suggests compiling separate executables for each instruction set or using
    GCC's ifunc (indirect functions). The latter is compiler-specific and risks
    crashes due to ODR violations when compiling the same function with
    different compiler flags. We provide two solutions: avoiding the need for
    multiple flags on recent compilers, and defending against ODR violations on
    older compilers (see HOWTO section below).

1.  Using built-in PPC vector types without any wrapper. This leads to much
    better code generation with GCC 4.8: https://godbolt.org/g/KYp7ew.
    By contrast, P0214R5 requires a wrapper class. We avoid this by using only
    the member operators provided by the PPC vectors; all other functions and
    typedefs are non-members and the user-visible `vec<T, Target>` interface is
    an alias template.

*   Omitting inefficient or non-performance-portable operations such as `hmax`,
    `operator[]`, and unsupported integer comparisons. Applications can often
    replace these operations at lower cost than emulating them.

*   Omitting `long double` types: these are not commonly available in hardware.

*   Simple implementation, less than a tenth of the size of the Vc library
    from which P0214 was derived (98,000 lines in https://github.com/VcDevel/Vc
    according to the gloc Chrome extension).

*   Avoiding hidden performance costs. P0214R5 allows implicit conversions from
    integer to float, which costs 3-4 cycles on x86. We make these conversions
    explicit to ensure their cost is visible.

## Prior API designs

The author has been writing SIMD code since 2002: first via assembly language,
then intrinsics, later Intel's `F32vec4` wrapper, followed by three generations
of custom vector classes. The first used macros to generate the classes, which
reduces duplication but also readability. The second used templates instead.
The third (used in highwayhash and PIK) added support for AVX2 and runtime
dispatch. The current design enables code generation for multiple platforms
and/or instruction sets from the same source, and improves runtime dispatch.

## Other related work

*   [Neat SIMD](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7568423)
    adopts a similar approach with interchangeable vector/scalar types and
    a compact interface. It allows access to the underlying intrinsics, but
    does not appear to be designed for other platforms than x86.

*   UME::SIMD ([code](https://goo.gl/yPeVZx), [paper](https://goo.gl/2xpZrk))
    also adopts an explicit vectorization model with vector classes.
    However, it exposes the union of all platform capabilities, which makes the
    API harder to learn (209-page spec) and reduces performance portability
    because it allows applications to use operations that are inefficient on
    other platforms.

*   Inastemp ([code](https://goo.gl/hg3USM), [paper](https://goo.gl/YcTU7S))
    is a vector library for scientific computing with some innovative features:
    automatic FLOPS counting, and "if/else branches" using lambda functions.
    It supports IBM Power8, but only provides float and double types.

We are unaware of any existing vector libraries with support for safely bundling
code targeting multiple instruction sets into the same binary (see HOWTO below).

## Use cases and HOWTO

*   Older compilers, single instruction set per platform: compile with a
    `COPTS_REQUIRE_?` plus `COPTS_ENABLE_?`; use fixed-size 128-bit vectors
    (`vec128` or `f32x4`) or width-agnostic `vec<float>`.

*   Older compilers, runtime dispatch: one library per instruction set, each
    compiled with a `COPTS_ENABLE_?` and `COPTS_REQUIRE_?`. The libraries
    can be built from the same source by instantiating a template that uses
    width-agnostic SIMD (`vec<uint8_t, SIMD_TARGET>`).
    Use `dispatch::Run` to choose the best available implementation at runtime.
    Prevent ODR violations by ensuring functions are inlined or defined within
    `SIMD_NAMESPACE`. This can be verified using binutils.

*   Newer compilers (GCC 4.9+, Clang 3.9+), single instruction set per platform:
    compile with `COPTS_ENABLE_?`, add `SIMD_ATTR` to any functions using SIMD,
    use `vec*<T>` (e.g. `vec256`) or its aliases (`f32x8`).

*   Newer compilers (GCC 4.9+, Clang 3.9+), runtime dispatch: compile with
    `COPTS_ENABLE_*`, add `SIMD_ATTR_?` to functions using that instruction set,
    select an implementation by checking CPU capabilities (e.g. with
    `dispatch::SupportedTargets`). SSE4 and AVX2 code can coexist in the same
    library/binary, but generating them from a single source requires macros or
    `#include` because each function needs a distinct `SIMD_ATTR_*` attribute.

## Demos

To compile on Unix systems: `make -j6`. We tested with Clang 3.4 and GCC 4.8.4.

`bin/simd_test` prints a bitfield of instruction sets that were
tested, e.g. `6` for SSE4=`4` and AVX2=`2`. The demo compiles the same source
file once per enabled instruction set. This approach has relatively modest
compiler requirements.

`bin/custom_dispatcher_test` also prints messages for every instruction set.
It contains few tests; the main purpose is to demonstrate compilation without
`-mavx2` flags. This approach requires Clang 3.9+ or GCC 4.9+, or MSVC 2015+.

## Example source code

```c++
void Copy(const uint8_t* SIMD_RESTRICT from, const size_t size,
          uint8_t* SIMD_RESTRICT to) {
  // Width-agnostic (library-specified NumLanes)
  using V = vec<uint8_t>;
  size_t i = 0;
  for (; i + NumLanes<V>() <= size; i += NumLanes<V>()) {
    const auto bytes = load(V(), from + i);
    store(bytes, to + i);
  }

  for (; i < size; i += NumLanes<vec1<uint8_t>>()) {
    // (Same loop body as above, could factor into a shared template)
    const auto bytes = load(vec1<uint8_t>(), from + i);
    store(bytes, to + i);
  }
}
```

```c++
void MulAdd(const T* SIMD_RESTRICT mul_array, const T* SIMD_RESTRICT add_array,
            const size_t size, T* SIMD_RESTRICT x_array) {
  // Type-agnostic (caller-specified lane type) and width-agnostic (uses
  // best available instruction set).
  using V = vec<T>;
  for (size_t i = 0; i < size; i += NumLanes<V>()) {
    const auto mul = load(V(), mul_array + i);
    const auto add = load(V(), add_array + i);
    auto x = load(V(), x_array + i);
    x = mul_add(mul, x, add);
    store(x, x_array + i);
  }
}
```

```c++
int GetMostSignificantBits(const uint8_t* SIMD_RESTRICT from) {
  // Fixed-size, can use template or type alias.
  static_assert(sizeof(vec128<uint8_t>) == sizeof(u8x16), "Size mismatch");
  const auto bytes = load(u8x16(), from);
  return ext::movemask(bytes);  // 16 bits, one from each byte
}
```

## Additional resources

*   [Overview of instructions per operation on different architectures][instmtx]

[instmtx]: instruction_matrix.pdf
