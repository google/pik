#ifndef PREVENT_ELISION_H_
#define PREVENT_ELISION_H_

namespace pik {

// Prevents the compiler from eliding the computations that led to "output".
template <class T>
inline void PreventElision(T&& output) {
#ifndef _MSC_VER
  // Works by indicating to the compiler that "output" is being read and
  // modified. The +r constraint avoids unnecessary writes to memory, but only
  // works for built-in types (typically FuncOutput).
  asm volatile("" : "+r"(output) : : "memory");
#else
  // MSVC does not support inline assembly anymore (and never supported GCC's
  // RTL constraints). Self-assignment with #pragma optimize("off") might be
  // expected to prevent elision, but it does not with MSVC 2015. Type-punning
  // with volatile pointers generates inefficient code on MSVC 2017.
  static std::atomic<T> dummy(T{});
  dummy.store(output, std::memory_order_relaxed);
#endif
}

}  // namespace pik

#endif  // PREVENT_ELISION_H_
