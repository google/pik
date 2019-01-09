// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef STATUS_H_
#define STATUS_H_

#include <cstdio>
#include <cstdlib>

#include "compiler_specific.h"

namespace pik {

#ifndef PIK_ENABLE_ASSERT
#define PIK_ENABLE_ASSERT 1
#endif

#if PIK_ENABLE_ASSERT || defined(FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION)
#define PIK_ASSERT(condition)                                            \
  while (!(condition)) {                                                 \
    fprintf(stderr, "Pik assert failed at %s:%d\n", __FILE__, __LINE__); \
    abort();                                                             \
  }
#else
#define PIK_ASSERT(condition)
#endif

#define PIK_CHECK(condition)                                            \
  while (!(condition)) {                                                \
    fprintf(stderr, "Pik check failed at %s:%d\n", __FILE__, __LINE__); \
    abort();                                                            \
  }

#define PIK_RETURN_IF_ERROR(condition) \
  while (!(condition)) return false

// Annotation for the location where an error condition is first noticed.
// Error codes are too unspecific to pinpoint the exact location, so we
// add a build flag that crashes and dumps stack at the actual error source.
#ifdef PIK_CRASH_ON_ERROR
inline bool PikFailure(const char* f, int l, const char* msg) {
  for (;;) {
    fprintf(stderr, "Pik failure at %s:%d: %s\n", f, l, msg);
    abort();
  }
  return false;
}
#define PIK_NOTIFY_ERROR(message_string) \
  (void)PikFailure(__FILE__, __LINE__, message_string)
#define PIK_FAILURE(message_string) \
  PikFailure(__FILE__, __LINE__, message_string)
#else
#define PIK_NOTIFY_ERROR(message_string)
#define PIK_FAILURE(message_string) false
#endif

// Drop-in replacement for bool that raises compiler warnings if not used
// after being returned from a function. Example:
// Status LoadFile(...) { return true; } is more compact than
// bool PIK_MUST_USE_RESULT LoadFile(...) { return true; }
class PIK_MUST_USE_RESULT Status {
 public:
  Status(bool ok) : ok_(ok) {}

  operator bool() const { return ok_; }

 private:
  bool ok_;
};

}  // namespace pik

#endif  // STATUS_H_
