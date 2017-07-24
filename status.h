// Copyright 2017 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef STATUS_H_
#define STATUS_H_

#include <cstdio>
#include <cstdlib>

namespace pik {

// Should be broad categories - we rely on the ability to crash/dump stack
// at the moment of failure via PIK_FAIL .
enum class Status {
  OK = 0,
  // TODO(user) Add some hierarchy to these error codes.
  INVALID_JPEG_INPUT = 1,
  NOT_IMPLEMENTED = 2,
  EMPTY_IMAGE = 3,
  INVALID_PIK_INPUT = 4,
  EMPTY_INPUT = 5,
  JPEG_DECODING_ERROR = 6,
  INVALID_FORMAT_CODE = 7,
  GUETZLI_FAILED = 8,
  WRONG_MAGIC = 9,
  RANGE_EXCEEEDED = 10,
};

#ifdef PIK_ENABLE_ASSERT
#define PIK_ASSERT(condition)                                   \
  while (!(condition)) {                                        \
    printf("Pik assert failed at %s:%d\n", __FILE__, __LINE__); \
    abort();                                                    \
  }
#else
#define PIK_ASSERT(condition)
#endif

#define PIK_CHECK(condition)                                   \
  while (!(condition)) {                                       \
    printf("Pik check failed at %s:%d\n", __FILE__, __LINE__); \
    abort();                                                   \
  }

// Evaluates an expression (typically function call) returning a Status;
// if the value is not OK, returns that value from the current function.
#define PIK_RETURN_IF_ERROR(expression) \
  for (;;) {                            \
    const Status status = (expression); \
    if (status != Status::OK) {         \
      return status;                    \
    }                                   \
    break;                              \
  }

// Annotation for the location where an error condition is first noticed.
// Error codes are too unspecific to pinpoint the exact location, so we
// add a build flag crashes and dumps stack at the actual error source.
#ifdef PIK_CRASH_ON_ERROR
#define PIK_NOTIFY_ERROR(message_string)                                    \
  for (;;) {                                                                \
    printf("Pik failure at %s:%d: %s", __FILE__, __LINE__, message_string); \
    abort();                                                                \
  }
#else
#define PIK_NOTIFY_ERROR(message_string)
#endif

}  // namespace pik

#endif  // STATUS_H_
