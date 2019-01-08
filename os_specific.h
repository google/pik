// Copyright 2016 Google Inc. All Rights Reserved.
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

#ifndef OS_SPECIFIC_H_
#define OS_SPECIFIC_H_

#include <string>
#include <vector>

#include "status.h"

namespace pik {

// Returns current time [seconds] from a monotonic clock with unspecified
// starting point - only suitable for computing elapsed time.
double Now();

// Returns CPU numbers in [0, N), where N is the number of bits in the
// thread's initial affinity (unaffected by any SetThreadAffinity).
std::vector<int> AvailableCPUs();

// Opaque.
struct ThreadAffinity;

// Caller must free() the return value.
ThreadAffinity* GetThreadAffinity();

// Restores a previous affinity returned by GetThreadAffinity.
Status SetThreadAffinity(ThreadAffinity* affinity);

// Ensures the thread is running on the specified cpu, and no others.
// Useful for reducing nanobenchmark variability (fewer context switches).
// Uses SetThreadAffinity.
Status PinThreadToCPU(const int cpu);

// Random choice of CPU avoids overloading any one core.
// Uses SetThreadAffinity.
Status PinThreadToRandomCPU();

// Executes a command in a subprocess.
Status RunCommand(const std::vector<std::string>& args);

}  // namespace pik

#endif  // OS_SPECIFIC_H_
