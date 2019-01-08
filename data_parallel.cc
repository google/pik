// Copyright 2018 Google Inc. All Rights Reserved.
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

#include "data_parallel.h"
#include "profiler.h"

namespace pik {

void ThreadPool::ThreadFunc(ThreadPool* self, const int thread) {
  // Until kWorkerExit command received:
  for (;;) {
    std::unique_lock<std::mutex> lock(self->mutex_);
    // Notify main thread that this thread is ready.
    if (++self->workers_ready_ == self->NumThreads()) {
      self->workers_ready_cv_.notify_one();
    }
  RESUME_WAIT:
    // Wait for a command.
    self->worker_start_cv_.wait(lock);
    const WorkerCommand command = self->worker_start_command_;
    switch (command) {
      case kWorkerWait:    // spurious wakeup:
        goto RESUME_WAIT;  // lock still held, avoid incrementing ready.
      case kWorkerOnce:
        lock.unlock();
        self->func_(self->arg_, thread, thread);
        break;
      case kWorkerExit:
        return;  // exits thread
      default:
        lock.unlock();
        RunRange(self, command, thread);
        break;
    }
  }
}

ThreadPool::ThreadPool(const int num_worker_threads)
    : num_worker_threads_(num_worker_threads),
      num_threads_(std::max(num_worker_threads, 1)) {
  PROFILER_ZONE("ThreadPool ctor");

  PIK_CHECK(num_worker_threads >= 0);
  PIK_CHECK(num_worker_threads <= kMaxThreads);
  threads_.reserve(num_worker_threads);

  // Suppress "unused-private-field" warning.
  (void)padding;

  // Safely handle spurious worker wakeups.
  worker_start_command_ = kWorkerWait;

  for (int i = 0; i < num_worker_threads; ++i) {
    threads_.emplace_back(ThreadFunc, this, i);
  }

  if (num_worker_threads_ != 0) {
    WorkersReadyBarrier();
  }

  // Warm up profiler on worker threads so its expensive initialization
  // doesn't count towards other timer measurements.
  RunOnEachThread(
      [](const int task, const int thread) { PROFILER_ZONE("@InitWorkers"); });
}

ThreadPool::~ThreadPool() {
  if (num_worker_threads_ != 0) {
    StartWorkers(kWorkerExit);
  }

  for (std::thread& thread : threads_) {
    PIK_ASSERT(thread.joinable());
    thread.join();
  }
}

}  // namespace pik
