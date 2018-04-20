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

#ifndef DATA_PARALLEL_H_
#define DATA_PARALLEL_H_

// Portable, low-overhead C++11 ThreadPool alternative to OpenMP for
// data-parallel computations.

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <algorithm>  // find_if  TODO: remove after PerThread
#include <atomic>
#include <condition_variable>  //NOLINT
#include <cstdlib>
#include <memory>
#include <mutex>   //NOLINT
#include <thread>  //NOLINT
#include <vector>

#define DATA_PARALLEL_CHECK(condition)                           \
  while (!(condition)) {                                         \
    printf("data_parallel check failed at line %d\n", __LINE__); \
    abort();                                                     \
  }

namespace pik {

// Highly scalable thread pool, especially suitable for data-parallel
// computations in the fork-join model, where clients need to know when all
// tasks have completed.
//
// Thread pools usually store small numbers of heterogeneous tasks in a queue.
// When tasks are identical or differ only by an integer input parameter, it is
// much faster to store just one function of an integer parameter and call it
// for each value. Conventional vector-of-tasks can be run in parallel using a
// lambda function adapter that simply calls task_funcs[task].
//
// This thread pool can efficiently load-balance millions of tasks using an
// atomic counter, thus avoiding per-task syscalls. With 48 hyperthreads and
// 1M tasks that add to an atomic counter, overall runtime is 10-20x higher
// when using std::async, and up to 200x for a queue-based ThreadPool.
//
// Usage:
// ThreadPool pool;
// pool.Run(0, 1000000, [](const int task, const int thread) { Func1(task); });
// // When Run returns, all of its tasks have finished.
// // The destructor waits until all worker threads have exited cleanly.
class ThreadPool {
  // Calls f(task, thread). Used for type erasure of Func arguments. The
  // signature must match TypeErasedFunc, hence a const void* argument.
  template <class Closure>
  static void CallClosure(const void* f, const int task, const int thread) {
    (*reinterpret_cast<const Closure*>(f))(task, thread);
  }

 public:
  // For per-thread arrays. Can increase if needed.
  static constexpr int kMaxThreads = 256;

  // Starts the given number of worker threads and blocks until they are ready.
  // "num_threads" defaults to one per hyperthread. If zero, all tasks run on
  // the main thread.
  explicit ThreadPool(
      const int num_threads = std::thread::hardware_concurrency())
      : num_threads_(num_threads) {
    DATA_PARALLEL_CHECK(num_threads >= 0);
    DATA_PARALLEL_CHECK(num_threads <= kMaxThreads);
    threads_.reserve(num_threads);
    for (int i = 0; i < num_threads; ++i) {
      threads_.emplace_back(ThreadFunc, this, i);
    }

    if (num_threads_ != 0) {
      WorkersReadyBarrier();
    }
  }

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator&(const ThreadPool&) = delete;

  // Waits for all threads to exit.
  ~ThreadPool() {
    if (num_threads_ != 0) {
      StartWorkers(kWorkerExit);
    }

    for (std::thread& thread : threads_) {
      thread.join();
    }
  }

  // Returns number of worker threads created (some may be sleeping and never
  // wake up in time to participate in Run).
  size_t NumThreads() const { return num_threads_; }

  // Runs func(task, thread) on worker thread(s) for every task in [begin, end).
  // "thread" is 0 if NumThreads() == 0, otherwise [0, NumThreads()).
  // Not thread-safe - no two calls to Run may overlap.
  // Subsequent calls will reuse the same threads.
  //
  // Precondition: 0 <= begin <= end.
  template <class Func>
  void Run(const int begin, const int end, const Func& func) {
    DATA_PARALLEL_CHECK(0 <= begin && begin <= end);
    if (begin == end) {
      return;
    }
    if (num_threads_ == 0) {
      const int thread = 0;
      for (int task = begin; task < end; ++task) {
        func(task, thread);
      }
      return;
    }

    const WorkerCommand worker_command = (WorkerCommand(end) << 32) + begin;
    // Ensure the inputs do not result in a reserved command.
    DATA_PARALLEL_CHECK(worker_command != kWorkerWait);
    DATA_PARALLEL_CHECK(worker_command != kWorkerOnce);
    DATA_PARALLEL_CHECK(worker_command != kWorkerExit);

    func_ = &CallClosure<Func>;
    arg_ = &func;
    num_reserved_.store(0, std::memory_order_relaxed);

    StartWorkers(worker_command);
    WorkersReadyBarrier();
  }

  // Runs func(thread, thread) on all thread(s) that may participate in Run.
  // If NumThreads() == 0, runs on the main thread with thread == 0, otherwise
  // concurrently called by each worker thread in [0, NumThreads()).
  template <class Func>
  void RunOnEachThread(const Func& func) {
    if (num_threads_ == 0) {
      const int thread = 0;
      func(thread, thread);
      return;
    }

    func_ = reinterpret_cast<TypeErasedFunc>(&CallClosure<Func>);
    arg_ = &func;
    StartWorkers(kWorkerOnce);
    WorkersReadyBarrier();
  }

 private:
  // After construction and between calls to Run, workers are "ready", i.e.
  // waiting on worker_start_cv_. They are "started" by sending a "command"
  // and notifying all worker_start_cv_ waiters. (That is why all workers
  // must be ready/waiting - otherwise, the notification will not reach all of
  // them and the main thread waits in vain for them to report readiness.)
  using WorkerCommand = uint64_t;

  // Special values; all others encode the begin/end parameters.
  static constexpr WorkerCommand kWorkerWait = ~0ULL;
  static constexpr WorkerCommand kWorkerOnce = ~1ULL;
  static constexpr WorkerCommand kWorkerExit = ~2ULL;

  void WorkersReadyBarrier() {
    std::unique_lock<std::mutex> lock(mutex_);
    workers_ready_cv_.wait(
        lock, [this]() { return workers_ready_ == threads_.size(); });
    workers_ready_ = 0;
  }

  // Precondition: all workers are ready.
  void StartWorkers(const WorkerCommand worker_command) {
    mutex_.lock();
    worker_start_command_ = worker_command;
    // Workers will need this lock, so release it before they wake up.
    mutex_.unlock();
    worker_start_cv_.notify_all();
  }

  // Attempts to reserve and perform some work from the global range of tasks,
  // which is encoded within "command". Returns after all tasks are reserved.
  static void RunRange(ThreadPool* self, const WorkerCommand command,
                       const int thread) {
    const int begin = command & 0xFFFFFFFF;
    const int end = command >> 32;
    const int num_tasks = end - begin;
    const int num_threads = static_cast<int>(self->num_threads_);

    // OpenMP introduced several "schedule" strategies:
    // "single" (static assignment of exactly one chunk per thread): slower.
    // "dynamic" (allocates k tasks at a time): competitive for well-chosen k.
    // "guided" (allocates k tasks, decreases k): computing k = remaining/n
    //   is faster than halving k each iteration. We prefer this strategy
    //   because it avoids user-specified parameters.

    for (;;) {
#if 0
      // dynamic
      const int my_size =
          std::max(num_tasks / (num_threads * 4), 1);
#else
      // guided
      const int num_reserved = self->num_reserved_.load();
      const int num_remaining = num_tasks - num_reserved;
      const int my_size = std::max(num_remaining / (num_threads * 4), 1);
#endif
      const int my_begin = begin + self->num_reserved_.fetch_add(my_size);
      const int my_end = std::min(my_begin + my_size, begin + num_tasks);
      // Another thread already reserved the last task.
      if (my_begin >= my_end) {
        break;
      }
      for (int task = my_begin; task < my_end; ++task) {
        self->func_(self->arg_, task, thread);
      }
    }
  }

  // What task to run on a worker thread. Points to code generated via
  // CallClosure. Arguments are arg_ (points to the lambda), task, thread.
  using TypeErasedFunc = void (*)(const void*, int, int);

  static void ThreadFunc(ThreadPool* self, const int thread) {
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

  // Unmodified after ctor, but cannot be const because we call thread::join().
  std::vector<std::thread> threads_;

  const size_t num_threads_;

  std::mutex mutex_;  // guards both cv and their variables.
  std::condition_variable workers_ready_cv_;
  size_t workers_ready_ = 0;
  std::condition_variable worker_start_cv_;
  WorkerCommand worker_start_command_;

  // Written by main thread, read by workers (after mutex lock/unlock).
  TypeErasedFunc func_;
  const void* arg_;

  // Updated by workers; alignment/padding avoids false sharing.
  alignas(64) std::atomic<int> num_reserved_{0};
  int padding[15];
};

// Accelerates multiple unsigned 32-bit divisions with the same divisor by
// precomputing a multiplier. This is useful for splitting a contiguous range of
// indices (the task index) into 2D indices. Exhaustively tested on dividends
// up to 4M with non-power of two divisors up to 2K.
class Divider {
 public:
  // "d" is the divisor (what to divide by).
  Divider(const uint32_t d) : shift_(31 - __builtin_clz(d)) {
    // Power of two divisors (including 1) are not supported because it is more
    // efficient to special-case them at a higher level.
    DATA_PARALLEL_CHECK((d & (d - 1)) != 0);

    // ceil_log2 = floor_log2 + 1 because we ruled out powers of two above.
    const uint64_t next_pow2 = 1ULL << (shift_ + 1);

    mul_ = ((next_pow2 - d) << 32) / d + 1;
  }

  // "n" is the numerator (what is being divided).
  inline uint32_t operator()(const uint32_t n) const {
    // Algorithm from "Division by Invariant Integers using Multiplication".
    // Its "sh1" is hardcoded to 1 because we don't need to handle d=1.
    const uint32_t hi = (uint64_t(mul_) * n) >> 32;
    return (hi + ((n - hi) >> 1)) >> shift_;
  }

 private:
  uint32_t mul_;
  const int shift_;
};

// DEPRECATED in favor of PerThread2.
//
// Thread-local storage with support for reduction (combining into one result).
// The "T" type must be unique to the call site because the list of threads'
// copies is a static member. (With knowledge of the underlying threads, we
// could eliminate this list and T allocations, but that is difficult to
// arrange and we prefer this to be usable independently of ThreadPool.)
//
// Usage:
// for (int i = 0; i < N; ++i) {
//   // in each thread:
//   T& my_copy = PerThread<T>::Get();
//   my_copy.Modify();
//
//   // single-threaded:
//   T& combined = PerThread<T>::Reduce();
//   Use(combined);
//   PerThread<T>::Destroy();
// }
//
// T is duck-typed and implements the following interface:
//
// // Returns true if T is default-initialized or Destroy was called without
// // any subsequent re-initialization.
// bool IsNull() const;
//
// // Releases any resources. Postcondition: IsNull() == true.
// void Destroy();
//
// // Merges in data from "victim". Precondition: !IsNull() && !victim.IsNull().
// void Assimilate(const T& victim);
template <class T>
class PerThread {
 public:
  // Returns reference to this thread's T instance (dynamically allocated,
  // so its address is unique). Callers are responsible for any initialization
  // beyond the default ctor.
  static T& Get() {
    static thread_local T* t;
    if (t == nullptr) {
      t = new T;
      static std::mutex mutex;
      std::lock_guard<std::mutex> lock(mutex);
      Threads().push_back(t);
    }
    return *t;
  }

  // Returns vector of all per-thread T. Used inside Reduce() or by clients
  // that require direct access to T instead of Assimilating them.
  // Function wrapper avoids separate static member variable definition.
  static std::vector<T*>& Threads() {
    static std::vector<T*> threads;
    return threads;
  }

  // Returns the first non-null T after assimilating all other threads' T
  // into it. Precondition: at least one non-null T exists (caller must have
  // called Get() and initialized the result).
  static T& Reduce() {
    std::vector<T*>& threads = Threads();

    // Find first non-null T
    const auto it = std::find_if(threads.begin(), threads.end(),
                                 [](const T* t) { return !t->IsNull(); });
    if (it == threads.end()) {
      abort();
    }
    T* const first = *it;

    for (const T* t : threads) {
      if (t != first && !t->IsNull()) {
        first->Assimilate(*t);
      }
    }
    return *first;
  }

  // Calls each thread's T::Reset to release resources and/or prepare for
  // reuse by the same threads/ThreadPool. Note that all T remain allocated
  // until DeleteAll because threads retain local pointers to them, which must
  // remain valid as long as threads can potentially call Get.
  static void Reset() {
    for (T* t : Threads()) {
      t->Reset();
    }
  }

  // Deallocates all threads' allocated T to avoid memory leak warnings.
  // No other member functions may be called after this is called!
  static void DeleteAll() {
    for (T* t : Threads()) {
      delete t;
    }
  }
};

// Enables efficient concurrent updates of T (e.g. counters/statistics) followed
// by reduction (combining all counters into one final value).
//
// Avoids user-allocated arrays of T by dynamically allocating them per thread,
// referenced via thread_local pointer. Avoids is_initialized checks in each
// task by initializing all threads' instances up-front. This requires support
// from ThreadPool; a sequence of Initialize/Reduce/Reset/Delete must be called
// with the same ThreadPool* argument.
//
// T is duck-typed and implements the following interface:
//   T(), operator=(const T&)
//   void Assimilate(const T& victim) : merges victim into *this.
//
// Usage:
//   ThreadPool pool(kNumThreads);
//   struct T {
//     void Assimilate(const T& victim) { counter += victim.counter; }
//     int counter = 0;
//   };
//   PerThread2<T>::Allocate(&pool);
//   START:
//   pool.Run(0, kNumTasks, []() {
//     T& my_copy = PerThread2<T>::Get();  // cheap: POD TLS access
//     my_copy.Modify();  // no synchronization needed
//   });
//
//   T& combined = PerThread2<T>::Reduce(&pool);  // internally synchronized
//   Use(combined);
//
//   // optional: PerThread2<T>::Reset(&pool); goto START;
//   PerThread2<T>::Delete(&pool);  // frees all memory
//
template <class T>
class PerThread2 {
 public:
  // Allocates each thread's instance of T and calls its default ctor.
  static void Allocate(ThreadPool* pool) {
    pool->RunOnEachThread([](int task, int thread) { GetPtrRef() = new T; });
  }

  // Returns this thread's dynamically allocated instance of T.
  // Must not be called before Allocate!
  static T& Get() { return *GetPtrRef(); }

  // Returns a pointer to T that has all other threads' T assimilated into it.
  static T& Reduce(ThreadPool* pool) {
    T* first = nullptr;
    std::mutex mutex;
    pool->RunOnEachThread([&first, &mutex](int task, int thread) {
      mutex.lock();
      if (first == nullptr) {
        first = &Get();
      } else {
        first->Assimilate(Get());
      }
      mutex.unlock();
    });
    return *first;
  }

  // Prepares for reuse by assigning a default-constructed T. Note that all T
  // remain allocated until Delete; each thread retains a local pointer, which
  // must remain valid as long as threads can potentially call Get.
  static void Reset(ThreadPool* pool) {
    pool->RunOnEachThread([](int task, int thread) { Get() = T(); });
  }

  // Deallocates all threads' allocated T to avoid memory leak warnings.
  // Do not call Get/Reduce/Reset again without calling Allocate first.
  static void Delete(ThreadPool* pool) {
    pool->RunOnEachThread([](int task, int thread) { delete GetPtrRef(); });
  }

 private:
  // Returns a reference to this thread's private pointer to T, which is
  // initially nullptr until Allocate is called.
  static T*& GetPtrRef() {
    static thread_local T* t;
    return t;
  }
};

}  // namespace pik

#endif  // DATA_PARALLEL_H_
