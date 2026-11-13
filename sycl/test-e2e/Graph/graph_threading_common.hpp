#pragma once

#include "graph_common.hpp"

#include <algorithm>
#include <exception>
#include <string>
#include <thread>
#include <vector>

/// Times a test repeats its record, execute and verify cycle to widen the
/// window for races.
constexpr unsigned DefaultRaceAttempts = 10;

/// Thread count for graph threading tests. Clamped so small machines still
/// exercise concurrency and large machines don't oversubscribe the device.
inline unsigned numTestThreads(unsigned MinThreads = 2,
                               unsigned MaxThreads = 4) {
  return std::clamp(std::thread::hardware_concurrency(), MinThreads,
                    MaxThreads);
}

/// Settings for a graph threading test.
struct ThreadTestConfig {
  property_list QueueProps;
  unsigned NumThreads = numTestThreads();
  unsigned RaceAttempts = DefaultRaceAttempts;
};

/// The current attempt of a threading test, used to tag failure messages with
/// the settings needed to reproduce the failure.
struct RaceAttempt {
  const ThreadTestConfig &Config;
  unsigned Index;

  /// Name followed by the attempt and settings, e.g.
  /// "Sum [attempt 3/10, 4 threads, in_order]".
  std::string label(const std::string &Name) const {
    const bool InOrder =
        Config.QueueProps.has_property<property::queue::in_order>();
    return Name + " [attempt " + std::to_string(Index) + "/" +
           std::to_string(Config.RaceAttempts) + ", " +
           std::to_string(Config.NumThreads) + " threads, " +
           (InOrder ? "in_order" : "out_of_order") + "]";
  }
};

/// Runs Attempt(RaceAttempt) Config.RaceAttempts times to widen the window for
/// races. Attempts check their results with assert, labelling messages with
/// RaceAttempt::label(). Exceptions are reported with the attempt and rethrown.
template <typename AttemptT>
void repeatRaceAttempts(const ThreadTestConfig &Config, AttemptT &&Attempt) {
  for (unsigned Index = 0; Index < Config.RaceAttempts; ++Index) {
    const RaceAttempt Current{Config, Index};
    try {
      Attempt(Current);
    } catch (...) {
      std::cerr << Current.label("Exception") << std::endl;
      throw;
    }
  }
}

/// Runs Func(ThreadIndex) on NumThreads threads which are released together by
/// a barrier to maximize contention. Blocks until all threads have joined, then
/// rethrows the first exception thrown by any thread.
template <typename FuncT>
void runConcurrently(unsigned NumThreads, FuncT &&Func) {
  Barrier SyncPoint{NumThreads};
  std::vector<std::exception_ptr> Errors(NumThreads);
  std::vector<std::thread> Threads;
  Threads.reserve(NumThreads);

  for (unsigned ThreadIndex = 0; ThreadIndex < NumThreads; ++ThreadIndex) {
    Threads.emplace_back([&, ThreadIndex] {
      SyncPoint.wait();
      try {
        Func(ThreadIndex);
      } catch (...) {
        Errors[ThreadIndex] = std::current_exception();
      }
    });
  }

  for (std::thread &Thread : Threads)
    Thread.join();

  for (std::exception_ptr &Error : Errors)
    if (Error)
      std::rethrow_exception(Error);
}
