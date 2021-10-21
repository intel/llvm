//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
#pragma once

#include <atomic>
#include <sstream>
#include <thread>
#include <unordered_map>

namespace xpti {
class ThreadID {
public:
  using thread_lut_t = std::unordered_map<std::string, int>;

  inline uint32_t enumID(std::thread::id &curr) {
    std::stringstream s;
    s << curr;
    std::string str(s.str());

    if (m_thread_lookup.count(str)) {
      return m_thread_lookup[str];
    } else {
      uint32_t enumID = m_tid++;
      m_thread_lookup[str] = enumID;
      return enumID;
    }
  }

  inline uint32_t enumID(const std::string &curr) {
    if (m_thread_lookup.count(curr)) {
      return m_thread_lookup[curr];
    } else {
      uint32_t enumID = m_tid++;
      m_thread_lookup[curr] = enumID;
      return enumID;
    }
  }

private:
  std::atomic<uint32_t> m_tid = {0};
  thread_lut_t m_thread_lookup;
};

namespace timer {
#include <cstdint>
using tick_t = uint64_t;
#if defined(_WIN32) || defined(_WIN64)
#include "windows.h"
inline xpti::timer::tick_t rdtsc() {
  LARGE_INTEGER qpcnt;
  int rval = QueryPerformanceCounter(&qpcnt);
  return qpcnt.QuadPart;
}
inline uint64_t getTSFrequency() {
  LARGE_INTEGER freq;
  QueryPerformanceFrequency(&freq);
  return freq.QuadPart * 1000;
}
inline uint64_t getCPU() { return GetCurrentProcessorNumber(); }
#else
#include <sched.h>
#include <time.h>
#if __x86_64__ || __i386__ || __i386
inline xpti::timer::tick_t rdtsc() {
  struct timespec ts;
  int status = clock_gettime(CLOCK_REALTIME, &ts);
  (void)status;
  return (static_cast<tick_t>(1000000000UL) * static_cast<tick_t>(ts.tv_sec) +
          static_cast<tick_t>(ts.tv_nsec));
}

inline uint64_t getTSFrequency() { return static_cast<uint64_t>(1E9); }

inline uint64_t getCPU() {
#ifdef __linux__
  return sched_getcpu();
#else
  return 0;
#endif
}
#else
#error Unsupported ISA
#endif

inline std::thread::id getThreadID() { return std::this_thread::get_id(); }
#endif
} // namespace timer
} // namespace xpti
