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
class thread_id {
public:
  typedef std::unordered_map<std::string, int> thread_lut_t;

  thread_id() : m_tid(0) {}
  ~thread_id() {}

  inline uint32_t enum_id(std::thread::id &curr) {
    std::stringstream s;
    s << curr;
    std::string str(s.str());

    if (m_thread_lookup.count(str)) {
      return m_thread_lookup[str];
    } else {
      uint32_t enum_id = m_tid++;
      m_thread_lookup[str] = enum_id;
      return enum_id;
    }
  }

  inline uint32_t enum_id(const std::string &curr) {
    if (m_thread_lookup.count(curr)) {
      return m_thread_lookup[curr];
    } else {
      uint32_t enum_id = m_tid++;
      m_thread_lookup[curr] = enum_id;
      return enum_id;
    }
  }

private:
  std::atomic<uint32_t> m_tid;
  thread_lut_t m_thread_lookup;
};

namespace timer {
#include <cstdint>
typedef uint64_t tick_t;
#if defined(_WIN32) || defined(_WIN64)
#include "windows.h"
inline xpti::timer::tick_t rdtsc() {
  LARGE_INTEGER qpcnt;
  int rval = QueryPerformanceCounter(&qpcnt);
  return qpcnt.QuadPart;
}
inline uint64_t get_ts_frequency() {
  LARGE_INTEGER freq;
  QueryPerformanceFrequency(&freq);
  return freq.QuadPart * 1000;
}
inline uint64_t get_cpu() { return GetCurrentProcessorNumber(); }
#else
#include <sched.h>
#include <time.h>
#if __x86_64__ || __i386__ || __i386
inline xpti::timer::tick_t rdtsc() {
  struct timespec ts;
  int status = clock_gettime(CLOCK_REALTIME, &ts);
  return (static_cast<tick_t>(1000000000UL) * static_cast<tick_t>(ts.tv_sec) +
          static_cast<tick_t>(ts.tv_nsec));
}

inline uint64_t get_ts_frequency() { return static_cast<uint64_t>(1E9); }

inline uint64_t get_cpu() {
#ifdef __linux__
  return sched_getcpu();
#else
  return 0;
#endif
}
#else
#error Unsupported ISA
#endif

inline std::thread::id get_thread_id() { return std::this_thread::get_id(); }
#endif
} // namespace timer
} // namespace xpti