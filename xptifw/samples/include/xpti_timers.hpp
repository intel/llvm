//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
#pragma once

#include <atomic>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>

namespace xpti {
namespace utils {
/// @brief Statistics class to compute mean, stddev, etc
/// @details This class can compute many staticsical values using running
/// average and related techniques so they can be computed on the fly
/// without any post processing times.
///
/// http://prod.sandia.gov/techlib/access-control.cgi/2008/086212.pdf
///
class statistics_t {
public:
  statistics_t() {
    //
    //  Reset for streaming compute
    //
    clear();

    m_min = std::numeric_limits<double>::max();
    m_max = -std::numeric_limits<double>::max();
  }

  void clear() {
    m_count = 0;
    m_total = 0.0;
    m_moment1 = 0.0;
    m_moment2 = 0.0;
    m_moment3 = 0.0;
    m_moment4 = 0.0;
  }
  void add_value(uint64_t val) {
    double delta, delta_by_n, delta_n_sq, temp;

    uint64_t current_count = m_count;

    if (m_min > val)
      m_min = val;
    if (m_max <= val)
      m_max = val;
    //
    //  We have added a new value, so update the
    //  count by 1
    //
    m_count++;
    m_total += val;
    delta = val - m_moment1;
    delta_by_n = delta / m_count;
    delta_n_sq = delta_by_n * delta_by_n;

    temp = delta * delta_by_n * current_count;
    //
    // Accumulate the mean
    //
    m_moment1 += delta_by_n;
    m_moment4 += temp * delta_n_sq * (m_count * m_count - 3 * m_count + 3) +
                 (6 * delta_n_sq * m_moment2) - (4 * delta_by_n * m_moment3);
    m_moment3 += temp * delta_by_n * (m_count - 2) - 3 * delta_by_n * m_moment2;
    //
    //  Accumulate the average variance
    //
    m_moment2 += temp;
  }

  double total() { return m_total; }
  long count() { return m_count; }
  //
  //
  //
  double mean() { return m_moment1; }

  double max() { return m_max; }

  double min() { return m_min; }

  double variance() {
    if (m_count - 1)
      return m_moment2 / (m_count - 1.0);
    else
      return 0.0;
  }
  double stddev() { return sqrt(variance()); }

  double skewness() {
    return sqrt((double)m_count) * m_moment3 / pow(m_moment2, 1.5);
  }

  double kurtosis() {
    return ((double)m_count * m_moment4) / (m_moment2 * m_moment2) - 3.0;
  }

private:
  uint64_t m_count;
  double m_moment1, m_moment2, m_moment3, m_moment4, m_min, m_max, m_total;
};

namespace timer {
#define MAX_STR_SIZE 2048

std::string get_timestamp_string() {
  auto curr = std::chrono::system_clock::now();
  auto local = std::chrono::system_clock::to_time_t(curr);
  std::stringstream ss;
  ss << std::put_time(std::localtime(&local), "%F_%H-%M-%S");
  return ss.str();
}

#if defined(_WIN32) || defined(_WIN64)
#include "windows.h"
#include <intrin.h>
class measurement_t {
public:
  measurement_t() { m_frequency = frequency(); }
  inline uint64_t clock() {
    LARGE_INTEGER qpcnt;
    int rval = QueryPerformanceCounter(&qpcnt);
    return qpcnt.QuadPart;
  }
  inline uint64_t frequency() {
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    return freq.QuadPart;
  }
  // https://learn.microsoft.com/en-us/windows/win32/sysinfo/acquiring-high-resolution-time-stamps
  inline double microseconds() { (double)(clock()) * 1000000 / m_frequency; }
  inline double clock_to_microsecs(uint64_t clocks) {
    return ((double)(clocks) * 1000000.0 / m_frequency);
  }
#if (defined(__x86_64_) || defined(__i386__) || defined(_i386))
  uint64_t clockticks() { return __rdtsc(); }
#else
  inline uint64_t clockticks() { return clock(); }
#endif
  inline uint64_t cpu() { return GetCurrentProcessorNumber(); }
  inline uint64_t thread() {
    std::hash<std::thread::id>{}(std::this_thread::get_id());
  }

private:
  uint64_t m_frequency;
};
#elif defined(__linux__)
#include <sched.h>
// https://stackoverflow.com/questions/42189976/calculate-system-time-using-rdtsc
// Discussion describes how clock_gettime() costs about 4 ns per call
class measurement_t {
public:
  measurement_t() { m_frequency = frequency(); }
  inline uint64_t clock() {
    struct timespec ts;
    int status = clock_gettime(CLOCK_REALTIME, &ts);
    return (static_cast<uint64_t>(1000000000UL) *
                static_cast<uint64_t>(ts.tv_sec) +
            static_cast<uint64_t>(ts.tv_nsec));
  }
#if defined(__x86_64_) || defined(__i386__) || defined(_i386)
  inline uint64_t clockticks() {
    unsigned int lo, hi;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
  }
#else
  inline uint64_t clockticks() { return clock(); }
#endif

  inline double microseconds() { return (double)clock() / 1000.0; }
  inline uint64_t frequency() { return static_cast<uint64_t>(1E9); }
  inline uint64_t cpu() { return sched_getcpu(); }
  inline uint64_t thread() {
    return std::hash<std::thread::id>{}(std::this_thread::get_id());
  }
  inline double clock_to_microsecs(uint64_t clocks) {
    // since 'clocks' is in nanoseconds, conversion is a simple divide
    return ((double)(clocks) / 1000.0);
  }

private:
  uint64_t m_frequency;
};

#else // For all other platforms, use std::chrono
class measurement_t {
public:
  measurement_t() { m_frequency = frequency(); }
  inline uint64_t clock() {
    auto curr_time = std::chrono::high_resolution_clock::now();
    uint64_t ts =
        std::chrono::time_point_cast<std::chrono::nanoseconds>(curr_time)
            .time_since_epoch()
            .count();
    return ts;
  }

  inline double microseconds() { return (double)clock() / 1000.0; }
  inline uint64_t frequency() { return static_cast<uint64_t>(1E9); }
  inline uint64_t cpu() { return 0; }
  inline uint64_t thread() {
    return std::hash<std::thread::id>{}(std::this_thread::get_id());
  }
  inline double clock_to_microsecs(uint64_t clocks) {
    // since 'clocks' is in nanoseconds, conversion is a simple divide
    return ((double)(clocks) / 1000.0);
  }

private:
  uint64_t m_frequency;
};

#endif

} // namespace timer

#if defined(_WIN32) || defined(_WIN64)
#include <processthreadsapi.h>

inline std::string get_application_name() {
  char buffer[MAX_STR_SIZE] = {0};
  DWORD status = GetModuleFileNameA(nullptr, buffer, MAX_STR_SIZE);
  if (status < 0)
    return "application";
  else {
    std::string path(buffer);
    return path.substr(path.find_last_of("/\\") + 1);
  }
}

inline uint64_t get_process_id() { return GetCurrentProcessId(); }

#elif defined(__linux__)
#include <unistd.h>

inline std::string get_application_name() {
  char buffer[MAX_STR_SIZE] = {0};
  size_t status = readlink("/proc/self/exe", buffer, MAX_STR_SIZE);
  if (status < 0)
    return "application";
  else {
    std::string path(buffer);
    return path.substr(path.find_last_of("/\\") + 1);
  }
}

inline uint64_t get_process_id() { return getpid(); }
#else
#error Unsupported system
#endif

} // namespace utils
} // namespace xpti
