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

#if defined(_WIN32) || defined(_WIN64)
// Fixes issue with std::max() and std::min()
// https://stackoverflow.com/questions/61921246/is-include-windows-h-bad-practice
#define NOMINMAX
#include <intrin.h>
#include <processthreadsapi.h>
#include <windows.h>
#endif

namespace xpti {
namespace utils {

/// @def MAX_STR_SIZE
/// @brief The maximum size of a string in the XPTI framework utils
/// functionality.
///
/// This variable defines the maximum size of a string collector functionality.
/// It is primarily used to limit overflow in legacy APIs.
///
constexpr size_t MAX_STR_SIZE = 2048;

/// @class statistics_t
/// @brief A class for computing and storing statistical data.
///
/// This class is used to compute and store statistical data such as mean,
/// variance, skewness, and kurtosis. It also keeps track of the minimum and
/// maximum values, the total of all values, and the count of values. The
/// staticsical values using running average and related techniques so they can
/// be computed on the fly without any post processing times.
///
/// http://prod.sandia.gov/techlib/access-control.cgi/2008/086212.pdf
///
class statistics_t {
public:
  /// @brief Constructor that initializes the statistics object.
  ///
  /// The constructor initializes all state data to 0, the minimum value to the
  /// maximum possible double value, and the maximum value to the smallest
  /// possible double value.
  ///
  statistics_t() {
    clear();
    m_min = std::numeric_limits<double>::max();
    m_max = -std::numeric_limits<double>::max();
  }

  /// @brief Resets all statistical data.
  ///
  /// This function resets all moments to 0, allowing the object to be reused
  /// for statistical data.
  ///
  void clear() {
    m_count = 0;
    m_total = 0.0;
    m_moment1 = 0.0;
    m_moment2 = 0.0;
    m_moment3 = 0.0;
    m_moment4 = 0.0;
  }

  /// @brief Adds a value to the statistical data object.
  ///
  /// This function adds a value to the statistical data which allows it to
  /// compute internal state and avereages as running means
  /// @param val The value to add.
  ///
  void add_value(uint64_t val) {
    double delta, delta_by_n, delta_n_sq, temp;

    uint64_t current_count = m_count;

    if (m_min > val)
      m_min = val;
    if (m_max <= val)
      m_max = val;
    //  We have added a new value, so update the count by 1
    m_count++;
    m_total += val;
    delta = val - m_moment1;
    delta_by_n = delta / m_count;
    delta_n_sq = delta_by_n * delta_by_n;

    temp = delta * delta_by_n * current_count;
    // Accumulate the mean
    m_moment1 += delta_by_n;
    m_moment4 += temp * delta_n_sq * (m_count * m_count - 3 * m_count + 3) +
                 (6 * delta_n_sq * m_moment2) - (4 * delta_by_n * m_moment3);
    m_moment3 += temp * delta_by_n * (m_count - 2) - 3 * delta_by_n * m_moment2;
    //  Accumulate the average variance
    m_moment2 += temp;
  }

  /// @brief Returns the total of all values.
  /// @return The total of all values.
  double total() { return m_total; }

  /// @brief Returns the count of values.
  /// @return The count of values.
  long count() { return m_count; }

  /// @brief Returns the mean of the values.
  /// @return The mean of the values.
  double mean() { return m_moment1; }

  /// @brief Returns the maximum value.
  /// @return The maximum value.
  double max() { return m_max; }

  /// @brief Returns the minimum value.
  /// @return The minimum value.
  double min() { return m_min; }

  /// @brief Returns the variance of the values.
  /// @return The variance of the values.
  double variance() {
    if (m_count - 1)
      return m_moment2 / (m_count - 1.0);
    else
      return 0.0;
  }

  /// @brief Returns the standard deviation of the values.
  /// @return The standard deviation of the values.
  double stddev() { return sqrt(variance()); }

  /// @brief Returns the skewness of the values.
  /// @return The skewness of the values.
  double skewness() {
    return sqrt((double)m_count) * m_moment3 / pow(m_moment2, 1.5);
  }

  /// @brief Returns the kurtosis of the values.
  /// @return The kurtosis of the values.
  double kurtosis() {
    return ((double)m_count * m_moment4) / (m_moment2 * m_moment2) - 3.0;
  }

private:
  uint64_t m_count; ///< The count of values.
  double m_moment1, ///< The first moment (mean).
      m_moment2,    ///< The second moment (variance).
      m_moment3,    ///< The third moment (skewness).
      m_moment4,    ///< The fourth moment (kurtosis).
      m_min,        ///< The minimum value.
      m_max,        ///< The maximum value.
      m_total;      ///< The total of all values.
};

namespace timer {

/// @brief Returns the current timestamp as a string.
///
/// This function gets the current time using the system clock, converts it to
/// local time, and then formats it as a string in the format
/// "YYYY-MM-DD_HH-MM-SS". This strimng will be used by the perf collector to
/// dump the collected data into a unique filename.
///
/// @return A string representing the current timestamp.
///
std::string get_timestamp_string() {
  auto curr = std::chrono::system_clock::now();
  auto local = std::chrono::system_clock::to_time_t(curr);
  std::stringstream ss;
  ss << std::put_time(std::localtime(&local), "%F_%H-%M-%S");
  return ss.str();
}

/// @class measurement_t
/// @brief A class for high-resolution time measurement.
///
/// This class provides methods for measuring time with high resolution. It uses
/// different methods for time measurement depending on the platform and the
/// available hardware.
///
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
    return ((double)(clocks)*1000000.0 / m_frequency);
  }
  inline uint64_t clockticks() {
#if defined(__x86_64_) || defined(__i386__) || defined(_i386)
    unsigned int lo, hi;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
#else
    return clock();
#endif
  }
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

// @brief Returns the name of the current application.
//
// This function retrieves the full path of the executable file of the current
// process, and then extracts the file name from the path. If the retrieval
// fails, it returns "application" as a default name. It employs different APIs
// to get the application name on Windows and Linux
//
// @return A string representing the name of the current application.
//
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

/// @brief Returns the ID of the current process.
///
/// This function uses the Windows API function GetCurrentProcessId to get the
/// ID of the current process.
///
/// @return The ID of the current process.
///
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

/// @brief Returns the ID of the current process.
///
/// This function uses the POSIX function getpid to get the ID of the current
/// process.
///
/// @return The ID of the current process.
///
inline uint64_t get_process_id() { return getpid(); }
#else
///
/// @brief Returns a default name for the application.
///
/// This function always returns "application" as the name of the application.
/// It can be used when the call is being made on an unsupported platform
///
/// @return A string representing the name of the application.
///
#include <stdlib.h>
#include <time.h>

static bool g_initialized = false;
inline std::string get_application_name() { return "application"; }
///
/// @brief Returns the ID of the current process.
///
/// This function returns a random number as the ID of the current process when
/// we are on an unsupported platform
///
/// @return The ID of the current process.
///
inline uint64_t get_process_id() {
  if (!g_initialized) {
    srand(time(0));
    g_initialized = true;
  }
  return rand();
}
#endif

} // namespace utils
} // namespace xpti
