//===--------- latency_tracker.cpp - common ------------------------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>

#include "logger/ur_logger.hpp"

#if defined(UR_ENABLE_LATENCY_HISTOGRAM)

#include <hdr/hdr_histogram.h>

static inline bool trackLatency = []() {
  try {
    auto map = getenv_to_map("UR_LOG_LATENCY");

    if (!map.has_value()) {
      return false;
    }

    auto it = map->find("level");
    return it != map->end() &&
           logger::str_to_level(it->second.front()) != logger::Level::QUIET;
  } catch (...) {
    return false;
  }
}();

static constexpr size_t numPercentiles = 7;
static constexpr double percentiles[numPercentiles] = {
    50.0, 90.0, 99.0, 99.9, 99.99, 99.999, 99.9999};

struct latencyValues {
  int64_t count;
  int64_t min;
  int64_t max;
  int64_t mean;
  int64_t stddev;
  int64_t percentileValues[numPercentiles];
};

using histogram_ptr =
    std::unique_ptr<struct hdr_histogram, decltype(&hdr_close)>;

static inline latencyValues getValues(const struct hdr_histogram *histogram) {
  latencyValues values;
  values.count = histogram->total_count;
  values.max = hdr_max(histogram);
  values.min = hdr_min(histogram);
  values.mean = static_cast<int64_t>(hdr_mean(histogram));
  values.stddev = static_cast<int64_t>(hdr_stddev(histogram));

  auto ret = hdr_value_at_percentiles(histogram, percentiles,
                                      values.percentileValues, numPercentiles);
  if (ret != 0) {
    logger::error("Failed to get percentiles from latency histogram");
  }

  return values;
}

class latency_printer {
public:
  latency_printer() : logger(logger::create_logger("latency", true, false)) {}

  inline void publishLatency(const std::string &name, histogram_ptr histogram) {
    auto [it, inserted] = values.try_emplace(name, std::move(histogram));
    if (!inserted) {
      // combine histograms
      hdr_add(it->second.get(), histogram.get());
    }
  }

  inline ~latency_printer() {
    if (trackLatency) {
      print();
    }
  }

  inline void print() {
    printHeader();

    for (auto &[name, histogram] : values) {
      auto value = getValues(histogram.get());
      auto f = groupDigits<int64_t>;
      logger.log(logger::Level::INFO,
                 "{},{},{},{},{},{},{},{},{},{},{},{},{},{},ns", name,
                 f(value.mean), f(value.percentileValues[0]),
                 f(value.percentileValues[1]), f(value.percentileValues[2]),
                 f(value.percentileValues[3]), f(value.percentileValues[4]),
                 f(value.percentileValues[5]), f(value.percentileValues[6]),
                 f(value.count), f(value.count * value.mean), f(value.min),
                 f(value.max), value.stddev);
    }
  }

private:
  inline void printHeader() {
    logger.log(logger::Level::INFO, "Latency histogram:");
    logger.log(logger::Level::INFO,
               "name,mean,p{},p{},p{},p{},p{},p{}"
               ",p{},count,sum,min,max,stdev,unit",
               percentiles[0], percentiles[1], percentiles[2], percentiles[3],
               percentiles[4], percentiles[5], percentiles[6]);
  }

  std::map<std::string, histogram_ptr> values;
  logger::Logger logger;
};

inline latency_printer &globalLatencyPrinter() {
  static latency_printer printer;
  return printer;
}

class latency_histogram {
public:
  inline latency_histogram(const char *name,
                           latency_printer &printer = globalLatencyPrinter(),
                           int64_t lowestDiscernibleValue = 1,
                           int64_t highestTrackableValue = 100'000'000'000,
                           int significantFigures = 3)
      : name(name), histogram(nullptr, nullptr), printer(printer) {
    if (trackLatency) {
      struct hdr_histogram *cHistogram;
      auto ret = hdr_init(lowestDiscernibleValue, highestTrackableValue,
                          significantFigures, &cHistogram);
      if (ret != 0) {
        logger::error("Failed to initialize latency histogram");
      }
      histogram = std::unique_ptr<struct hdr_histogram, decltype(&hdr_close)>(
          cHistogram, &hdr_close);
    }
  }

  latency_histogram(const latency_histogram &) = delete;
  latency_histogram(latency_histogram &&) = delete;

  inline ~latency_histogram() {
    if (!trackLatency || !histogram) {
      return;
    }

    if (hdr_min(histogram.get()) == std::numeric_limits<int64_t>::max()) {
      logger::info("[{}] latency: no data", name);
      return;
    }

    printer.publishLatency(name, std::move(histogram));
  }

  inline void trackValue(int64_t value) {
    hdr_record_value(histogram.get(), value);
  }

private:
  const char *name;
  histogram_ptr histogram;
  latency_printer &printer;
};

class latency_tracker {
public:
  inline explicit latency_tracker(latency_histogram &stats)
      : stats(trackLatency ? &stats : nullptr), begin() {
    if (trackLatency) {
      begin = std::chrono::steady_clock::now();
    }
  }
  inline latency_tracker() {}
  inline ~latency_tracker() {
    if (stats) {
      auto tp = std::chrono::steady_clock::now();
      auto diffNanos =
          std::chrono::duration_cast<std::chrono::nanoseconds>(tp - begin)
              .count();
      stats->trackValue(static_cast<int64_t>(diffNanos));
    }
  }

  latency_tracker(const latency_tracker &) = delete;
  latency_tracker &operator=(const latency_tracker &) = delete;

  inline latency_tracker(latency_tracker &&rhs) noexcept
      : stats(rhs.stats), begin(rhs.begin) {
    rhs.stats = nullptr;
  }

  inline latency_tracker &operator=(latency_tracker &&rhs) noexcept {
    if (this != &rhs) {
      this->~latency_tracker();
      new (this) latency_tracker(std::move(rhs));
    }
    return *this;
  }

private:
  latency_histogram *stats{nullptr};
  std::chrono::time_point<std::chrono::steady_clock> begin;
};

// To resolve __COUNTER__
#define CONCAT(a, b) a##b

// Each tracker has it's own thread-local histogram.
// At program exit, all histograms for the same scope are
// aggregated.
#define TRACK_SCOPE_LATENCY_CNT(name, cnt)                                     \
  static thread_local latency_histogram CONCAT(histogram, cnt)(name);          \
  latency_tracker CONCAT(tracker, cnt)(CONCAT(histogram, cnt));
#define TRACK_SCOPE_LATENCY(name) TRACK_SCOPE_LATENCY_CNT(name, __COUNTER__)

#else // UR_ENABLE_LATENCY_HISTOGRAM

#define TRACK_SCOPE_LATENCY(name)

#endif // UR_ENABLE_LATENCY_HISTOGRAM
