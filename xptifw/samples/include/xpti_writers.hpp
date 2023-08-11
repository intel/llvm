//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
#pragma once
#include "../../basic_test/cl_processor.hpp"
#include "xpti_timers.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

constexpr bool MeasureEventCost = true;

namespace xpti {
bool ShowInColors = false;
enum class FileFormat {
  JSON = 1,                        // Output in Perfetto UI JSON format
  CSV = 1 << 1,                    // Output summary statistics as a CSV
  Table = 1 << 2,                  // Output summary statistics as a Table
  Stack = 1 << 3,                  // Output call-stack as indented text
  All = JSON | CSV | Table | Stack // Output in all supported formats
};

/// @brief Enumerator defining current state of a record_t
/// @details As a record_t is being populated, the record_t::Flags will contain
/// one or more enum values OR'd together
enum class RecordFlags {
  InvalidRecord = 1, // All record_t's will have when initialized
  /// For it to be a valid record, Name, Category, TSBegin, TSEnd, CorrID, TID
  /// must be set; CpuID and PID are optional.
  ValidRecord = 1 << 2,      // When one or more attributes have been set
  BeginTimePresent = 1 << 3, // Begin time is set
  EndTimePresent = 1 << 4,   // End time is set
  NamePresent = 1 << 5,      // Name is available
  CategoryPresent = 1 << 6,  // Category is available
  TimeInClockticks = 1 << 7, // If the timestamps are captures in clockticks
  TimeInNanoseconds = 1 << 8 // If the timestamps are in nanoseconds
};

/// @brief Record structure representing a scoped call record
/// @details THe record structure encapsulates information from scoped calls or
/// kernel seubmission, execution scopes that can be represented in a timeline
/// and primarily used for creating performance profiles
struct record_t {
  uint64_t TSBegin = 0; // Begin time of a scoped call identified by CorrID
  uint64_t TSEnd = 0;   // End time of a scoped call identified by CorrID
  uint64_t CorrID = 0;  // Correlation ID of the scoped calls
  uint64_t HWID = 0;    // CpuID or HW device ID of execution
  uint64_t TID = 0;     // Thread ID or a logical ID of execution
  uint64_t PID = 0;     // Process ID
  uint64_t UID = 0;     // Universal ID, if available
  uint64_t Flags = 1;   // Initialilze it to InvalidRecord
  std::string Name;     // Name of the scoped call
  std::string Category; // Category of the call, usually stream name
};

using rec_tuple = std::pair<uint64_t, record_t>;
using records_t = std::vector<rec_tuple>;
using ordered_records_t = std::multimap<uint64_t, record_t>;
class span_t;
using spans_t = std::map<uint64_t, span_t>;
/// @brief Span class for managing parent child relationship tasks
/// @detail When we record traces from many streams, we get many overlapping
/// tasks which help us build call stacks or caller/callee relationship
/// information that can be used to analyze efficiency of the call stack
class span_t {
public:
  span_t()
      : m_min(std::numeric_limits<uint64_t>::max()), m_max(0),
        m_label(nullptr) {}
  span_t(uint64_t min, uint64_t max, const char *label)
      : m_min(min), m_max(max), m_label(label) {}
  ~span_t() {}

  /// @brief Checks child tasks to see if one of them is a parent
  /// @param min Min value of the range
  /// @param max Max value of the range
  /// @return The child tasks that completely overlaps the range
  span_t *check_scope(uint64_t min, uint64_t max) {
    // Check to see m_children have a scope that should be the parent of the
    // current time interval
    for (auto &e : m_children) {
      if (e.second.m_min <= min && e.second.m_max >= max) {
        return &e.second;
      }
    }
    // No child overlaps it
    return nullptr;
  }

  /// @brief Insert a new range into the model
  /// @param min The min value of the range
  /// @param max The max value of the range
  /// @param label The label describing the range
  void insert(uint64_t min, uint64_t max, const char *label) {
    // See if the range has a parent scope under which the range should be
    // embedded
    span_t *parent = check_scope(min, max);
    if (parent) {
      // If so, insert the range under the parent scope as a child
      parent->insert(min, max, label);
    } else {
      // If no overlapping parent scope is found (recursively), add it as a
      // child to the scope of the nearest parent
      m_children[min] = span_t(min, max, label);
    }
  }

  /// @brief Experimental recursive print method
  /// @param level The level of the print indicated by indent
  /// @param root The span at the current level
  void print_level(int level, span_t &root, double duration,
                   double build_cost) {
    // Color code level 1's API calls so we can show the cost of the kernels
    // after the first kernel cost is discounted
    std::string bold_color("\033[1;36m");
    std::string time_color("\033[1;32m");
    std::string reset("\033[0m");
    double curr_duration = (root.m_max - root.m_min + 1);
    std::cout << "+";
    for (int i = 0; i < level * 2; ++i)
      std::cout << "-";
    std::cout << "+";
    std::cout << "  "
              << ((std::string(root.m_label).size() > 25)
                      ? (std::string(root.m_label).substr(0, 25) +
                         std::string("..."))
                      : root.m_label);
    if (ShowInColors && level == 1) {
      std::cout << " " << time_color << std::fixed
                << (double)curr_duration / 1000 << " us " << reset;
    } else {
      std::cout << " " << std::fixed << (double)curr_duration / 1000 << " us ";
    }
    std::cout << " [" << std::setprecision(2)
              << (curr_duration / duration * 100) << "%] ";
    if (level == 1) {
      if (ShowInColors) {
        std::cout << bold_color << "   Adjusted: [" << std::setprecision(2)
                  << (curr_duration / (duration - build_cost) * 100) << "%]"
                  << reset;
      } else {
        std::cout << "   Adjusted: [" << std::setprecision(2)
                  << (curr_duration / (duration - build_cost) * 100) << "%]";
      }
    }
    std::cout << "\n";
    for (auto &s : root.m_children) {
      print_level(level + 1, s.second, curr_duration, build_cost);
    }
  }
  void print() {
    double first_time_cost;
    std::cout << "Application:" << m_label << " [" << m_min << "," << m_max
              << "]\n";
    // Find the first submit_impl (or algorithm in the future when "sycl"
    // stream works properly) and remove the time from the submit scope from
    // the application runtime to discount for the build/JIT costs
    for (auto &s : m_children) {
      if (std::string_view(s.second.m_label) == "submit_impl") {
        first_time_cost = s.second.m_max - s.second.m_min + 1;
        break;
      }
    }

    for (auto &s : m_children) {
      print_level(1, s.second, (m_max - m_min + 1), first_time_cost);
    }
  }

  uint64_t m_min, m_max;
  const char *m_label;
  spans_t m_children;
};

class writer {
public:
  virtual void init() = 0;
  virtual void fini() = 0;
  virtual void write(record_t &r) = 0;
};

class json_writer : public writer {
public:
  json_writer(bool synchronous = false)
      : m_synchronous(synchronous), m_write_done(false) {
    init();
  }
  ~json_writer() {}

  void init() final {
    m_file_name = xpti::utils::get_application_name();
    if (m_file_name.empty()) {
      m_file_name =
          "output." + std::to_string(xpti::utils::get_process_id()) + ".json";
    } else {
      m_file_name += "_" + xpti::utils::timer::get_timestamp_string() + ".json";
    }
    std::cout << "Output file name: " << m_file_name << std::endl;
    if (m_synchronous) {
      if (!m_file_name.empty()) {
        std::lock_guard<std::mutex> _{m_mutex};
        m_io.open(m_file_name);
        if (m_io.is_open()) {
          m_io << std::fixed;
          m_io << "{\n";
          m_io << "  \"traceEvents\": [\n";
        } else {
          std::cerr << "Error opening file for write: " << m_file_name
                    << std::endl;
        }
      } else {
        throw std::runtime_error("Unable to generate file name for write!");
      }
    }
  }
  void fini() final {

    std::lock_guard<std::mutex> _{m_mutex};
    if (m_synchronous && m_io.is_open() && !m_write_done) {
      // add an empty element for not ending with '}, ]'
      m_io << "{\"name\": \"\", \"cat\": \"\", \"ph\": \"\", \"pid\": \"\", "
              "\"tid\": \"\", \"ts\": \"\"}\n";

      m_io << "],\n";
      m_io << "\"displayTimeUnit\":\"ns\"\n}\n";
      m_io.close();
      m_write_done = true;
    } else {
      // Burst write when the application exits
      if (!m_io.is_open() && !m_write_done) {
        m_io.open(m_file_name);
        if (m_io.is_open()) {
          m_io << std::fixed;
          m_io << "{\n";
          m_io << "  \"traceEvents\": [\n";
          for (auto &r : m_records) {
            fast_write(r.second);
          }
          // add an empty element for not ending with '}, ]'
          m_io << "{\"name\": \"\", \"cat\": \"\", \"ph\": \"\", \"pid\": "
                  "\"\", "
                  "\"tid\": \"\", \"ts\": \"\"}\n";

          m_io << "],\n";
          m_io << "\"displayTimeUnit\":\"ns\"\n}\n";
          m_io.close();
          m_write_done = true;
        } else {
          std::cerr << "Error opening file for write: " << m_file_name
                    << std::endl;
        }
      }
    }
  }

  void write(record_t &r) override {
    std::lock_guard<std::mutex> _{m_mutex};
    if (m_synchronous && m_io.is_open()) {
      m_io << "{\"name\": \"" << r.Name << "\", ";
      m_io << "\"cat\": \"" << r.Category << "\", ";
      m_io << "\"ph\": \"X\", ";
      m_io << "\"pid\": \"" << r.PID << "\", ";
      m_io << "\"tid\": \"" << r.TID << "\", ";
      m_io << "\"ts\": \"" << m_measure.clock_to_microsecs(r.TSBegin) << "\",";
      m_io << "\"dur\": \"" << m_measure.clock_to_microsecs(r.TSEnd - r.TSBegin)
           << "\"},";
      m_io << std::endl;
    } else {
      m_records.push_back(std::make_pair(r.TSBegin, r));
    }
  }

  void fast_write(record_t &r) {
    m_io << "{\"name\": \"" << r.Name << "\", ";
    m_io << "\"cat\": \"" << r.Category << "\", ";
    m_io << "\"ph\": \"X\", ";
    m_io << "\"pid\": \"" << r.PID << "\", ";
    m_io << "\"tid\": \"" << r.TID << "\", ";
    m_io << "\"ts\": \"" << m_measure.clock_to_microsecs(r.TSBegin) << "\",";
    m_io << "\"dur\": \"" << m_measure.clock_to_microsecs(r.TSEnd - r.TSBegin)
         << "\"},";
    m_io << std::endl;
  }

protected:
  std::mutex m_mutex;

private:
  bool m_synchronous;
  bool m_write_done;
  std::string m_file_name;
  std::ofstream m_io;
  records_t m_records;
  xpti::utils::timer::measurement_t m_measure;
};

using table_model_t = test::utils::TableModel;

class table_writer : public writer {
public:
  using function_stats_t =
      std::unordered_map<std::string, xpti::utils::statistics_t>;
  using stats_tuple_t = std::pair<std::string, xpti::utils::statistics_t>;
  using duration_stats_t = std::multimap<double, stats_tuple_t>;

  table_writer() : m_min(std::numeric_limits<uint64_t>::max()), m_max(0) {
    init();
  }
  ~table_writer() {}

  void init() final {
    m_model = table_model_t(35);
    m_app_name = xpti::utils::get_application_name();
    if (m_app_name.empty()) {
      m_file_name =
          "report." + std::to_string(xpti::utils::get_process_id()) + ".csv";
    } else {
      m_file_name = m_app_name + "_" +
                    xpti::utils::timer::get_timestamp_string() + ".csv";
    }
    if (m_file_name.empty()) {
      throw std::runtime_error("Unable to generate file name for write!");
    }
  }

  void write(record_t &r) override {
    std::lock_guard<std::mutex> _{m_mutex};
    m_records.push_back(std::make_pair(r.TSBegin, r));
    if (m_min > r.TSBegin)
      m_min = r.TSBegin;
    if (m_max < r.TSEnd)
      m_max = r.TSEnd;
  }

  void compute_stats(record_t &r) {
    auto &func_stats = m_functions[r.Name];
    func_stats.add_value((r.TSEnd - r.TSBegin));
  }

  void fini() final {
    std::lock_guard<std::mutex> _{m_mutex};
    enum class TableColumns : int {
      Count,
      Min,
      Max,
      Mean,
      StandardDeviation,
      Skewness,
      Kurtosis
    };
    test::utils::titles_t col_headers{
        "Count", "Min (ns)", "Max (ns)", "Mean (ns)", "Std Dev (ns)",
    };
    m_model.setHeaders(col_headers);
    // Set the root of the call stack tree
    m_root = span_t(m_min, m_max, m_app_name.c_str());
    for (auto &r : m_records) {
      m_ordered_records.insert(std::make_pair(r.second.TSBegin, r.second));
      compute_stats(r.second);
    }
    // Build the tree of trace scopes that mimics a call stack
    for (auto &r : m_ordered_records) {
      m_root.insert(r.second.TSBegin, r.second.TSEnd, r.second.Name.c_str());
    }

    // Reorder the stats in ascending order of mean times so all small
    // functions are at the top
    for (auto &f : m_functions) {
      m_durations.insert(
          std::make_pair(f.second.mean(), std::make_pair(f.first, f.second)));
    }
    // Burst write when the application exits
    if (!m_io.is_open()) {
      m_io.open(m_file_name);
      // Setup CSV file
      m_io << "sep=|\n";
      m_io << "Function|Count|Min|Max|Average|Std. Dev.\n";
      int row_id = 0;
      for (auto &stats : m_durations) {
        uint64_t count = stats.second.second.count();
        double mean = stats.second.second.mean(),
               stddev = stats.second.second.stddev(),
               min = stats.second.second.min(), max = stats.second.second.max();
        // Output CSV line
        m_io << stats.second.first << "|" << count << "|" << min << "|" << max
             << "|" << mean << "|" << stddev << "\n";
        // Record data in table model for pretty printing
        std::string func_name;
        if (stats.second.first.length() > 25) {
          func_name = stats.second.first.substr(0, 30);
          func_name += "...";
        } else
          func_name = stats.second.first;

        auto &row = m_model.addRow(row_id++, func_name.c_str());
        row[(int)TableColumns::Count] = count;
        row[(int)TableColumns::Min] = min;
        row[(int)TableColumns::Max] = max;
        row[(int)TableColumns::Mean] = mean;
        row[(int)TableColumns::StandardDeviation] = stddev;
      }
      m_io.close();
      m_model.print();
      m_root.print();
    } else {
      std::cerr << "Error opening file for write: " << m_file_name << std::endl;
    }
  }

protected:
  std::mutex m_mutex;

private:
  table_model_t m_model;
  std::string m_file_name;
  std::string m_app_name;
  std::ofstream m_io;
  records_t m_records;
  ordered_records_t m_ordered_records;
  function_stats_t m_functions;
  duration_stats_t m_durations;
  uint64_t m_min, m_max;
  span_t m_root;
};
} // namespace xpti
