//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
#pragma once
#include "xpti_helpers.hpp"
#include "xpti_timers.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace xpti {
/// @brief Enumerator defining output formats
/// @details Defines the supported output format types which will be used by the
/// writer classes
enum class FileFormat {
  JSON = 1,                  // Output in Perfetto UI JSON format
  Table = 1 << 1,            // Output summary statistics as a Table/CSV
  Stack = 1 << 2,            // Output call-stack as indented text
  All = JSON | Table | Stack // Output in all supported formats
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
  char *Name;           // Name of the scoped call
  std::string Category; // Category of the call, usually stream name
};

using table_model_t = xpti::utils::TableModel;
using records_t = std::vector<record_t>;
using ordered_records_t = std::multimap<uint64_t, record_t>;
class span_t;
using spans_t = std::map<uint64_t, span_t>;

/// @brief Data model definition
class data_model {
private:
  records_t m_records;
  ordered_records_t m_ordered_records;
  int64_t m_count = 0;
  uint64_t m_min, m_max;

public:
  data_model();
  ~data_model();
  void add(record_t &r);
  void finalize();
  uint64_t min();
  uint64_t max();
  records_t &data();
  ordered_records_t &ordered_data();
  void print(char *Name = nullptr);
};

using data_model_ptr = std::shared_ptr<data_model>;
class span_t;
using spans_t = std::map<uint64_t, span_t>;

/// @brief Writer interface definition
class writer {
public:
  virtual void init() = 0;
  virtual void fini() = 0;
};

/// @brief Span class for managing parent child relationship tasks
/// @detail When we record traces from many streams, we get many overlapping
/// tasks which help us build call stacks or caller/callee relationship
/// information that can be used to analyze efficiency of the call stack
class span_t {
public:
  enum class TableColumns : int {
    Time,
    PercentParent,
    PercentTotal,
    PercentTotalAdj,
    PercentEmptyAdj
  };

  span_t()
      : m_min(std::numeric_limits<uint64_t>::max()), m_max(0), m_label(nullptr),
        m_time(0), m_per_parent(0), m_per_total(0), m_per_tot_adj(0),
        m_per_empty_adj(0) {
    m_model = table_model_t(50, 3);
  }
  span_t(uint64_t min, uint64_t max, const char *label)
      : m_min(min), m_max(max), m_label(label), m_time(0), m_per_parent(0),
        m_per_total(0), m_per_tot_adj(0), m_per_empty_adj(0) {
    m_model = table_model_t(50, 3);
  }
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

  void clip_to_steady_state() {
    size_t record_count = 0, skip_count = 0;
    std::string record_marker("submit_impl");
    for (auto &s : m_children) {
      if (record_marker == s.second.m_label)
        record_count++;
    }
    std::cout << "Records found with '" << record_marker
              << "': " << record_count << "\n";
    // For steady state, we will utilize only half of the records by ignoring
    // the first half
    if (record_count > 5) {
      skip_count = record_count / 3;
      // Clip the data model to one half of the records
      record_count = 0;
      spans_t::iterator it, limit;
      uint64_t new_min;
      for (it = m_children.begin(); it != m_children.end(); ++it) {
        if (record_marker == it->second.m_label) {
          if (record_count < skip_count) {
            record_count++;
          } else {
            limit = it;
            new_min = it->first;
            break;
          }
        }
      }
      m_children.erase(m_children.begin(), limit);
      m_min = new_min;
    }
  }

  /// @brief Experimental recursive print method
  /// @param level The level of the print indicated by indent
  /// @param root The span at the current level
  void print_level(int level, span_t &root, double parent_duration,
                   double total_duration, double ignore_cost,
                   bool show_in_colors = false) {
    // Color code level 1's API calls so we can show the cost of the kernels
    // after the first kernel cost is discounted
    std::string bold_color("\033[1;36m");
    std::string time_color("\033[1;32m");
    std::string reset("\033[0m");

    double curr_duration = (root.m_max - root.m_min + 1);
    std::cout << "+";
    for (int i = 0; i < level; ++i)
      std::cout << "-";
    std::cout << "+";
    std::cout << "  "
              << ((std::string(root.m_label).size() > 25)
                      ? (std::string(root.m_label).substr(0, 25) +
                         std::string("..."))
                      : root.m_label);
    // Print the times recorded [%parent, %total, %total_adjusted]
    if (show_in_colors) {
      std::cout << " " << time_color << std::fixed << root.m_time << " us "
                << reset;
      std::cout << " [" << std::setprecision(2) << root.m_per_parent << "%, "
                << root.m_per_total << "%, " << time_color << root.m_per_tot_adj
                << reset << "%, " << root.m_per_empty_adj << "%]";
    } else {
      std::cout << " " << std::fixed << root.m_time << " us ";
      std::cout << " [" << std::setprecision(2) << root.m_per_parent << "%, "
                << root.m_per_total << "%, " << root.m_per_tot_adj << "%, "
                << root.m_per_empty_adj << "%]";
    }
    std::cout << "\n";
    for (auto &s : root.m_children) {
      print_level(level + 1, s.second, curr_duration, total_duration,
                  ignore_cost);
    }
  }

  void print(bool show_in_colors = false) {
    double first_time_cost;
    std::cout << "Application:" << m_label << " [" << m_min << "," << m_max
              << "]\n";
    std::cout << "\n\nDecoder for information printed:\n";
    std::cout << "1: Level in the stack [+-+ root level, +--+ level 1]\n";
    std::cout << "2: Function Name\n";
    std::cout << "3: Duration of the function in microseconds\n";
    std::cout << "4: Percentage of parent\n";
    std::cout << "5: Percentage of total\n";
    std::cout << "6: Adjusted percentages (removing the function's first "
                 "occurrence as provided in the ignore list)\n";
    std::cout << "7: Self time pecentage\n\n";
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
      print_level(1, s.second, (m_max - m_min + 1), (m_max - m_min + 1),
                  first_time_cost, show_in_colors);
    }
  }

  /// @brief Scans the call stack information and gets the ignore time
  /// @param root The node to compute the ignore time, if valid
  /// @param str_map The list of fuction calls to ignore on first occurrence
  /// @return Total time to ignore or not account for from the total runtime
  double ignore_time(span_t &root,
                     xpti::utils::string::first_check_map_t *str_map) {
    double ignore_time_acc = 0.0;
    // Scan all the children of the current node and see if any of the children
    // are in the ignore list and happen to be the first occurrence of the
    // function call
    for (auto &s : root.m_children) {
      if (str_map->check(s.second.m_label)) {
        ignore_time_acc += s.second.m_max - s.second.m_min + 1;
      }
      // Recurse to curren't node's childen
      ignore_time_acc += ignore_time(s.second, str_map);
    }
    return ignore_time_acc;
  }

  /// @brief Finds the ignore time
  /// @param str_map THe list of functions to ignore on first occurrence
  /// @return The ignore time for all functions in the ignore list
  double find_ignore_time(xpti::utils::string::first_check_map_t *str_map) {
    return ignore_time(*this, str_map);
  }

  /// @brief Compute the statistics for a ffive node in the call stack
  /// @param level  The current level in the call stack
  /// @param root The node in the current level to compute stats for
  /// @param parent_duration The duration of parent scope
  /// @param total_duration THe duration of application scope
  /// @param ignore_cost The total time to ignore from stats
  void compute_level(int level, span_t &root, double parent_duration,
                     double total_duration, double ignore_cost) {
    double recorded_time = 0.0;
    double curr_duration = (root.m_max - root.m_min + 1);
    root.m_time = curr_duration / 1000.0;
    root.m_per_parent = curr_duration / parent_duration * 100.0;
    root.m_per_total = curr_duration / total_duration * 100.0;
    root.m_per_tot_adj = curr_duration / (total_duration - ignore_cost) * 100.0;

    for (auto &s : root.m_children) {
      recorded_time += (s.second.m_max - s.second.m_min + 1);
      compute_level(level + 1, s.second, curr_duration, total_duration,
                    ignore_cost);
    }
    if (root.m_children.size())
      root.m_per_empty_adj =
          (curr_duration - recorded_time) / (parent_duration) * 100.0;
    else
      root.m_per_empty_adj = 0;
  }
  /// @brief Compute the required metrics to compare runs
  /// @param str_map  Functions calls to ignore first occurrences of
  void compute_metrics(xpti::utils::string::first_check_map_t *str_map) {
    // Clip the data to steady state first
    clip_to_steady_state();
    double ignore_time = find_ignore_time(str_map);
    std::cout << "Time to be ignored: " << std::fixed << std::setprecision(3)
              << ignore_time << "\n";
    double curr_duration = (m_max - m_min + 1);
    m_time = curr_duration / 1000.0;
    m_per_parent = 100.0;
    m_per_total = 100.0;
    m_per_tot_adj = 100.0;

    for (auto &s : m_children) {
      compute_level(1, s.second, (m_max - m_min + 1), (m_max - m_min + 1),
                    ignore_time);
    }
  }

  /// @brief Build a table row
  /// @param level Currentl level operating in
  /// @param root The current node in the stack
  void add_to_table(int level, span_t &root) {
    std::string stack_depth;
    for (int i = 1; i < level; ++i)
      stack_depth += "--";
    stack_depth += "+";

    std::string label =
        (std::string(root.m_label).size() > 25)
            ? (std::string(root.m_label).substr(0, 25) + std::string("..."))
            : std::string(root.m_label);
    label += stack_depth;
    auto &row = m_model.addRow(m_row++, label.c_str());
    row[(int)TableColumns::Time] = root.m_time;
    row[(int)TableColumns::PercentParent] = root.m_per_parent;
    row[(int)TableColumns::PercentTotal] = root.m_per_total;
    row[(int)TableColumns::PercentTotalAdj] = root.m_per_tot_adj;
    row[(int)TableColumns::PercentEmptyAdj] = root.m_per_empty_adj;

    for (auto &s : root.m_children) {
      add_to_table(level + 1, s.second);
    }
  }

  void print_table(bool debug = false, bool show_in_colors = false) {
    if (debug) {
      // The columns being computed for the call stack
      xpti::utils::titles_t col_headers{"Time(us)", "% Parent", "% Total",
                                        "% Tot(Adj)", "% Empty(Dep)"};
      m_model.setHeaders(col_headers);
      for (auto &s : m_children) {
        add_to_table(1, s.second);
      }
      m_model.print();
    }
  }

  uint64_t m_min, m_max;
  uint32_t m_row = 0;
  double m_time, m_per_parent, m_per_total, m_per_tot_adj, m_per_empty_adj;
  const char *m_label;
  table_model_t m_model;
  spans_t m_children;
  std::vector<double> m_simulation;
};

/// @brief  A class that implements a summary table writer
/// @details This class computes the mean and standard deviations for all
/// functions encountered during the run and presents them in table format
class table_writer : public writer {
public:
  using function_stats_t =
      std::unordered_map<std::string, xpti::utils::statistics_t>;
  using stats_tuple_t = std::pair<std::string, xpti::utils::statistics_t>;
  using duration_stats_t = std::multimap<double, stats_tuple_t>;

  table_writer(xpti::utils::string::first_check_map_t *list,
               xpti::data_model *model, bool debug_flag = false)
      : m_ignore(list), m_data_model(model), m_debug(debug_flag),
        m_min(std::numeric_limits<uint64_t>::max()), m_max(0) {
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

  void compute_stats(record_t &r) {
    if (m_ignore->check(r.Name))
      return;
    auto &func_stats = m_functions[r.Name];
    func_stats.add_value((r.TSEnd - r.TSBegin));
  }

  void fini() final {
    if (!m_data_model)
      return;
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
    xpti::utils::titles_t col_headers{
        "Count", "Min (ns)", "Max (ns)", "Mean (ns)", "Std Dev (ns)",
    };
    m_model.setHeaders(col_headers);
    auto data = m_data_model->ordered_data();
    for (auto &r : data) {
      compute_stats(r.second);
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
      if (m_debug)
        m_model.print();
    } else {
      std::cerr << "Error opening file for write: " << m_file_name << std::endl;
    }
  }

protected:
  std::mutex m_mutex;

private:
  xpti::utils::string::first_check_map_t *m_ignore;
  xpti::data_model *m_data_model = nullptr;
  bool m_debug;
  table_model_t m_model;
  std::string m_file_name;
  std::string m_app_name;
  std::ofstream m_io;
  function_stats_t m_functions;
  duration_stats_t m_durations;
  uint64_t m_min, m_max;
};

/// @brief  A class that implements a stack format writer
/// @details This class organizes the data in stack format by looking at nested
/// elements and uses the span_t class to present them in text format output.
class stack_writer : public writer {
public:
  using function_stats_t =
      std::unordered_map<std::string, xpti::utils::statistics_t>;
  using stats_tuple_t = std::pair<std::string, xpti::utils::statistics_t>;
  using duration_stats_t = std::multimap<double, stats_tuple_t>;

  stack_writer(xpti::utils::string::first_check_map_t *list,
               xpti::data_model *model, bool debug_flag = false,
               bool show_in_colors = false)
      : m_ignore(list), m_data_model(model), m_debug(debug_flag),
        m_show_in_colors(show_in_colors) {
    init();
  }
  ~stack_writer() {}

  void init() final {
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

  void fini() final {
    std::lock_guard<std::mutex> _{m_mutex};
    m_min = m_data_model->min();
    m_max = m_data_model->max();
    // Set the root of the call stack tree
    m_root = span_t(m_min, m_max, m_app_name.c_str());
    auto ordered_data = m_data_model->ordered_data();
    // Build the tree of trace scopes that mimics a call stack
    for (auto &r : ordered_data) {
      m_root.insert(r.second.TSBegin, r.second.TSEnd, r.second.Name);
    }

    m_root.compute_metrics(m_ignore);
    m_root.print(m_show_in_colors);
    m_root.print_table();
  }

protected:
  std::mutex m_mutex;

private:
  xpti::utils::string::first_check_map_t *m_ignore;
  xpti::data_model *m_data_model = nullptr;
  bool m_debug;
  bool m_show_in_colors;
  std::string m_file_name;
  std::string m_app_name;
  std::ofstream m_io;
  uint64_t m_min, m_max;
  span_t m_root;
};

/// @brief JSON writer that implements the writer interface
class json_writer : public writer {
public:
  json_writer(xpti::data_model *model)
      : m_data_model(model), m_write_done(false) {
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
  }
  void fini() final {

    std::lock_guard<std::mutex> _{m_mutex};
    // Burst write when the application exits
    if (!m_io.is_open() && !m_write_done) {
      m_io.open(m_file_name);
      if (m_io.is_open()) {
        m_io << std::fixed;
        m_io << "{\n";
        m_io << "  \"traceEvents\": [\n";
        auto records = m_data_model->data();
        for (auto &r : records) {
          fast_write(r);
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
  xpti::data_model *m_data_model = nullptr;
  bool m_write_done;
  std::string m_file_name;
  std::ofstream m_io;
  xpti::utils::timer::measurement_t m_measure;
};
} // namespace xpti