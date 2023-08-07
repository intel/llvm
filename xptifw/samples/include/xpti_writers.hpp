//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
#pragma once
#include "xpti_timers.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace xpti {
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
  uint64_t Flags = 1;   // Initialilze it to InvalidRecord
  std::string Name;     // Name of the scoped call
  std::string Category; // Category of the call, usually stream name
};

class writer {
public:
  using rec_tuple = std::pair<uint64_t, record_t>;
  using records_t = std::vector<rec_tuple>;
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
          m_io
              << "{\"name\": \"\", \"cat\": \"\", \"ph\": \"\", \"pid\": \"\", "
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

class table_model {
public:
  using table_row_t = std::map<int, long double>;
  using table_t = std::map<int, table_row_t>;
  using titles_t = std::vector<std::string>;
  using row_titles_t = std::map<int, std::string>;

  table_model() {}

  void set_headers(titles_t &Titles) { m_column_titles = Titles; }

  table_row_t &add_row(int Row, std::string &RowName) {
    if (m_row_titles.count(Row)) {
      std::cout << "Warning: Row title already specified!\n";
    }
    m_row_titles[Row] = RowName;
    return m_table[Row];
  }

  table_row_t &add_row(int Row, const char *RowName) {
    if (m_row_titles.count(Row)) {
      std::cout << "Warning: Row title already specified!\n";
    }
    m_row_titles[Row] = RowName;
    return m_table[Row];
  }

  table_row_t &operator[](int Row) { return m_table[Row]; }

  void print() {
    std::cout << std::setw(35) << " ";
    for (auto &Title : m_column_titles) {
      std::cout << std::setw(14) << Title; // Column headers
    }
    std::cout << "\n";

    for (auto &Row : m_table) {
      std::cout << std::setw(35) << m_row_titles[Row.first];
      for (auto &Data : Row.second) {
        std::cout << std::fixed << std::setw(14) << std::setprecision(0)
                  << Data.second;
      }
      std::cout << "\n";
    }
    std::cout << "\n";
  }

private:
  titles_t m_column_titles;
  row_titles_t m_row_titles;
  table_t m_table;
};

class table_writer : public writer {
public:
  using function_stats_t =
      std::unordered_map<std::string, xpti::utils::statistics_t>;
  using stats_tuple_t = std::pair<std::string, xpti::utils::statistics_t>;
  using duration_stats_t = std::multimap<double, stats_tuple_t>;

  table_writer() { init(); }
  ~table_writer() {}

  void init() final {
    m_file_name = xpti::utils::get_application_name();
    if (m_file_name.empty()) {
      m_file_name =
          "report." + std::to_string(xpti::utils::get_process_id()) + ".csv";
    } else {
      m_file_name += "_" + xpti::utils::timer::get_timestamp_string() + ".csv";
    }
    if (m_file_name.empty()) {
      throw std::runtime_error("Unable to generate file name for write!");
    }
  }

  void write(record_t &r) override {
    std::lock_guard<std::mutex> _{m_mutex};
    m_records.push_back(std::make_pair(r.TSBegin, r));
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
    table_model::titles_t col_headers{
        "Count", "Min (ns)", "Max (ns)", "Mean (ns)", "Std Dev (ns)",
    };
    m_model.set_headers(col_headers);
    for (auto &r : m_records) {
      compute_stats(r.second);
    }
    // Reorder the stats in ascending order of mean times so all small functions
    // are at the top
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

        auto &row = m_model.add_row(row_id++, func_name.c_str());
        row[(int)TableColumns::Count] = count;
        row[(int)TableColumns::Min] = min;
        row[(int)TableColumns::Max] = max;
        row[(int)TableColumns::Mean] = mean;
        row[(int)TableColumns::StandardDeviation] = stddev;
      }
      m_io.close();
      m_model.print();
    } else {
      std::cerr << "Error opening file for write: " << m_file_name << std::endl;
    }
  }

protected:
  std::mutex m_mutex;

private:
  table_model m_model;
  std::string m_file_name;
  std::ofstream m_io;
  records_t m_records;
  function_stats_t m_functions;
  duration_stats_t m_durations;
};
} // namespace xpti
