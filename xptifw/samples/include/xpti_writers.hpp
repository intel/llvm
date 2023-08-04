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
  virtual void init() = 0;
  virtual void fini() = 0;
  virtual void write(record_t &r) = 0;
};

class json_writer : public writer {
public:
  using rec_tuple = std::pair<uint64_t, record_t>;
  using records_t = std::vector<rec_tuple>;
  json_writer(bool synchronous = false)
      : m_synchronous(synchronous), m_write_done(false) {
    init();
  }
  ~json_writer() {}

  void init() final {
    std::cout << "Created JSON Writer\n";
    m_file_name = xpti::utils::get_application_name();
    if (m_file_name.empty()) {
      m_file_name =
          "output." + std::to_string(xpti::utils::get_process_id()) + ".json";
    } else {
      m_file_name += "_" + xpti::utils::timer::get_timestamp_string() + ".json";
    }
    std::cout << "Output file name: " << m_file_name << std::endl;
    if (m_synchronous) {
      std::cout << "In synchronous mode\n";
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
    std::cout << "Dumping records and closing JSON Writer\n";

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
        std::cout << "In burst mode\n";
        m_io.open(m_file_name);
        if (m_io.is_open()) {
          m_io << std::fixed;
          // sort the vector first
          // auto compare = [&](rec_tuple &a, rec_tuple &b) {
          //   return a.first < b.first;
          // };
          // std::sort(m_records.begin(), m_records.end(), compare);
          std::cout << "Writing header\n";
          m_io << "{\n";
          m_io << "  \"traceEvents\": [\n";
          std::cout << "Writing records\n";
          for (auto &r : m_records) {
            fast_write(r.second);
          }
          std::cout << "Writing footer\n";
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
} // namespace xpti
