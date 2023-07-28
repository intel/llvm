//==----------------- writer.hpp -------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <fstream>
#include <iomanip>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

struct record_t {
  std::string name;
  std::string cat;
  std::string phase;
  size_t PID;
  size_t TID;
  double TS;
};

class Writer {
public:
  virtual void init() = 0;
  virtual void finalize() = 0;
  virtual void writeBegin(std::string_view Name, std::string_view Category,
                          size_t PID, size_t TID, size_t TimeStamp) = 0;
  virtual void writeEnd(std::string_view Name, std::string_view Category,
                        size_t PID, size_t TID, size_t TimeStamp) = 0;
  virtual void writeBufferBegin(std::string_view Name,
                                std::string_view Category, size_t PID,
                                size_t TID, size_t TimeStamp) = 0;
  virtual void writeBufferEnd(std::string_view Name, std::string_view Category,
                              size_t PID, size_t TID, size_t TimeStamp) = 0;
  virtual void writeRecord(record_t &r) = 0;
  virtual ~Writer() = default;
};

class JSONWriter : public Writer {
public:
  explicit JSONWriter(const std::string &OutPath) : MOutFile(OutPath) {}

  void init() final {
    std::lock_guard<std::mutex> _{MWriteMutex};

    MOutFile << std::fixed << std::setprecision(3);
    MOutFile << "{\n";
    MOutFile << "  \"traceEvents\": [\n";
  }

  void writeBufferBegin(std::string_view Name, std::string_view Category,
                        size_t PID, size_t TID, size_t TimeStamp) override {
    std::lock_guard<std::mutex> _{MWriteMutex};

    record_t r;
    r.name = Name;
    r.cat = Category;
    r.phase = "B";
    r.PID = PID;
    r.TID = TID;
    r.TS = (double)TimeStamp / 1000;
    MRecords.push_back(r); // should use emplace_back
  }

  void writeBufferEnd(std::string_view Name, std::string_view Category,
                      size_t PID, size_t TID, size_t TimeStamp) override {
    std::lock_guard<std::mutex> _{MWriteMutex};

    record_t r;
    r.name = Name;
    r.cat = Category;
    r.phase = "E";
    r.PID = PID;
    r.TID = TID;
    r.TS = (double)TimeStamp / 1000; // microseconds
    MRecords.push_back(r);           // should use emplace_back
  }

  void writeBegin(std::string_view Name, std::string_view Category, size_t PID,
                  size_t TID, size_t TimeStamp) override {
    std::lock_guard<std::mutex> _{MWriteMutex};

    if (!MOutFile.is_open())
      return;

    MOutFile << "{\"name\": \"" << Name << "\", ";
    MOutFile << "\"cat\": \"" << Category << "\", ";
    MOutFile << "\"ph\": \"B\", ";
    MOutFile << "\"pid\": \"" << PID << "\", ";
    MOutFile << "\"tid\": \"" << TID << "\", ";
    MOutFile << "\"ts\": \"" << TimeStamp << "\"},";
    MOutFile << std::endl;
  }

  void writeEnd(std::string_view Name, std::string_view Category, size_t PID,
                size_t TID, size_t TimeStamp) override {
    std::lock_guard<std::mutex> _{MWriteMutex};

    if (!MOutFile.is_open())
      return;

    MOutFile << "{\"name\": \"" << Name << "\", ";
    MOutFile << "\"cat\": \"" << Category << "\", ";
    MOutFile << "\"ph\": \"E\", ";
    MOutFile << "\"pid\": \"" << PID << "\", ";
    MOutFile << "\"tid\": \"" << TID << "\", ";
    MOutFile << "\"ts\": \"" << TimeStamp << "\"},";
    MOutFile << std::endl;
  }

  void writeRecord(record_t &r) override {
    MOutFile << "{\"name\": \"" << r.name << "\", ";
    MOutFile << "\"cat\": \"" << r.cat << "\", ";
    MOutFile << "\"ph\": \"" << r.phase << "\", ";
    MOutFile << "\"pid\": \"" << r.PID << "\", ";
    MOutFile << "\"tid\": \"" << r.TID << "\", ";
    MOutFile << "\"ts\": \"" << r.TS << "\"},";
    MOutFile << std::endl;
  }

  void finalize() final {
    std::lock_guard<std::mutex> _{MWriteMutex};

    if (!MOutFile.is_open())
      return;

    for (auto &r : MRecords) {
      // Assumes file is open
      writeRecord(r);
    }

    // add an empty element for not ending with '}, ]'
    MOutFile << "{\"name\": \"\", \"cat\": \"\", \"ph\": \"\", \"pid\": \"\", "
                "\"tid\": \"\", \"ts\": \"\"}\n";

    MOutFile << "],\n";
    MOutFile << "\"displayTimeUnit\":\"ns\"\n}\n";
    MOutFile.close();
  }

  ~JSONWriter() { finalize(); }

private:
  std::mutex MWriteMutex;
  std::ofstream MOutFile;
  std::vector<record_t> MRecords;
};
