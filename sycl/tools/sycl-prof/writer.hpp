//==----------------- writer.hpp -------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <filesystem>
#include <fstream>
#include <mutex>
#include <string_view>

class Writer {
public:
  virtual void init() = 0;
  virtual void finalize() = 0;
  virtual void writeBegin(std::string_view Name, std::string_view Category,
                          unsigned long PID, unsigned long TID,
                          unsigned long TimeStamp) = 0;
  virtual void writeEnd(std::string_view Name, std::string_view Category,
                        unsigned long PID, unsigned long TID,
                        unsigned long TimeStamp) = 0;
  virtual ~Writer() = default;
};

class JSONWriter : public Writer {
public:
  explicit JSONWriter(std::filesystem::path OutPath) : MOutFile(OutPath) {}

  void init() final {
    std::lock_guard _{MWriteMutex};

    MOutFile << "{\n";
    MOutFile << "  \"traceEvents\": [\n";
  }

  void writeBegin(std::string_view Name, std::string_view Category,
                  unsigned long PID, unsigned long TID,
                  unsigned long TimeStamp) override {
    std::lock_guard _{MWriteMutex};

    if (!MOutFile.is_open())
      return;

    MOutFile << "{\"name\": \"" << Name << "\", ";
    MOutFile << "\"cat\": \"" << Category << "\", ";
    MOutFile << "\"ph\": \"B\", ";
    MOutFile << "\"pid\": \"" << PID << "\", ";
    MOutFile << "\"tid\": \"" << TID << "\", ";
    MOutFile << "\"ts\": \"" << TimeStamp << "\"},\n";
  }

  void writeEnd(std::string_view Name, std::string_view Category,
                unsigned long PID, unsigned long TID,
                unsigned long TimeStamp) override {
    std::lock_guard _{MWriteMutex};

    if (!MOutFile.is_open())
      return;

    MOutFile << "{\"name\": \"" << Name << "\", ";
    MOutFile << "\"cat\": \"" << Category << "\", ";
    MOutFile << "\"ph\": \"E\", ";
    MOutFile << "\"pid\": \"" << PID << "\", ";
    MOutFile << "\"tid\": \"" << TID << "\", ";
    MOutFile << "\"ts\": \"" << TimeStamp << "\"},\n";
  }

  void finalize() final {
    std::lock_guard _{MWriteMutex};

    if (!MOutFile.is_open())
      return;

    MOutFile << "],\n";
    MOutFile << "\"displayTimeUnit\":\"ns\"\n}\n";
    MOutFile.close();
  }

  ~JSONWriter() { finalize(); }

private:
  std::mutex MWriteMutex;
  std::ofstream MOutFile;
};
