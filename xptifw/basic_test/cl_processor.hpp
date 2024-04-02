//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
#pragma once

#include "xpti/xpti_trace_framework.hpp"
#include "xpti_helpers.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <stdint.h>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace test {
namespace utils {
enum class OptionType { Boolean, Integer, Float, String, Range };

class ScopedTimer {
public:
  using time_unit_t =
      std::chrono::time_point<std::chrono::high_resolution_clock>;
  ScopedTimer(uint64_t &ns, double &ratio, size_t count = 1)
      : MDuration{ns}, MAverage{ratio}, MInstances{count} {
    MBefore = std::chrono::high_resolution_clock::now();
  }

  ~ScopedTimer() {
    MAfter = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(MAfter - MBefore);
    MDuration = duration.count();
    MAverage = (double)MDuration / MInstances;
  }

private:
  uint64_t &MDuration;
  double &MAverage;
  size_t MInstances;
  time_unit_t MBefore, MAfter;
};

class CommandLineOption {
public:
  CommandLineOption()
      : MRequired(false), MType(OptionType::String),
        MHelp("No help available.") {}

  CommandLineOption &setRequired(bool yesOrNo) {
    MRequired = yesOrNo;
    return *this;
  }
  CommandLineOption &setType(OptionType type) {
    MType = type;
    return *this;
  }
  CommandLineOption &setHelp(std::string help) {
    MHelp = help;
    return *this;
  }
  CommandLineOption &setAbbreviation(std::string abbr) {
    MAbbrev = abbr;
    return *this;
  }

  std::string &abbreviation() { return MAbbrev; }
  std::string &help() { return MHelp; }
  OptionType type() { return MType; }
  bool required() { return MRequired; }

private:
  bool MRequired;
  OptionType MType;
  std::string MHelp;
  std::string MAbbrev;
};

class CommandLineParser {
public:
  using CommandLineOptions_t =
      std::unordered_map<std::string, CommandLineOption>;
  using key_value_t = std::unordered_map<std::string, std::string>;

  CommandLineParser() {
    MReservedKey = "--help";
    MReservedKeyAbbr = "-h";
  }

  void parse(int argc, char **argv) {
    MCommandLineOptions.resize(argc);
    // Go through the command-line options list and build an internal
    MAppName = argv[0];
    for (int i = 1; i < argc; ++i) {
      MCommandLineOptions[i - 1] = argv[i];
    }

    buildAbbreviationTable();

    if (!checkOptions()) {
      printHelp();
      exit(-1);
    }
  }

  CommandLineOption &addOption(std::string Key) {
    if (Key == MReservedKey) {
      std::cout << "Option[" << Key
                << "] is a reserved option. Ignoring the addOption() call!\n";
      // throw an exception here;
    }
    if (MOptionHelpLUT.count(Key)) {
      std::cout << "Option " << Key << " has already been registered!\n";
      return MOptionHelpLUT[Key];
    }

    return MOptionHelpLUT[Key];
  }

  std::string &query(const char *Key) {
    if (MOptionHelpLUT.count(Key)) {
      return MValueLUT[Key];
    } else if (MAbbreviatedOptionLUT.count(Key)) {
      std::string FullKey = MAbbreviatedOptionLUT[Key];
      if (MValueLUT.count(FullKey)) {
        return MValueLUT[FullKey];
      }
      return MEmptyString;
    }
    return MEmptyString;
  }

private:
  void buildAbbreviationTable() {
    for (auto &Option : MOptionHelpLUT) {
      std::string &abbr = Option.second.abbreviation();
      if (!abbr.empty()) {
        MAbbreviatedOptionLUT[abbr] = Option.first;
      }
    }
  }

  void printHelp() {
    std::cout << "Usage:- \n";
    std::cout << "      " << MAppName << " ";
    // Print all required options first
    for (auto &Option : MOptionHelpLUT) {
      if (Option.second.required()) {
        std::cout << Option.first << " ";
        switch (Option.second.type()) {
        case OptionType::Integer:
          std::cout << "<integer> ";
          break;
        case OptionType::Float:
          std::cout << "<float> ";
          break;
        case OptionType::Boolean:
          std::cout << " ";
          break;
        case OptionType::String:
          std::cout << "<string> ";
          break;
        case OptionType::Range:
          std::cout << "<val1,val2,begin:end:Step> ";
          break;
        }
      }
    }
    // Print the optional flags next.
    for (auto &Option : MOptionHelpLUT) {
      if (!Option.second.required()) {
        std::cout << "[" << Option.first << " ";
        switch (Option.second.type()) {
        case OptionType::Integer:
          std::cout << "<integer>] ";
          break;
        case OptionType::Float:
          std::cout << "<float>] ";
          break;
        case OptionType::Boolean:
          std::cout << "] ";
          break;
          break;
        case OptionType::String:
          std::cout << "<string>] ";
          break;
        case OptionType::Range:
          std::cout << "<val1,val2,begin:end:Step>] ";
          break;
        }
      }
    }
    std::cout << "\n      Options supported:\n";
    // Print help for all of the options
    for (auto &Option : MOptionHelpLUT) {
      std::stringstream Help(Option.second.help());
      std::string HelpLine;
      bool FirstTime = true;

      while (std::getline(Help, HelpLine, '\n')) {
        if (FirstTime) {
          std::string options =
              Option.first + ", " + Option.second.abbreviation();
          FirstTime = false;
          std::cout << "      " << std::left << std::setw(20) << options << " "
                    << HelpLine << "\n";
        } else {
          std::cout << "      " << std::left << std::setw(20) << " "
                    << " " << HelpLine << "\n";
        }
      }
    }
  }

  bool checkOptions() {
    bool Pass = true;
    std::string PrevKey;
    for (auto &Option : MCommandLineOptions) {
      std::size_t Pos = Option.find_first_of("-");
      if (std::string::npos != Pos) {
        //  We have an option provided; let's check to see if it is verbose or
        //  abbreviated
        Pos = Option.find_first_of("-", Pos + 1);
        if (std::string::npos != Pos) {
          // We have a verbose option
          if (Option == MReservedKey) {
            printHelp();
            exit(-1);
          } else if (MOptionHelpLUT.count(Option) == 0) {
            std::cout << "Unknown option[" << Option << "]!\n";
            Pass = false;
          }
          MValueLUT[Option] = "true";
          PrevKey = Option;
        } else {
          // We have an abbreviated option
          if (Option == MReservedKeyAbbr) {
            printHelp();
            exit(-1);
          } else if (MAbbreviatedOptionLUT.count(Option) == 0) {
            std::cout << "Unknown option[" << Option << "] detected.\n";
            Pass = false;
          }
          PrevKey = MAbbreviatedOptionLUT[Option];
          MValueLUT[PrevKey] = "true";
        }
      } else {
        // No idea why stringstream will decode the last \n as a "" string; this
        // handles that case
        if (PrevKey.empty() && Option.empty())
          break;
        // We have an option value
        if (PrevKey.empty()) {
          std::cout << "Value[" << Option
                    << "] provided without specifying an option\n";
          Pass = false;
        } else {
          MValueLUT[PrevKey] = Option;
          PrevKey = MEmptyString;
        }
      }
    }

    for (auto &Option : MOptionHelpLUT) {
      // Check to see if an option is required; If so, check to see if there's a
      // value associated with it.
      if (Option.second.required()) {
        if (!MValueLUT.count(Option.first)) {
          std::cout << "Option[" << Option.first
                    << "] is required and not provided.\n";
          Pass = false;
        }
      }
    }

    return Pass;
  }

  std::vector<std::string> MCommandLineOptions;
  CommandLineOptions_t MOptionHelpLUT;
  key_value_t MAbbreviatedOptionLUT;
  key_value_t MValueLUT;
  std::string MEmptyString;
  std::string MReservedKey;
  std::string MReservedKeyAbbr;
  std::string MAppName;
};

class RangeDecoder {
public:
  RangeDecoder(std::string &RangeStr) : MRange(RangeStr) {
    // Split by commas first followed by : for begin,end, Step
    std::stringstream Elements(RangeStr);
    std::string Element;
    while (std::getline(Elements, Element, ',')) {
      if (Element.find_first_of("-:") == std::string::npos) {
        MElements.insert(std::stol(Element));
      } else {
        std::stringstream R(Element);
        std::vector<std::string> RangeTokens;
        std::string SubStr;
        // Now split by :
        while (std::getline(R, SubStr, ':')) {
          RangeTokens.push_back(SubStr);
        }
        // RangeTokens should have three entries; Second entry is the Step
        std::cout << RangeTokens[0] << ";" << RangeTokens[1] << std::endl;
        long Step = std::stol(RangeTokens[2]);
        for (long i = std::stol(RangeTokens[0]); i <= std::stol(RangeTokens[1]);
             i += Step) {
          MElements.insert(i);
        }
      }
    }
  }

  std::set<long> &decode() { return MElements; }

private:
  std::string MRange;
  std::set<long> MElements;
};
} // namespace utils

namespace semantic {
class TestCorrectness {
public:
  enum class SemanticTests {
    StringTableTest = 1,
    TracePointTest,
    NotificationTest
  };

  TestCorrectness(test::utils::CommandLineParser &Parser) : MParser(Parser) {
    xptiInitialize("xpti", 20, 0, "xptiTests");
  }

  void run() {
    auto &V = MParser.query("--type");
    if (V != "semantic")
      return;

    test::utils::RangeDecoder td(MParser.query("--num-threads"));
    MThreads = td.decode();
    test::utils::RangeDecoder rd(MParser.query("--test-id"));
    MTests = rd.decode();

    runTests();
  }

  void runTests() {
    for (auto Test : MTests) {
      switch ((SemanticTests)Test) {
      case SemanticTests::StringTableTest:
        runStringTableTests();
        break;
      case SemanticTests::TracePointTest:
        runTracepointTests();
        break;
      case SemanticTests::NotificationTest:
        runNotificationTests();
        break;
      default:
        std::cout << "Unknown test type [" << Test << "]: use 1,2,3 or 1:3:1\n";
        break;
      }
    }
    MTable.print();
  }

private:
  void runStringTableTests();
  void runStringTableTestThreads(int RunNo, int NThreads,
                                 xpti::utils::TableModel &Table);
  void runTracepointTests();
  void runTracepointTestThreads(int RunNo, int nt,
                                xpti::utils::TableModel &Table);
  void runNotificationTests();
  void runNotificationTestThreads(int RunNo, int NThreads,
                                  xpti::utils::TableModel &Table);

  test::utils::CommandLineParser &MParser;
  xpti::utils::TableModel MTable;
  std::set<long> MThreads, MTests;
  long MTracepoints;
  const char *MSource = "foo.cpp";
  uint64_t MInstanceID = 0;
};
} // namespace semantic

namespace performance {
constexpr int MaxTracepoints = 100000;
constexpr int MinTracepoints = 10;
class TestPerformance {
public:
  struct record {
    std::string fn;
    uint64_t lookup;
  };
  enum class PerformanceTests { DataStructureTest = 1, InstrumentationTest };

  TestPerformance(test::utils::CommandLineParser &Parser) : MParser(Parser) {
    xptiInitialize("xpti", 20, 0, "xptiTests");
  }

  std::string makeRandomString(uint8_t Length, std::mt19937_64 &Gen) {
    if (Length > 25) {
      Length = 25;
    }
    // A=65, a=97
    std::string s(Length, '\0');
    for (int i = 0; i < Length; ++i) {
      int ascii = MCaseU(Gen);
      int value = MCharU(Gen);
      s[i] = (ascii ? value + 97 : value + 65);
    }
    return s;
  }

  void run() {
    auto &V = MParser.query("--type");
    if (V != "performance")
      return;

    test::utils::RangeDecoder Td(MParser.query("--num-threads"));
    MThreads = Td.decode();
    MTracepoints = std::stol(MParser.query("--trace-points"));
    if (MTracepoints > MaxTracepoints) {
      std::cout << "Reducing trace points to " << MaxTracepoints << "!\n";
      MTracepoints = MaxTracepoints;
    }
    if (MTracepoints < 0) {
      std::cout << "Setting trace points to " << MinTracepoints << "!\n";
      MTracepoints = MinTracepoints;
    }

    test::utils::RangeDecoder Rd(MParser.query("--test-id"));
    MTests = Rd.decode();

    std::string Dist = MParser.query("--tp-frequency");
    if (Dist.empty()) {
      // By default, we assume that for every trace point that is created, we
      // will visit it NINE more times.
      MTracepointInstances = MTracepoints * 10;
    } else {
      float Value = std::stof(Dist);
      if (Value > 100) {
        std::cout << "Trace point creation frequency limited to 100%!\n";
        Value = 100;
      }
      if (Value < 0) {
        std::cout << "Trace point creation frequency set to 1%!\n";
        Value = 1;
      }
      // If not, we compute the number of trace point instances based on the
      // trace point frequency value; If the frequency is 10%, then every 10th
      // trace point create will be creating a new trace point. If it is 2%,
      // then every 50th trace point will create call will result in a new
      // trace point.
      MTracepointInstances =
          (long)((1.0 / (std::stof(Dist) / 100)) * MTracepoints);
    }
    // Check to see if overheads to model are set; if not assume 1.0%
    Dist = MParser.query("--overhead");
    if (!Dist.empty()) {
      MOverhead = std::stof(Dist);
      if (MOverhead < 0.1) {
        std::cout << "Overheads to be modeled clamped to range - 0.1%!\n";
        MOverhead = 0.1;
      } else if (MOverhead > 15) {
        std::cout << "Overheads to be modeled clamped to range - 15%!\n";
        MOverhead = 15;
      }
    }

    // If the number of trace points(TP) required to run tests on is 1000, then
    // we will run our string table tests on the number of TPs we compute. For a
    // TP frequency of 10%, we will have TP instances be 1000x10
    MStringTableEntries = MTracepointInstances;
    // Mersenne twister RNG engine that is uniform distribution
    std::random_device QRd;
    std::mt19937_64 Gen(QRd());
    // Generate the pseudo-random numbers for trace points and string table
    // random lookup
    MTracepointU = std::uniform_int_distribution<int32_t>(0, MTracepoints - 1);
    MStringTableU =
        std::uniform_int_distribution<int32_t>(0, MStringTableEntries - 1);
    MCharU = std::uniform_int_distribution<int32_t>(0, 25);
    MCaseU = std::uniform_int_distribution<int32_t>(0, 1);

    MRndmSTIndex.resize(MStringTableEntries);
    MRndmTPIndex.resize(MStringTableEntries);
    for (int i = 0; i < MStringTableEntries; ++i) {
      MRndmSTIndex[i] = MStringTableU(Gen);
    }
    for (int i = 0; i < MStringTableEntries; ++i) {
      MRndmTPIndex[i] = MTracepointU(Gen);
    }
    // Generate the strings we will be registering with the string table and
    // also the random lookup table for trace points
    for (int i = 0; i < MTracepointInstances; ++i) {
      record Rec;
      Rec.lookup = MRndmTPIndex[i]; // 0-999999999
      std::string Str = makeRandomString(5, Gen);
      Rec.fn = Str + std::to_string(Rec.lookup);
      MRecords.push_back(Rec);
      Str = makeRandomString(8, Gen) + std::to_string(i);
      MFunctions.push_back(Str);
      Str = makeRandomString(8, Gen) + std::to_string(i);
      MFunctions2.push_back(Str);
    }
    // Done with the setup; now run the tests
    runTests();
  }

  void runTests() {
    for (auto Test : MTests) {
      switch ((PerformanceTests)Test) {
      case PerformanceTests::DataStructureTest:
        runDataStructureTests();
        break;
      case PerformanceTests::InstrumentationTest:
        runInstrumentationTests();
        break;
      default:
        std::cout << "Unknown test type [" << Test << "]: use 1,2 or 1:2:1\n";
        break;
      }
    }
    MTable.print();
  }

private:
  void runDataStructureTests();
  void runDataStructureTestsThreads(int RunNo, int NThreads,
                                    xpti::utils::TableModel &Table);
  void runInstrumentationTests();
  void runInstrumentationTestsThreads(int RunNo, int NThreads,
                                      xpti::utils::TableModel &Table);

  test::utils::CommandLineParser &MParser;
  xpti::utils::TableModel MTable;
  std::set<long> MThreads, MTests;
  long MTracepoints;
  long MTracepointInstances;
  long MStringTableEntries;
  const char *MSource = "foo.cpp";
  uint64_t MInstanceID = 0;
  std::uniform_int_distribution<int32_t> MTracepointU, MStringTableU, MCharU,
      MCaseU;
  std::vector<int> MRndmTPIndex, MRndmSTIndex;
  std::vector<record> MRecords;
  std::vector<std::string> MFunctions, MFunctions2;
  double MOverhead = 1.0;
};
} // namespace performance
} // namespace test
