//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
#pragma once
#include "xpti_trace_framework.hpp"

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

// We are using C++ 11, hence we cannot use
// std::variant or std::any
typedef std::map<int, long double> table_row_t;
typedef std::map<int, table_row_t> table_t;
typedef std::vector<std::string> titles_t;
class scoped_timer {
public:
  typedef std::chrono::time_point<std::chrono::high_resolution_clock>
      time_unit_t;
  scoped_timer(uint64_t &ns, double &ratio, size_t count = 1)
      : m_duration{ns}, m_average{ratio}, m_instances{count} {
    m_before = std::chrono::high_resolution_clock::now();
  }

  ~scoped_timer() {
    m_after = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
        m_after - m_before);
    m_duration = duration.count();
    m_average = (double)m_duration / m_instances;
  }

private:
  uint64_t &m_duration;
  double &m_average;
  size_t m_instances;
  time_unit_t m_before, m_after;
};

class cl_option {
public:
  cl_option()
      : m_required(false), m_type(OptionType::String),
        m_help("No help available.") {}
  ~cl_option() {}

  cl_option &set_required(bool yesOrNo) {
    m_required = yesOrNo;
    return *this;
  }
  cl_option &set_type(OptionType type) {
    m_type = type;
    return *this;
  }
  cl_option &set_help(std::string help) {
    m_help = help;
    return *this;
  }
  cl_option &set_abbreviation(std::string abbr) {
    m_abbrev = abbr;
    return *this;
  }

  std::string &abbreviation() { return m_abbrev; }
  std::string &help() { return m_help; }
  OptionType type() { return m_type; }
  bool required() { return m_required; }

private:
  bool m_required;
  OptionType m_type;
  std::string m_help;
  std::string m_abbrev;
};

class cl_parser {
public:
  typedef std::unordered_map<std::string, cl_option> cl_options_t;
  typedef std::unordered_map<std::string, std::string> key_value_t;

  cl_parser() {
    m_reserved_key = "--help";
    m_reserved_key_abbr = "-h";
  }

  ~cl_parser() {}

  void parse(int argc, char **argv) {
    m_cl_options.resize(argc);
    // Go through the command-line options list and build an internal
    m_app_name = argv[0];
    for (int i = 1; i < argc; ++i) {
      m_cl_options[i - 1] = argv[i];
    }

    build_abbreviation_table();

    if (!check_options()) {
      print_help();
      exit(-1);
    }
  }

  cl_option &add_option(std::string key) {
    if (key == m_reserved_key) {
      std::cout << "Option[" << key
                << "] is a reserved option. Ignoring the add_option() call!\n";
      // throw an exception here;
    }
    if (m_option_help_lut.count(key)) {
      std::cout << "Option " << key << " has already been registered!\n";
      return m_option_help_lut[key];
    }

    return m_option_help_lut[key];
  }

  std::string &query(const char *key) {
    if (m_option_help_lut.count(key)) {
      return m_value_lut[key];
    } else if (m_abbreviated_option_lut.count(key)) {
      std::string full_key = m_abbreviated_option_lut[key];
      if (m_value_lut.count(full_key)) {
        return m_value_lut[full_key];
      }
      return m_empty_string;
    }
  }

private:
  void build_abbreviation_table() {
    for (auto &o : m_option_help_lut) {
      std::string &abbr = o.second.abbreviation();
      if (!abbr.empty()) {
        m_abbreviated_option_lut[abbr] = o.first;
      }
    }
  }

  void print_help() {
    std::cout << "Usage:- \n";
    std::cout << "      " << m_app_name << " ";
    // Print all required options first
    for (auto &op : m_option_help_lut) {
      if (op.second.required()) {
        std::cout << op.first << " ";
        switch (op.second.type()) {
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
          std::cout << "<val1,val2,begin:end:step> ";
          break;
        }
      }
    }
    // Print the optional flags next.
    for (auto &op : m_option_help_lut) {
      if (!op.second.required()) {
        std::cout << "[" << op.first << " ";
        switch (op.second.type()) {
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
          std::cout << "<val1,val2,begin:end:step>] ";
          break;
        }
      }
    }
    std::cout << "\n      Options supported:\n";
    // Print help for all of the options
    for (auto &op : m_option_help_lut) {
      std::stringstream help(op.second.help());
      std::string help_line;
      bool first = true;

      while (std::getline(help, help_line, '\n')) {
        if (first) {
          std::string options = op.first + ", " + op.second.abbreviation();
          first = false;
          std::cout << "      " << std::left << std::setw(20) << options << " "
                    << help_line << "\n";
        } else {
          std::cout << "      " << std::left << std::setw(20) << " "
                    << " " << help_line << "\n";
        }
      }
    }
  }

  bool check_options() {
    bool pass = true;
    std::string prev_key;
    for (auto &op : m_cl_options) {
      std::size_t pos = op.find_first_of("-");
      if (std::string::npos != pos) {
        //  We have an option provided; let's check to see if it is verbose or
        //  abbreviated
        pos = op.find_first_of("-", pos + 1);
        if (std::string::npos != pos) {
          // We have a verbose option
          if (op == m_reserved_key) {
            print_help();
            exit(-1);
          } else if (m_option_help_lut.count(op) == 0) {
            std::cout << "Unknown option[" << op << "]!\n";
            pass = false;
          }
          m_value_lut[op] = "true";
          prev_key = op;
        } else {
          // We have an abbreviated option
          if (op == m_reserved_key_abbr) {
            print_help();
            exit(-1);
          } else if (m_abbreviated_option_lut.count(op) == 0) {
            std::cout << "Unknown option[" << op << "] detected.\n";
            pass = false;
          }
          prev_key = m_abbreviated_option_lut[op];
          m_value_lut[prev_key] = "true";
        }
      } else {
        // No idea why stringstream will decode the last \n as a "" string; this
        // handles that case
        if (prev_key.empty() && op.empty())
          break;
        // We have an option value
        if (prev_key.empty()) {
          std::cout << "Value[" << op
                    << "] provided without specifying an option\n";
          pass = false;
        } else {
          m_value_lut[prev_key] = op;
          prev_key = m_empty_string;
        }
      }
    }

    for (auto &op : m_option_help_lut) {
      // Check to see if an option is required; If so, check to see if there's a
      // value associated with it.
      if (op.second.required()) {
        if (!m_value_lut.count(op.first)) {
          std::cout << "Option[" << op.first
                    << "] is required and not provided.\n";
          pass = false;
        }
      }
    }

    return pass;
  }

  std::vector<std::string> m_cl_options;
  cl_options_t m_option_help_lut;
  key_value_t m_abbreviated_option_lut;
  key_value_t m_value_lut;
  std::string m_empty_string;
  std::string m_reserved_key;
  std::string m_reserved_key_abbr;
  std::string m_app_name;
};

class table_model {
public:
  typedef std::map<int, std::string> row_titles_t;
  table_model() {}

  void set_headers(titles_t &titles) { m_column_titles = titles; }

  table_row_t &add_row(int row, std::string &row_name) {
    if (m_row_titles.count(row)) {
      std::cout << "Warning: Row title already specified!\n";
    }
    m_row_titles[row] = row_name;
    return m_table[row];
  }
  table_row_t &add_row(int row, const char *row_name) {
    if (m_row_titles.count(row)) {
      std::cout << "Warning: Row title already specified!\n";
    }
    m_row_titles[row] = row_name;
    return m_table[row];
  }

  table_row_t &operator[](int row) { return m_table[row]; }

  void print() {
    std::cout << std::setw(14) << " ";
    for (auto &title : m_column_titles) {
      std::cout << std::setw(14) << title; // Column headers
    }
    std::cout << "\n";

    for (auto &row : m_table) {
      std::cout << std::setw(14) << m_row_titles[row.first];
      int prev_col = 0;
      for (auto &data : row.second) {
        std::cout << std::fixed << std::setw(14) << std::setprecision(0)
                  << data.second;
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

class range_decoder {
public:
  range_decoder(std::string &range_str) : m_range(range_str) {
    // Split by commas first followed by : for begin,end, step
    std::stringstream elements(range_str);
    std::string element;
    while (std::getline(elements, element, ',')) {
      if (element.find_first_of("-:") == std::string::npos) {
        m_elements.insert(std::stol(element));
      } else {
        std::stringstream r(element);
        std::vector<std::string> range_tokens;
        std::string e, b;
        // Now split by :
        while (std::getline(r, e, ':')) {
          range_tokens.push_back(e);
        }
        // range_tokens should have three entries; Second entry is the step
        std::cout << range_tokens[0] << ";" << range_tokens[1] << std::endl;
        long step = std::stol(range_tokens[2]);
        for (long i = std::stol(range_tokens[0]);
             i <= std::stol(range_tokens[1]); i += step) {
          m_elements.insert(i);
        }
      }
    }
  }

  std::set<long> &decode() { return m_elements; }

private:
  std::string m_range;
  std::set<long> m_elements;
};
} // namespace utils

namespace semantic {
class test_correctness {
public:
  enum class SemanticTests {
    StringTableTest = 1,
    TracePointTest,
    NotificationTest
  };

  test_correctness(test::utils::cl_parser &parser) : m_parser(parser) {
    xptiInitialize("xpti", 20, 0, "xptiTests");
  }

  void run() {
    auto &v = m_parser.query("--type");
    if (v != "semantic")
      return;

    test::utils::range_decoder td(m_parser.query("--num-threads"));
    m_threads = td.decode();
    test::utils::range_decoder rd(m_parser.query("--test-id"));
    m_tests = rd.decode();

    run_tests();
  }

  void run_tests() {
    for (auto test : m_tests) {
      switch ((SemanticTests)test) {
      case SemanticTests::StringTableTest:
        run_string_table_tests();
        break;
      case SemanticTests::TracePointTest:
        run_tracepoint_tests();
        break;
      case SemanticTests::NotificationTest:
        run_notification_tests();
        break;
      default:
        std::cout << "Unknown test type [" << test << "]: use 1,2,3 or 1:3:1\n";
        break;
      }
    }
    m_table.print();
  }

private:
  void run_string_table_tests();
  void run_string_table_test_threads(int run_no, int nt,
                                     test::utils::table_model &t);
  void run_tracepoint_tests();
  void run_tracepoint_test_threads(int run_no, int nt,
                                   test::utils::table_model &t);
  void run_notification_tests();
  void run_notification_test_threads(int run_no, int nt,
                                     test::utils::table_model &t);

  test::utils::cl_parser &m_parser;
  test::utils::table_model m_table;
  std::set<long> m_threads, m_tests;
  long m_tracepoints;
  const char *m_source = "foo.cpp";
  uint64_t m_instance_id = 0;
};
} // namespace semantic

namespace performance {
constexpr int MaxTracepoints = 100000;
constexpr int MinTracepoints = 10;
class test_performance {
public:
  struct record {
    std::string fn;
    uint64_t lookup;
  };
  enum class PerformanceTests { DataStructureTest = 1, InstrumentationTest };

  test_performance(test::utils::cl_parser &parser) : m_parser(parser) {
    xptiInitialize("xpti", 20, 0, "xptiTests");
  }

  std::string make_random_string(uint8_t length, std::mt19937_64 &gen) {
    if (length > 25) {
      length = 25;
    }
    // A=65, a=97
    std::string s(length, '\0');
    for (int i = 0; i < length; ++i) {
      int ascii = m_case(gen);
      int value = m_char(gen);
      s[i] = (ascii ? value + 97 : value + 65);
    }
    return s;
  }

  void run() {
    auto &v = m_parser.query("--type");
    if (v != "performance")
      return;

    test::utils::range_decoder td(m_parser.query("--num-threads"));
    m_threads = td.decode();
    m_tracepoints = std::stol(m_parser.query("--trace-points"));
    if (m_tracepoints > MaxTracepoints) {
      std::cout << "Reducing trace points to " << MaxTracepoints << "!\n";
      m_tracepoints = MaxTracepoints;
    }
    if (m_tracepoints < 0) {
      std::cout << "Setting trace points to " << MinTracepoints << "!\n";
      m_tracepoints = MinTracepoints;
    }

    test::utils::range_decoder rd(m_parser.query("--test-id"));
    m_tests = rd.decode();

    std::string dist = m_parser.query("--tp-frequency");
    if (dist.empty()) {
      // By default, we assume that for every trace point that is created, we
      // will visit it NINE more times.
      m_tp_instances = m_tracepoints * 10;
    } else {
      float value = std::stof(dist);
      if (value > 100) {
        std::cout << "Trace point creation frequency limited to 100%!\n";
        value = 100;
      }
      if (value < 0) {
        std::cout << "Trace point creation frequency set to 1%!\n";
        value = 1;
      }
      // If not, we compute the number of trace point instances based on the
      // trace point frequency value; If the frequency is 10%, then every 10th
      // trace point create will be creating a new trace point. If it is 2%,
      // then every 50th trace point will create call will result in a new
      // trace point.
      m_tp_instances = (long)((1.0 / (std::stof(dist) / 100)) * m_tracepoints);
    }
    // Check to see if overheads to model are set; if not assume 1.0%
    dist = m_parser.query("--overhead");
    if (!dist.empty()) {
      m_overhead = std::stof(dist);
      if (m_overhead < 0.1) {
        std::cout << "Overheads to be modeled clamped to range - 0.1%!\n";
        m_overhead = 0.1;
      } else if (m_overhead > 15) {
        std::cout << "Overheads to be modeled clamped to range - 15%!\n";
        m_overhead = 15;
      }
    }

    // If the number of trace points(TP) required to run tests on is 1000, then
    // we will run our string table tests on the number of TPs we compute. For a
    // TP frequency of 10%, we will have TP instances be 1000x10
    m_st_entries = m_tp_instances;
    // Mersenne twister RNG engine that is uniform distribution
    std::random_device q_rd;
    std::mt19937_64 gen(q_rd());
    // Generate the pseudo-random numbers for trace points and string table
    // random lookup
    m_tp = std::uniform_int_distribution<int32_t>(0, m_tracepoints - 1);
    m_st = std::uniform_int_distribution<int32_t>(0, m_st_entries - 1);
    m_char = std::uniform_int_distribution<int32_t>(0, 25);
    m_case = std::uniform_int_distribution<int32_t>(0, 1);

    m_rnd_st.resize(m_st_entries);
    m_rnd_tp.resize(m_st_entries);
    for (int i = 0; i < m_st_entries; ++i) {
      m_rnd_st[i] = m_st(gen);
    }
    for (int i = 0; i < m_st_entries; ++i) {
      m_rnd_tp[i] = m_tp(gen);
    }
    // Generate the strings we will be registering with the string table and
    // also the random lookup table for trace points
    for (int i = 0; i < m_tp_instances; ++i) {
      record r;
      r.lookup = m_rnd_tp[i]; // 0-999999999
      std::string str = make_random_string(5, gen);
      r.fn = str + std::to_string(r.lookup);
      m_records.push_back(r);
      str = make_random_string(8, gen) + std::to_string(i);
      m_functions.push_back(str);
      str = make_random_string(8, gen) + std::to_string(i);
      m_functions2.push_back(str);
    }
    // Done with the setup; now run the tests
    run_tests();
  }

  void run_tests() {
    for (auto test : m_tests) {
      switch ((PerformanceTests)test) {
      case PerformanceTests::DataStructureTest:
        run_data_structure_tests();
        break;
      case PerformanceTests::InstrumentationTest:
        run_instrumentation_tests();
        break;
      default:
        std::cout << "Unknown test type [" << test << "]: use 1,2 or 1:2:1\n";
        break;
      }
    }
    m_table.print();
  }

private:
  void run_data_structure_tests();
  void run_data_structure_tests_threads(int run_no, int nt,
                                        test::utils::table_model &t);
  void run_instrumentation_tests();
  void run_instrumentation_tests_threads(int run_no, int nt,
                                         test::utils::table_model &t);

  test::utils::cl_parser &m_parser;
  test::utils::table_model m_table;
  std::set<long> m_threads, m_tests;
  long m_tracepoints;
  long m_tp_instances;
  long m_st_entries;
  const char *m_source = "foo.cpp";
  uint64_t m_instance_id = 0;
  std::uniform_int_distribution<int32_t> m_tp, m_st, m_char, m_case;
  std::vector<int> m_rnd_tp, m_rnd_st;
  std::vector<record> m_records;
  std::vector<std::string> m_functions, m_functions2;
  double m_overhead = 1.0;
};
} // namespace performance
} // namespace test
