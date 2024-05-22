//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
#pragma once
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace xpti {
namespace utils {
// We are using C++ 11, hence we cannot use
// std::variant or std::any
using table_row_t = std::map<int, long double>;
using table_t = std::map<int, table_row_t>;
using titles_t = std::vector<std::string>;

class TableModel {
public:
  using row_titles_t = std::map<int, std::string>;

  TableModel(int main_column_width = 14, int precision = 0)
      : m_main_colw(main_column_width), m_precision(precision) {}

  void setHeaders(titles_t &Titles) { MColumnTitles = Titles; }

  table_row_t &addRow(int Row, std::string &RowName) {
    if (MRowTitles.count(Row)) {
      std::cout << "Warning: Row title already specified!\n";
    }
    MRowTitles[Row] = RowName;
    return MTable[Row];
  }

  table_row_t &addRow(int Row, const char *RowName) {
    if (MRowTitles.count(Row)) {
      std::cout << "Warning: Row title already specified!\n";
    }
    MRowTitles[Row] = RowName;
    return MTable[Row];
  }

  table_row_t &operator[](int Row) { return MTable[Row]; }

  void print() {
    std::cout << std::setw(m_main_colw) << " ";
    for (auto &Title : MColumnTitles) {
      std::cout << std::setw(14) << Title; // Column headers
    }
    std::cout << "\n";

    for (auto &Row : MTable) {
      std::cout << std::setw(m_main_colw) << MRowTitles[Row.first];
      for (auto &Data : Row.second) {
        std::cout << std::fixed << std::setw(14)
                  << std::setprecision(m_precision) << Data.second;
      }
      std::cout << "\n";
    }
    std::cout << "\n";
  }

private:
  titles_t MColumnTitles;
  row_titles_t MRowTitles;
  table_t MTable;
  int m_main_colw;
  int m_precision = 0;
};

namespace string {
/// @brief A simple string decoder for use with perf collector
/// @details Since the environment variables could pack multiple ',' separated
/// values, this decode will help parsing the strings and using it in the
/// collector
class simple_string_decoder_t {
public:
  using string_list_t = std::vector<std::string>;

  simple_string_decoder_t(const std::string &separator) {
    if (!separator.empty()) {
      m_separator = separator;
    } else {
      m_separator = ";";
    }
  }

  ~simple_string_decoder_t() {}

  string_list_t &decode(const std::string &input) {
    std::string::size_type prev_pos = 0, pos = 0;
    m_tokens.clear();

    while (pos != std::string::npos) {
      pos = input.find(m_separator, prev_pos);
      if (pos != std::string::npos) {
        // Tokens delimited by 'separator'
        std::string token = input.substr(prev_pos, (pos - prev_pos));
        prev_pos = pos + m_separator.size();
        m_tokens.push_back(token);
      } else {
        // Last token in the string
        std::string token = input.substr(prev_pos);
        m_tokens.push_back(token);
      }
    }

    return m_tokens;
  }

private:
  std::string m_separator;
  string_list_t m_tokens;
};

class list_t {
public:
  using strings_t = std::unordered_map<std::string, bool>;
  list_t() = default;
  ~list_t() = default;

  const char *add(const char *stream) {
    auto res = m_streams.insert(std::make_pair(stream, false));
    if (res.second) {
      return (*res.first).first.c_str();
    }
    return nullptr;
  }

  bool empty() { return (m_streams.size() == 0); }

  bool check(const char *str) {
    if (m_streams.count(str)) {
      m_streams[str] = true;
      return true;
    }
    return false;
  }

  void compact() {
    std::vector<std::string> UnavailableStreams;
    for (auto &e : m_streams) {
      if (!e.second) {
        UnavailableStreams.push_back(e.first);
      }
    }
    for (auto &s : UnavailableStreams) {
      remove(s.c_str());
    }
  }

  void remove(const char *stream) {
    // print();
    auto res = m_streams.erase(stream);
    // std::cout << "Unregistering stream: " << stream << ": Return Value (" <<
    // res
    //           << ") <-> Size: " << m_streams.size() << std::endl;
  }

  void print() {
    for (auto &e : m_streams) {
      std::cout << "Streams: " << e.first << "\n";
    }
  }

private:
  strings_t m_streams;
};

class first_check_map_t {
public:
  using string_count_t = std::unordered_map<std::string, std::atomic<int>>;
  first_check_map_t() = default;
  ~first_check_map_t() = default;

  const char *add(const char *stream) {
    auto res = m_strings.insert(std::make_pair(stream, 0));
    if (res.first != m_strings.end()) {
      return (*res.first).first.c_str();
    }
    return nullptr;
  }

  bool empty() { return (m_strings.size() == 0); }

  bool check(const char *str) {
    auto res = m_strings.find(str);
    if (res == m_strings.end())
      return false;
    else {
      (*res).second++;
      return ((*res).second == 1);
    }
  }

private:
  string_count_t m_strings;
};

} // namespace string
} // namespace utils
} // namespace xpti