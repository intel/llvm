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

} // namespace utils
} // namespace xpti