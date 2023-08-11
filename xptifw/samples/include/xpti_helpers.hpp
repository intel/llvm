//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
#pragma once
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

namespace xpti {
namespace utils {
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
  using strings_t = std::unordered_set<std::string>;
  list_t() = default;
  ~list_t() = default;

  const char *add(const char *stream) {
    auto res = m_streams.insert(stream);
    if (res.second) {
      return (*res.first).c_str();
    }
    return nullptr;
  }

  bool empty() { return (m_streams.size() == 0); }

  bool check(const char *str) {
    auto res = m_streams.find(str);
    if (res == m_streams.end())
      return false;

    else
      return true;
  }

  void remove(const char *stream) {
    auto res = m_streams.erase(stream);
    std::cout << "Unregistering stream: " << stream << ": Return Value (" << res
              << ") <-> Size: " << m_streams.size() << std::endl;
  }

private:
  strings_t m_streams;
};

} // namespace string
} // namespace utils
} // namespace xpti