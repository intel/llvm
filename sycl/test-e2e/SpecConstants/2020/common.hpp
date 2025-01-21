// Common functionality for SYCL 2020 specialization constants tests:
// definition of custom data types, helper functions, etc.

#pragma once

#include <iostream>
#include <string>

struct custom_type_nested {
  static constexpr char default_c_value = 'a';
  static constexpr float default_f_value = 1.7;

  constexpr custom_type_nested() = default;
  constexpr custom_type_nested(char c, float f) : c(c), f(f) {}

  char c = default_c_value;
  float f = default_f_value;
};

inline bool operator==(const custom_type_nested &lhs,
                       const custom_type_nested &rhs) {
  return lhs.c == rhs.c && lhs.f == rhs.f;
}

inline bool operator!=(const custom_type_nested &lhs,
                       const custom_type_nested &rhs) {
  return !(lhs == rhs);
}

inline std::ostream &operator<<(std::ostream &out,
                                const custom_type_nested &v) {
  return out << "custom_type_nested { .c = " << v.c << ", .f = " << v.f << "}";
}

struct custom_type {
  static constexpr unsigned long long default_ull_value = 42;

  constexpr custom_type() = default;
  constexpr custom_type(char c, float f, unsigned long long ull)
      : n(c, f), ull(ull) {}

  custom_type_nested n;
  unsigned long long ull = default_ull_value;
};

inline bool operator==(const custom_type &lhs, const custom_type &rhs) {
  return lhs.n == rhs.n && lhs.ull == rhs.ull;
}

inline bool operator!=(const custom_type &lhs, const custom_type &rhs) {
  return !(lhs == rhs);
}

inline std::ostream &operator<<(std::ostream &out, const custom_type &v) {
  return out << "custom_type { .n = \n\t" << v.n << ",\n .ull = " << v.ull
             << "}";
}

template <typename T>
bool check_value(const T &ref, const T &got, const std::string &variable_name) {
  if (got != ref) {
    std::cout << "Unexpected value of " << variable_name << ": " << got
              << " (got) vs " << ref << " (expected)" << std::endl;
    return false;
  }

  return true;
}
