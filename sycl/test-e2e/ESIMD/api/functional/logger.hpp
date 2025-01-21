//===-- logger.hpp - Define functions for print messages to console. ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declarations of the functions that provides easier
/// way to printing messages to the console.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "type_traits.hpp"
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>

namespace esimd_test::api::functional {

// Interface for any test case description class to use for logs within generic
// assertions
//
// Should provide generic test case description with no specific error details,
// for example: "simd<int, 8> with post-increment operator"
struct ITestDescription {
  virtual ~ITestDescription() = default;
  virtual std::string to_string() const = 0;
};

namespace log {
namespace detail {

template <bool Value> struct support_flag {
  static constexpr bool is_supported() { return Value; }
};

// Provides type for floating-point hex representation for logging purposes
//
// In case some fixed-width type is not supported a fallback to the primary
// template is used, with no template specialization and actual type provided
template <typename T> struct fp_hex_representation : support_flag<false> {};
#ifdef UINT16_MAX
template <> struct fp_hex_representation<sycl::half> : support_flag<true> {
  using type = std::uint16_t;
};
#endif
#ifdef UINT32_MAX
template <> struct fp_hex_representation<float> : support_flag<true> {
  using type = std::uint32_t;
};
#endif
#ifdef UINT64_MAX
template <> struct fp_hex_representation<double> : support_flag<true> {
  using type = std::uint64_t;
};
#endif

} // namespace detail

// Provides a stringification helper for safe specialization
// Benefits:
//  - makes partial specialization possible
//  - avoids unexpected behaviour for function specializations and overloads
template <typename T> struct StringMaker {
  static std::string stringify(T val) {
    if constexpr (std::is_convertible_v<T, std::string>) {
      return val;
    } else if constexpr (type_traits::is_sycl_floating_point_v<T>) {
      // Define the output precision based on the type precision itself
      // For example, state 9 decimal digits for 32-bit floating point
      const auto significand_decimal_digits = sizeof(T) * 2 + 1;

      std::ostringstream out;
      out.precision(significand_decimal_digits);
      out << val << " [";
      if constexpr (detail::fp_hex_representation<T>::is_supported()) {
        // Print out hex representation using type-punning
        using hex_type = typename detail::fp_hex_representation<T>::type;
        const auto &representation = reinterpret_cast<const hex_type &>(val);
        out << "0x" << std::hex << representation;
      } else {
        // Make support gap explicit to address if required
        out << " - ";
      }
      out << "]";
      return out.str();
    } else if constexpr (std::is_enum_v<T>) {
      // Any enumeration requires explicit cast to the underlying type
      // Please note that StringMaker can be safely specialized for any
      // enumeration type for more human-readable logs if required
      return std::to_string(static_cast<std::underlying_type_t<T>>(val));
    } else {
      return std::to_string(val);
    }
  }
};

// Provides generic stringification for logging purposes
// To tweak for specific type, please:
// - either provide overload,
// - or specialize the StringMaker for this type
template <typename T> static std::string stringify(T val) {
  return log::StringMaker<T>::stringify(val);
}

// Overload to improve performance a bit; works as the first-class citizen
inline const std::string &stringify(const std::string &val) { return val; }

// Print line to the log using a custom set of parameters
//
// Lambda can be used to print out any custom information,
// so every input parameter is one of the following:
//  - lambda returning the message to log
//  - the message itself
//  - scalar data value
//
// Usage examples:
//
//    log.print_line([&]{
//      return "Unexpected exception for " + TestDescriptionT(srcType, dstType);
//    });
//    log.print_line("Value: ", 42.25f);
//    log.print_line("String view instance"sv);
//
template <typename... argsT> inline void print_line(const argsT &...args) {
  static_assert(sizeof...(argsT) > 0, "Zero arguments not supported");

  (std::cout << ... << [&]() {
    if constexpr (std::is_invocable_v<decltype(args)>) {
      // Using factory method
      return stringify(args());
    } else {
      // Using data value
      return stringify(args);
    }
  }());
  // Force output buffer flush after each log line to have all logs
  // available in case of `abort()` because of a test crash
  std::cout << std::endl;
}

// Output debug information
// Provides output only if the ESIMD_TESTS_VERBOSE_LOG macro was defined.
//
// Lambda parameters support makes the message construction as lazy as possible,
// for example:
//
//    log.debug([&]{
//      return "Running for test case: " +
//             TestDescriptionT(srcType, dstType).to_string();
//    });
//
// would not trigger any std::string construction and manipulation with debug
// logging disabled
template <typename... argsT> inline void debug(const argsT &...args) {
#ifdef ESIMD_TESTS_VERBOSE_LOG
  print_line(args...);
#else
  // Suppress unused variables warning
  (static_cast<void>(args), ...);
#endif
}

// Syntax sugar for tracing test cases; could be usefull in case of test crash
// or freeze
template <typename TestCaseDescriptionT, typename... argsT>
inline void trace(const argsT &...args) {
#ifdef ESIMD_TESTS_VERBOSE_LOG
  const ITestDescription &description = TestCaseDescriptionT(args...);
  print_line("Running for test case: ", description.to_string());
#else
  // Suppress unused variables warning
  (static_cast<void>(args), ...);
#endif
}

// Output non-failure message alongside with test case description
template <typename... detailsT>
void note(const ITestDescription &test_description,
          const detailsT &...details) {
  print_line("Test case: ", test_description.to_string(), ". ", details...);
}

#if !defined ESIMD_TESTS_DISABLE_DEPRECATED_TEST_DESCRIPTION_FOR_LOGS
// API left for backward compatibility only
// Use log::note(const ITestDescription&, ..) instead
[[deprecated]] void note(const std::string& message) {
  print_line(message);
}
#endif

// Output failure details alongside with failed test case description
template <typename... detailsT>
void fail(const ITestDescription &test_description,
          const detailsT &...details) {
  print_line("Failed for ", test_description.to_string(), ". ", details...);
}

} // namespace log

} // namespace esimd_test::api::functional
