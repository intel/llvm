//===-- type_coverage.hpp - Define generic functions for type coverage. ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides a generic way to run tests with different types.
///
//===----------------------------------------------------------------------===//

#pragma once

#include <string>
#include <type_traits>
#include <utility>

namespace esimd_test::api::functional {

// Integer pack to store provided int values
template <int... T> struct values_pack {
  values_pack() {}
};

// Type pack to store types and underlying data type names to use with
// type_name_string
template <typename... T> struct named_type_pack {
  const std::string names[sizeof...(T)];

  template <typename... nameListT>
  named_type_pack(nameListT &&... nameList)
      : names{std::forward<nameListT>(nameList)...} {}
};

enum class tested_types { all, fp, uint, sint };

// Factory method to retrieve pre-defined named_type_pack, to have the same
// default type coverage over the tests
template <tested_types required> auto get_tested_types() {
  if constexpr (required == tested_types::all) {
    return named_type_pack<char, unsigned char, signed char, short,
                           unsigned short, int, unsigned int, long,
                           unsigned long, float, sycl::half, double, long long,
                           unsigned long long>(
        {"char", "unsigned char", "signed char", "short", "unsigned short",
         "int", "unsigned int", "long", "unsigned long", "float", "sycl::half",
         "double", "long long", "unsigned long long"});
  } else if constexpr (required == tested_types::fp) {
    return named_type_pack<float, sycl::half, double>(
        {"float", "sycl::half", "double"});
  } else if constexpr (required == tested_types::uint) {
    if constexpr (!std::is_signed_v<char>) {
      return named_type_pack<unsigned char, unsigned short, unsigned int,
                             unsigned long, unsigned long long, char>(
          {"unsigned char", "unsigned short", "unsigned int", "unsigned long",
           "unsigned long long", "char"});
    } else {
      return named_type_pack<unsigned char, unsigned short, unsigned int,
                             unsigned long, unsigned long long>(
          {"unsigned char", "unsigned short", "unsigned int", "unsigned long",
           "unsigned long long"});
    }
  } else if constexpr (required == tested_types::sint) {
    if constexpr (std::is_signed_v<char>) {
      return named_type_pack<signed char, short, int, long, long long, char>(
          {"signed char", "short", "int", "long", "long long", "char"});
    } else {
      return named_type_pack<signed char, short, int, long, long long>(
          {"signed char", "short", "int", "long", "long long"});
    }
  } else {
    static_assert(required != required, "Unexpected tested type");
  }
}

// Factory method to retrieve pre-defined values_pack, to have the same
// default dimensions over the tests
auto get_all_dimensions() { return values_pack<1, 8, 16, 32>(); }

// Run action for each of types given by named_type_pack instance
template <template <typename, int, typename...> class Action, int N,
          typename... ActionArgsT, typename... Types, typename... ArgsT>
inline bool for_all_types(const named_type_pack<Types...> &type_list,
                          ArgsT &&... args) {
  bool passed{true};

  size_t type_name_index = 0;

  // Run action for each type from named_type_pack... parameter pack
  ((passed &= Action<Types, N, ActionArgsT...>{}(
        std::forward<ArgsT>(args)..., type_list.names[type_name_index]),
    ++type_name_index),
   ...);
  // The unary right fold expression is used for parameter pack expansion.
  // Every expression with comma operator is strictly sequenced, so we can
  // increment safely. And of course the fold expression would not be optimized
  // out due to side-effects.
  // Additional pair of brackets is required because of precedence of increment
  // operator relative to the comma operator.
  //
  // Note that there is actually no difference in left or right fold expression
  // for the comma operator, as it would give the same order of actions
  // execution and the same order of the type name index increment: both the
  // "(expr0, (exr1, expr2))" and "((expr0, expr1), expr2)" would give the same
  //  result as simple "expr0, expr1, expr2"

  return passed;
}

// Calls for_all_types for each vector length by values_pack instance
template <template <typename, int, typename...> class Action,
          typename... ActionArgsT, typename... Types, int... Dims,
          typename... ArgsT>
inline bool for_all_types_and_dims(const named_type_pack<Types...> &type_list,
                                   const values_pack<Dims...> &dim_list,
                                   ArgsT &&... args) {
  bool passed{true};

  // Run action for each value from values_pack... parameter pack
  ((passed &= for_all_types<Action, Dims, ActionArgsT...>(
        type_list, std::forward<ArgsT>(args)...)),
   ...);
  // The unary right fold expression is used for parameter pack expansion.
  // An additional pair of brackets is required because of precedence of any
  // operator relatively to the comma operator.

  return passed;
}

} // namespace esimd_test::api::functional
