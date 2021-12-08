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

namespace esimd_test {
namespace api {
namespace functional {

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
#ifdef TEST_HALF
    return named_type_pack<char, unsigned char, signed char, short,
                           unsigned short, int, unsigned int, long,
                           unsigned long, float, sycl::half, double, long long,
                           unsigned long long>(
        {"char", "unsigned char", "signed char", "short", "unsigned short",
         "int", "unsigned int", "long", "unsigned long", "float", "sycl::half",
         "double", "long long", "unsigned long long"});
#else
    return named_type_pack<char, unsigned char, signed char, short,
                           unsigned short, int, unsigned int, long,
                           unsigned long, float, double, long long,
                           unsigned long long>(
        {"char", "unsigned char", "signed char", "short", "unsigned short",
         "int", "unsigned int", "long", "unsigned long", "float", "double",
         "long long", "unsigned long long"});
#endif
  } else if constexpr (required == tested_types::fp) {
#ifdef TEST_HALF
    return named_type_pack<float, sycl::half, double>(
        {"float", "sycl::half", "double"});
#else
    return named_type_pack<float, double>({"float", "double"});
#endif
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

// Run action for each of types given by named_type_pack instance
template <template <typename, typename...> class action,
          typename... actionArgsT, typename... types, typename... argsT>
inline bool for_all_types(const named_type_pack<types...> &typeList,
                          argsT &&... args) {
  bool passed{true};

  // run action for each type from types... parameter pack
  size_t typeNameIndex = 0;

  int packExpansion[] = {
      (passed &= action<types, actionArgsT...>{}(std::forward<argsT>(args)...,
                                                 typeList.names[typeNameIndex]),
       ++typeNameIndex,
       0 // Dummy initialization value
       )...};
  // Every initializer clause is sequenced before any initializer clause that
  // follows it in the braced-init-list. Every expression in comma operator is
  // also strictly sequenced. So we can use increment safely. We still should
  // discard dummy results, but this initialization should not be optimized out
  // due side-effects
  static_cast<void>(packExpansion);

  return passed;
}

} // namespace functional
} // namespace api
} // namespace esimd_test
