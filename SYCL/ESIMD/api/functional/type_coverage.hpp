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

#include "type_traits.hpp"

#include <cassert>
#include <string>
#include <type_traits>
#include <utility>

namespace esimd_test::api::functional {

// Generic type pack with no specific type names provided
template <typename... Types> struct unnamed_type_pack {
  static_assert(sizeof...(Types) > 0, "Empty pack is not supported");

  // Syntax sugar to align usage with the named_type_pack
  static auto inline generate() { return unnamed_type_pack<Types...>{}; }
};

// Generic type pack with specific type names provided; intended to use mainly
// for type coverage over the ESIMD API tests
template <typename... Types> class named_type_pack {
  static_assert(sizeof...(Types) > 0, "Empty pack is not supported");

  template <typename... nameListT>
  named_type_pack(nameListT &&...nameList)
      : names{std::forward<nameListT>(nameList)...} {}

  template <typename T> static inline auto generate_name() {
    if constexpr (type_traits::has_static_member::to_string<T>::value) {
      const auto result = T::to_string();
      static_assert(std::is_same_v<decltype(result), const std::string>,
                    "Unexpected return type for the T::to_string() method");
      return result;
    } else {
      constexpr auto always_false = !std::is_same_v<T, T>;
      static_assert(always_false,
                    "There is no static method T::to_string() for this type");
    }
  }

public:
  // We need a specific names to differentiate types on logic level, with no
  // dependency on actual type implementation and typeid
  const std::string names[sizeof...(Types)];

  // Factory function to properly generate the type pack
  //
  // There are two possible use-cases for generation:
  // - either each type has a corresponding name provided,
  // - or each type have a static T::to_string() method available
  //
  // For example:
  //   struct var_decl {
  //     static std::string to_string() { return "variable declaration"; }
  //   };
  //   struct rval_in_expr {
  //     static std::string to_string() { return "rvalue in an expression"; }
  //   };
  //   const auto types =
  //      named_type_pack<char, signed char>::generate("char", "signed char");
  //   const auto contexts =
  //      named_type_pack<var_decl, rval_in_expr>::generate();
  //
  template <typename... nameListT>
  static auto generate(nameListT &&...nameList) {
    if constexpr (sizeof...(nameListT) == 0) {
      // No names provided explicitly, try to generate them
      return named_type_pack<Types...>(generate_name<Types>()...);
    } else {
      // Make requirement explicit to have more clear error message
      static_assert(sizeof...(Types) == sizeof...(nameListT));
      return named_type_pack<Types...>(std::forward<nameListT>(nameList)...);
    }
  }
};

// Generic value pack to use for any type of compile-time lists
template <typename T, T... values> struct value_pack {

  // Factory function to generate the corresponding type pack with no names
  // stored
  //
  // Might be useful to store plain integral values or enumeration values.
  // For example:
  //   const auto bytes = value_pack<int, 1, 2, 8>::generate_unnamed();
  //
  static inline auto generate_unnamed() {
    return unnamed_type_pack<std::integral_constant<T, values>...>::generate();
  }

  // Factory function to generate the type pack with stringified values stored
  // within
  static inline auto generate_named() {
    if constexpr (std::is_enum_v<T>) {
      // Any enumeration requires explicit cast to the underlying type
      return named_type_pack<std::integral_constant<T, values>...>::generate(
          std::to_string(static_cast<std::underlying_type_t<T>>(values))...);
    } else {
      return named_type_pack<std::integral_constant<T, values>...>::generate(
          std::to_string(values)...);
    }
  }

  // Factory function to generate the type pack with names given for each value
  //
  // For example:
  //   enum class ctx : int {
  //     var_decl = 0,
  //     rval_in_expr
  //   };
  //   const auto contexts =
  //     value_pack<ctx, ctx::var_decl, ctx::rval_in_expr>::generate_named(
  //         "variable declaration", "rvalue in an expression");
  //
  template <typename... argsT>
  static inline auto generate_named(argsT &&...args) {
    return named_type_pack<std::integral_constant<T, values>...>::generate(
        std::forward<argsT>(args)...);
  }
};

// Alias to use mainly for simd vector sizes; no overhead as alias doesn't
// declare a new type
template <int... values> using integer_pack = value_pack<int, values...>;

namespace sfinae {
namespace details {
template <typename T> struct is_type_pack_t : std::false_type {};

template <typename... Types>
struct is_type_pack_t<named_type_pack<Types...>> : std::true_type {};

template <typename... Types>
struct is_type_pack_t<unnamed_type_pack<Types...>> : std::true_type {};
} // namespace details

template <typename T>
using is_not_a_type_pack =
    std::enable_if_t<!details::is_type_pack_t<T>::value, bool>;

} // namespace sfinae

// Generic function to run specific action for every combination of each of the
// types given by appropriate type pack instances.
// Virtually any combination of named and unnamed type packs is supported.
// Supports different types of compile-time value lists via value pack.
template <template <typename...> class Action, typename... ActionArgsT,
          typename HeadT, typename... ArgsT,
          sfinae::is_not_a_type_pack<HeadT> = true>
inline bool for_all_combinations(HeadT &&head, ArgsT &&...args) {
  // The first non-pack argument passed into the for_all_combinations stops the
  // recursion
  return Action<ActionArgsT...>{}(std::forward<HeadT>(head),
                                  std::forward<ArgsT>(args)...);
}

// Overload to handle the iteration over the types within the named type pack
template <template <typename...> class Action, typename... ActionArgsT,
          typename... HeadTypes, typename... ArgsT>
inline bool for_all_combinations(const named_type_pack<HeadTypes...> &head,
                                 ArgsT &&...args) {
  bool passed = true;

  // Run the next level of recursion for each type from the head named_type_pack
  // instance. Each recursion level unfolds the first argument passed and adds a
  // type name as the last argument.
  size_t type_name_index = 0;

  ((passed &= for_all_combinations<Action, ActionArgsT..., HeadTypes>(
        std::forward<ArgsT>(args)..., head.names[type_name_index]),
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
  assert((type_name_index == sizeof...(HeadTypes)) && "Pack expansion failed");

  return passed;
}

// Overload to handle the iteration over the types within the unnamed type pack
template <template <typename...> class Action, typename... ActionArgsT,
          typename... HeadTypes, typename... ArgsT>
inline bool for_all_combinations(const unnamed_type_pack<HeadTypes...> &head,
                                 ArgsT &&...args) {
  // Using fold expression to iterate over all types within type pack
  bool passed = true;
  size_t count = 0;

  ((passed &= for_all_combinations<Action, ActionArgsT..., HeadTypes>(
        std::forward<ArgsT>(args)...),
    ++count),
   ...);
  // Ensure there is no silent miss for coverage
  assert((count == sizeof...(HeadTypes)) && "Pack expansion failed");

  return passed;
}

// Overload to trigger failure in case of either invalid usage or recursion
// failure
template <template <typename...> class, typename... ArgsT>
inline bool for_all_combinations() {
  constexpr auto always_false = sizeof...(ArgsT) != sizeof...(ArgsT);
  static_assert(always_false, "No packs provided to iterate over");
}

// Provides alias to types that can be used in tests:
//  core - all C++ data types, except specific data types
//  fp - all floating point C++ data types
//  fp_extra - specific, non C++ data types
//  uint - all unsigned C++ integral data types
//  sint - all signed C++ integral data types
enum class tested_types { core, fp, fp_extra, uint, sint };

// Factory method to retrieve pre-defined named_type_pack, to have the same
// default type coverage over the tests
template <tested_types required> auto get_tested_types() {
  if constexpr (required == tested_types::core) {
    return named_type_pack<
        char, unsigned char, signed char, short, unsigned short, int,
        unsigned int, long, unsigned long, float, long long,
        unsigned long long>::generate("char", "unsigned char", "signed char",
                                      "short", "unsigned short", "int",
                                      "unsigned int", "long", "unsigned long",
                                      "float", "long long",
                                      "unsigned long long");
  } else if constexpr (required == tested_types::fp) {
    return named_type_pack<float>::generate("float");
  } else if constexpr (required == tested_types::fp_extra) {
    return named_type_pack<sycl::half, double>::generate("sycl::half",
                                                         "double");
  } else if constexpr (required == tested_types::uint) {
    if constexpr (!std::is_signed_v<char>) {
      return named_type_pack<unsigned char, unsigned short, unsigned int,
                             unsigned long, unsigned long long,
                             char>::generate("unsigned char", "unsigned short",
                                             "unsigned int", "unsigned long",
                                             "unsigned long long", "char");
    } else {
      return named_type_pack<
          unsigned char, unsigned short, unsigned int, unsigned long,
          unsigned long long>::generate("unsigned char", "unsigned short",
                                        "unsigned int", "unsigned long",
                                        "unsigned long long");
    }
  } else if constexpr (required == tested_types::sint) {
    if constexpr (std::is_signed_v<char>) {
      return named_type_pack<signed char, short, int, long, long long,
                             char>::generate("signed char", "short", "int",
                                             "long", "long long", "char");
    } else {
      return named_type_pack<signed char, short, int, long,
                             long long>::generate("signed char", "short", "int",
                                                  "long", "long long");
    }
  } else {
    static_assert(required != required, "Unexpected tested type");
  }
}

// Syntax sugar to retrieve simd vector sizes in a consistent way
template <int... Values> auto inline get_sizes() {
  return integer_pack<Values...>::generate_unnamed();
}

// Factory method to retrieve pre-defined values_pack, to have the same
// default sizes over the tests
auto inline get_all_sizes() { return get_sizes<1, 8, 16, 32>(); }

// It's a deprecated function and it exists only for backward compatibility and
// it should be deleted in the future. Use get_all_sizes() instead.
auto inline get_all_dimensions() { return get_all_sizes(); }

// It's a deprecated function and it exists only for backward compatibility and
// it should be deleted in the future. Use get_sizes() instead.
template <int... Values> auto inline get_dimensions() {
  return get_sizes<Values...>();
}

} // namespace esimd_test::api::functional
