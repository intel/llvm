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

#include "logger.hpp"
#include "type_traits.hpp"

#include <cassert>
#include <string>
#include <type_traits>
#include <utility>

namespace esimd_test::api::functional {

//----------------------------------------------------------------------------//
// Forward-declare all type and value packs provided below
//----------------------------------------------------------------------------//

template <typename... Types> struct unnamed_type_pack;
template <typename... Types> struct named_type_pack;
template <typename T, T... values> struct value_pack;

// Alias to use mainly for simd vector sizes; no overhead as alias doesn't
// declare a new type
template <int... values> using integer_pack = value_pack<int, values...>;

//----------------------------------------------------------------------------//
// SFINAE helpers for type and value packs
//----------------------------------------------------------------------------//

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

//----------------------------------------------------------------------------//
// Implementation details for type packs filtration support
//----------------------------------------------------------------------------//

namespace detail {

template <typename... T> struct type_accumulator {};

// Helper function to accumulate the filtered types without re-ordering them
template <template <typename> class FilterT, bool ExpectedValue, typename T,
          typename... Types, typename... PackTypesT>
auto filter_unnamed_type_pack_impl(
    const type_accumulator<PackTypesT...> & = {}) {
  // Define if we should filter out the current type or store it
  constexpr bool store_type = FilterT<T>::value == ExpectedValue;

  if constexpr (sizeof...(Types) == 0) {
    if constexpr (store_type) {
      return unnamed_type_pack<PackTypesT..., T>::generate();
    } else {
      static_assert(sizeof...(PackTypesT) > 0, "All types were filtered out");
      return unnamed_type_pack<PackTypesT...>::generate();
    }
  } else {
    using next_pack_type =
        std::conditional_t<store_type, type_accumulator<PackTypesT..., T>,
                           type_accumulator<PackTypesT...>>;
    return filter_unnamed_type_pack_impl<FilterT, ExpectedValue, Types...>(
        next_pack_type{});
  }
}

// Helper function to accumulate the filtered types and corresponding type names
// without re-ordering them
template <template <typename> class FilterT, bool ExpectedValue, typename T,
          typename... Types, int N, typename... PackTypesT,
          typename... NameListT>
auto filter_named_type_pack_impl(const std::string (&names)[N],
                                 int typeNameIndex,
                                 const type_accumulator<PackTypesT...> &,
                                 const NameListT &...nameList) {
  // Define if we should filter out the current type or store it
  constexpr bool store_type = FilterT<T>::value == ExpectedValue;
  const auto &name = names[typeNameIndex];

  ++typeNameIndex;

  if constexpr (sizeof...(Types) == 0) {
    if constexpr (store_type) {
      return named_type_pack<PackTypesT..., T>::generate(nameList..., name);
    } else {
      static_assert(sizeof...(PackTypesT) > 0, "All types were filtered out");
      return named_type_pack<PackTypesT...>::generate(nameList...);
    }
  } else {
    if constexpr (store_type) {
      return filter_named_type_pack_impl<FilterT, ExpectedValue, Types...>(
          names, typeNameIndex, type_accumulator<PackTypesT..., T>{},
          nameList..., name);
    } else {
      return filter_named_type_pack_impl<FilterT, ExpectedValue, Types...>(
          names, typeNameIndex, type_accumulator<PackTypesT...>{}, nameList...);
    }
  }
}

// Filters out types using the filter functor template
template <template <typename> class FilterT, bool ExpectedValue,
          typename... Types>
inline auto filter_type_pack() {
  return filter_unnamed_type_pack_impl<FilterT, ExpectedValue, Types...>();
}

// Filters out types using the filter functor template and stores corrensponding
// type names appropriately
template <template <typename> class FilterT, bool ExpectedValue,
          typename... Types, int N>
inline auto filter_type_pack(const std::string (&names)[N]) {
  return filter_named_type_pack_impl<FilterT, ExpectedValue, Types...>(
      names, 0, type_accumulator<>{});
}

} // namespace detail

//----------------------------------------------------------------------------//
// All type and value packs implementation
//----------------------------------------------------------------------------//

// Generic type pack with no specific type names provided
template <typename... Types> struct unnamed_type_pack {
  static_assert(sizeof...(Types) > 0, "Empty pack is not supported");

  // Syntax sugar to align usage with the named_type_pack
  static auto inline generate() { return unnamed_type_pack<Types...>{}; }

  // Makes possible to generate new unnamed type pack by filtering out types
  // with the given filter functor. Does not allow to filter out all the types.
  //
  // For example:
  //   const auto types = unnamed_type_pack<int, unsigned int>::generate();
  //   const auto unsigned_types = types.filter_by<std::is_unsigned>();
  //   const auto non_fp_types =
  //       types.filter_by<std::is_floating_point, false>();
  //
  template <template <typename> class FilterT, bool ExpectedValue = true>
  auto filter_by() const {
    return detail::filter_type_pack<FilterT, ExpectedValue, Types...>();
  }
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

  // Makes possible to generate new named type pack by filtering out types and
  // names with the given filter functor.
  // Does not allow to filter out all the types.
  //
  // For example:
  //   const auto types = named_type_pack<int, unsigned int>::generate("int",
  //                                                                   "uint");
  //  template <typename DataT> using is_uint =
  //      std::bool_constant<std::is_unsigned_v<DataT> &&
  //                         std::is_integral_v<DataT>>;
  //
  //   const auto uint_types = types.filter_by<is_uint>();
  //   const auto unsigned_types = types.filter_by<std::is_unsigned>();
  //
  template <template <typename> class FilterT, bool ExpectedValue = true>
  auto filter_by() const {
    return detail::filter_type_pack<FilterT, ExpectedValue, Types...>(names);
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
  // within. Would work with or without specific StringMaker specializations.
  // For example:
  //   const auto targets =
  //     value_pack<sycl::target, sycl::target::device>::generate_named();
  //
  static inline auto generate_named() {
    return named_type_pack<std::integral_constant<T, values>...>::generate(
        log::stringify(values)...);
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

  // Factory function to generate the type pack using generator function
  //
  // It could be especially usefull for value packs with enums. For example:
  //   enum class access {read, write};
  //   template <access ... values>
  //   using access_pack = value_pack<access, values...>;
  //
  //   const auto generator = [&](const access& value) {
  //        switch (value) {
  //        case access:read:
  //           return "read access";
  //           break;
  //        ...
  //        };
  //     };
  //   const auto accesses =
  //     access_pack<access::read, access::write>::generate_named_by(generator);
  //
  template <typename GeneratorT>
  static auto generate_named_by(const GeneratorT &nameGenerator) {
    static_assert(std::is_invocable_v<GeneratorT, T>,
                  "Unexpected name generator type");
    return generate_named(nameGenerator(values)...);
  }
};

//----------------------------------------------------------------------------//
// Support of the combinatorial type and value packs usage
//----------------------------------------------------------------------------//

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

//----------------------------------------------------------------------------//
// Specific pack generation helpers
//----------------------------------------------------------------------------//

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
#ifdef ESIMD_TESTS_FULL_COVERAGE
    return named_type_pack<
        char, unsigned char, signed char, short, unsigned short, int,
        unsigned int, long, unsigned long, float, long long,
        unsigned long long>::generate("char", "unsigned char", "signed char",
                                      "short", "unsigned short", "int",
                                      "unsigned int", "long", "unsigned long",
                                      "float", "long long",
                                      "unsigned long long");
#else
    return named_type_pack<float, int, unsigned int, signed char>::generate(
        "float", "int", "unsigned int", "signed char");
#endif
  } else if constexpr (required == tested_types::fp) {
    return named_type_pack<float>::generate("float");
  } else if constexpr (required == tested_types::fp_extra) {
    return named_type_pack<sycl::half, double>::generate("sycl::half",
                                                         "double");
  } else if constexpr (required == tested_types::uint) {
#ifdef ESIMD_TESTS_FULL_COVERAGE
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
#else
    return named_type_pack<unsigned int>::generate("unsigned int");
#endif
  } else if constexpr (required == tested_types::sint) {
#ifdef ESIMD_TESTS_FULL_COVERAGE
    if constexpr (std::is_signed_v<char>) {
      return named_type_pack<signed char, short, int, long, long long,
                             char>::generate("signed char", "short", "int",
                                             "long", "long long", "char");
    } else {
      return named_type_pack<signed char, short, int, long,
                             long long>::generate("signed char", "short", "int",
                                                  "long", "long long");
    }
#else
    return named_type_pack<int, signed char>::generate("int", "signed char");
#endif
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
auto inline get_all_sizes() {
#ifdef ESIMD_TESTS_FULL_COVERAGE
  return get_sizes<1, 8, 16, 32>();
#else
  return get_sizes<1, 8>();
#endif
}

// It's a deprecated function and it exists only for backward compatibility and
// it should be deleted in the future. Use get_all_sizes() instead.
auto inline get_all_dimensions() { return get_all_sizes(); }

// It's a deprecated function and it exists only for backward compatibility and
// it should be deleted in the future. Use get_sizes() instead.
template <int... Values> auto inline get_dimensions() {
  return get_sizes<Values...>();
}

} // namespace esimd_test::api::functional
