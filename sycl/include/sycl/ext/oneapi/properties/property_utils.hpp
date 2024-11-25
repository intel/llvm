//==-- property_utils.hpp --- SYCL extended property list common utilities -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/oneapi/properties/property.hpp>
#include <sycl/ext/oneapi/properties/property_value.hpp>

#include <array>       // for tuple_element
#include <stddef.h>    // for size_t
#include <tuple>       // for tuple
#include <type_traits> // for false_type, true_...
#include <variant>     // for tuple

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {
namespace detail {

//******************************************************************************
// Misc
//******************************************************************************

#if __has_builtin(__type_pack_element)
template <int N, typename... Ts>
using nth_type_t = __type_pack_element<N, Ts...>;
#else
template <int N, typename T, typename... Ts> struct nth_type {
  using type = typename nth_type<N - 1, Ts...>::type;
};

template <typename T, typename... Ts> struct nth_type<0, T, Ts...> {
  using type = T;
};

template <int N, typename... Ts>
using nth_type_t = typename nth_type<N, Ts...>::type;
#endif

//******************************************************************************
// Property value tooling
//******************************************************************************

// Simple helpers for containing primitive types as template arguments.
template <size_t... Sizes> struct SizeList {};
template <char... Sizes> struct CharList {};

// Helper for converting characters to a constexpr string.
template <char... Chars> struct CharsToStr {
  static constexpr const char value[] = {Chars..., '\0'};
};

// Helper for converting a list of size_t values to a comma-separated string
// representation. This is done by extracting the digit one-by-one and when
// finishing a value, the parsed result is added to a separate list of
// "parsed" characters with the delimiter.
template <typename List, typename ParsedList, char... Chars>
struct SizeListToStrHelper;

// Specialization for when we are in the process of converting a non-zero value
// (Value). Chars will have the already converted digits of the original value
// being converted. Instantiation of this will convert the least significant
// digit in Value.
// Example:
//  - Current: SizeListToStrHelper<SizeList<12>, CharList<'1', '0', ','>, '3'>
//  - Next: SizeListToStrHelper<SizeList<1>, CharList<'1', '0', ','>, '2', '3'>
//  - Outermost: SizeListToStrHelper<SizeList<10,123>, CharList<>>
//  - Final: SizeListToStrHelper<SizeList<0>,
//                               CharList<'1', '0', ','>, '1', '2', '3'>>
//  - Result string: "10,123"
template <size_t Value, size_t... Values, char... ParsedChars, char... Chars>
struct SizeListToStrHelper<SizeList<Value, Values...>, CharList<ParsedChars...>,
                           Chars...>
    : SizeListToStrHelper<SizeList<Value / 10, Values...>,
                          CharList<ParsedChars...>, '0' + (Value % 10),
                          Chars...> {};

// Specialization for when we have reached 0 in the current value we are
// converting. In this case we are done with converting the current value and
// we insert the converted digits from Chars into ParsedChars.
// Example:
//  - Current: SizeListToStrHelper<SizeList<0,123>, CharList<>, '1', '0'>
//  - Next: SizeListToStrHelper<SizeList<123>, CharList<'1', '0', ','>>
//  - Outermost: SizeListToStrHelper<SizeList<10,123>, CharList<>>
//  - Final: SizeListToStrHelper<SizeList<0>,
//                               CharList<'1', '0', ','>, '1', '2', '3'>>
//  - Result string: "10,123"
template <size_t... Values, char... ParsedChars, char... Chars>
struct SizeListToStrHelper<SizeList<0, Values...>, CharList<ParsedChars...>,
                           Chars...>
    : SizeListToStrHelper<SizeList<Values...>,
                          CharList<ParsedChars..., Chars..., ','>> {};

// Specialization for the special case where the value we are converting is 0
// but the list of converted digits is empty. This means there was a 0 value in
// the list and we can add it to ParsedChars directly.
// Example:
//  - Current: SizeListToStrHelper<SizeList<0,123>, CharList<>>
//  - Next: SizeListToStrHelper<SizeList<123>, CharList<'0', ','>>
//  - Outermost: SizeListToStrHelper<SizeList<0,123>, CharList<>>
//  - Final: SizeListToStrHelper<SizeList<0>,
//                               CharList<'0', ','>, '1', '2', '3'>>
//  - Result string: "0,123"
template <size_t... Values, char... ParsedChars>
struct SizeListToStrHelper<SizeList<0, Values...>, CharList<ParsedChars...>>
    : SizeListToStrHelper<SizeList<Values...>,
                          CharList<ParsedChars..., '0', ','>> {};

// Specialization for when we have reached 0 in the current value we are
// converting and there a no more values to parse. In this case we are done with
// converting the current value and we insert the converted digits from Chars
// into ParsedChars. We do not add a ',' as it is the end of the list.
// Example:
//  - Current: SizeListToStrHelper<SizeList<0>, CharList<'1', '0', ','>, '1',
//  '2', '3'>>
//  - Next: None.
//  - Outermost: SizeListToStrHelper<SizeList<10,123>, CharList<>>
//  - Final: SizeListToStrHelper<SizeList<0>,
//                               CharList<'1', '0', ','>, '1', '2', '3'>>
//  - Result string: "10,123"
template <char... ParsedChars, char... Chars>
struct SizeListToStrHelper<SizeList<0>, CharList<ParsedChars...>, Chars...>
    : CharsToStr<ParsedChars..., Chars...> {};

// Specialization for when we have reached 0 in the current value we are
// converting and there a no more values to parse, but the list of converted
// digits is empty. This means the last value in the list was a 0 so we can add
// that to the ParsedChars and finish.
// Example:
//  - Current: SizeListToStrHelper<SizeList<0>, CharList<'1', '0', ','>>>
//  - Next: None.
//  - Outermost: SizeListToStrHelper<SizeList<10,0>, CharList<>>
//  - Final: SizeListToStrHelper<SizeList<0>, CharList<>, '1', '0'>>
//  - Result string: "10,0"
template <char... ParsedChars>
struct SizeListToStrHelper<SizeList<0>, CharList<ParsedChars...>>
    : CharsToStr<ParsedChars..., '0'> {};

// Specialization for the empty list of values to convert. This results in an
// empty string.
template <>
struct SizeListToStrHelper<SizeList<>, CharList<>> : CharsToStr<> {};

// Converts size_t values to a comma-separated string representation.
template <size_t... Sizes>
struct SizeListToStr : SizeListToStrHelper<SizeList<Sizes...>, CharList<>> {};

//******************************************************************************
// Property mutual exclusivity
//******************************************************************************

// Specializations of the following trait should not consider itself a
// conflicting property.
template <typename PropKey, typename Properties>
struct ConflictingProperties : std::false_type {};

//******************************************************************************
// Conditional property meta-info
//******************************************************************************

// Base class for property meta info that is ignored when propagating
// information through the compiler.
struct IgnoredPropertyMetaInfo {
  static constexpr const char *name = "";
  static constexpr std::nullptr_t value = nullptr;
};

// Trait for picking either property meta information for PropT if Condition is
// true or ignored information if Condition is false.
template <typename PropT, bool Condition>
struct ConditionalPropertyMetaInfo
    : std::conditional_t<Condition, PropertyMetaInfo<PropT>,
                         IgnoredPropertyMetaInfo> {};

} // namespace detail
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
