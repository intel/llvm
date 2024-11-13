//==-- property_utils.hpp --- SYCL extended property list common utilities -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/boost/mp11/algorithm.hpp>        // for mp_sort_q
#include <sycl/detail/boost/mp11/detail/mp_list.hpp>   // for mp_list
#include <sycl/detail/boost/mp11/detail/mp_rename.hpp> // for mp_rename
#include <sycl/detail/boost/mp11/integral.hpp>         // for mp_bool
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
template <typename... property_tys> struct properties_type_list;

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

template <typename T, typename PropList> struct PrependProperty {};
template <typename T, typename... Ts>
struct PrependProperty<T, properties_type_list<Ts...>> {
  using type = properties_type_list<T, Ts...>;
};

//******************************************************************************
// Property identification
//******************************************************************************

// Checks that all types in a tuple are valid properties.
template <typename T> struct AllPropertyValues {};
template <typename... Ts>
struct AllPropertyValues<std::tuple<Ts...>> : std::true_type {};
template <typename T, typename... Ts>
struct AllPropertyValues<std::tuple<T, Ts...>>
    : std::conditional_t<IsPropertyValue<T>::value,
                         AllPropertyValues<std::tuple<Ts...>>,
                         std::false_type> {};

//******************************************************************************
// Property type sorting
//******************************************************************************

// Splits a tuple into head and tail if ShouldSplit is true. If ShouldSplit is
// false the head will be void and the tail will be the full tuple.
template <typename T1, bool ShouldSplit> struct HeadSplit {};
template <typename T, typename... Ts>
struct HeadSplit<properties_type_list<T, Ts...>, true> {
  using htype = T;
  using ttype = properties_type_list<Ts...>;
};
template <typename... Ts> struct HeadSplit<properties_type_list<Ts...>, false> {
  using htype = void;
  using ttype = properties_type_list<Ts...>;
};

// Selects the one of two types that is not void. This assumes that at least one
// of the two template arguemnts is void.
template <typename LHS, typename RHS> struct SelectNonVoid {};
template <typename LHS> struct SelectNonVoid<LHS, void> {
  using type = LHS;
};
template <typename RHS> struct SelectNonVoid<void, RHS> {
  using type = RHS;
};

// Sort types accoring to their PropertyID.
struct SortByPropertyId {
  template <typename T1, typename T2>
  using fn = sycl::detail::boost::mp11::mp_bool<(PropertyID<T1>::value <
                                                 PropertyID<T2>::value)>;
};
template <typename... Ts> struct Sorted {
  static_assert(detail::AllPropertyValues<std::tuple<Ts...>>::value,
                "Unrecognized property in property list.");
  using properties = sycl::detail::boost::mp11::mp_list<Ts...>;
  using sortedProperties =
      sycl::detail::boost::mp11::mp_sort_q<properties, SortByPropertyId>;
  using type =
      sycl::detail::boost::mp11::mp_rename<sortedProperties,
                                           detail::properties_type_list>;
};

//******************************************************************************
// Property merging
//******************************************************************************

// Merges two sets of properties, failing if two properties are the same but
// with different values.
// NOTE: This assumes that the properties are in sorted order.
template <typename LHSPropertyT, typename RHSPropertyT> struct MergeProperties;

template <>
struct MergeProperties<properties_type_list<>, properties_type_list<>> {
  using type = properties_type_list<>;
};

template <typename... LHSPropertyTs>
struct MergeProperties<properties_type_list<LHSPropertyTs...>,
                       properties_type_list<>> {
  using type = properties_type_list<LHSPropertyTs...>;
};

template <typename... RHSPropertyTs>
struct MergeProperties<properties_type_list<>,
                       properties_type_list<RHSPropertyTs...>> {
  using type = properties_type_list<RHSPropertyTs...>;
};

// Identical properties are allowed, but only one will carry over.
template <typename PropertyT, typename... LHSPropertyTs,
          typename... RHSPropertyTs>
struct MergeProperties<properties_type_list<PropertyT, LHSPropertyTs...>,
                       properties_type_list<PropertyT, RHSPropertyTs...>> {
  using merge_tails =
      typename MergeProperties<properties_type_list<LHSPropertyTs...>,
                               properties_type_list<RHSPropertyTs...>>::type;
  using type = typename PrependProperty<PropertyT, merge_tails>::type;
};

template <typename... LHSPropertyTs, typename... RHSPropertyTs>
struct MergeProperties<properties_type_list<LHSPropertyTs...>,
                       properties_type_list<RHSPropertyTs...>> {
  using l_head = nth_type_t<0, LHSPropertyTs...>;
  using r_head = nth_type_t<0, RHSPropertyTs...>;
  static_assert(
      PropertyID<l_head>::value != PropertyID<r_head>::value,
      "Failed to merge property lists due to conflicting properties.");
  static constexpr bool left_has_min =
      PropertyID<l_head>::value < PropertyID<r_head>::value;
  using l_split =
      HeadSplit<properties_type_list<LHSPropertyTs...>, left_has_min>;
  using r_split =
      HeadSplit<properties_type_list<RHSPropertyTs...>, !left_has_min>;
  using min = typename SelectNonVoid<typename l_split::htype,
                                     typename r_split::htype>::type;
  using merge_tails = typename MergeProperties<typename l_split::ttype,
                                               typename r_split::ttype>::type;
  using type = typename PrependProperty<min, merge_tails>::type;
};

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

template <typename Properties, typename T>
struct NoConflictingPropertiesHelper {};

template <typename Properties, typename... Ts>
struct NoConflictingPropertiesHelper<Properties, std::tuple<Ts...>>
    : std::true_type {};

template <typename Properties, typename T, typename... Ts>
struct NoConflictingPropertiesHelper<Properties, std::tuple<T, Ts...>>
    : NoConflictingPropertiesHelper<Properties, std::tuple<Ts...>> {};

template <typename Properties, typename... Rest, typename PropT,
          typename... PropValuesTs>
struct NoConflictingPropertiesHelper<
    Properties, std::tuple<property_value<PropT, PropValuesTs...>, Rest...>>
    : std::conditional_t<
          ConflictingProperties<PropT, Properties>::value, std::false_type,
          NoConflictingPropertiesHelper<Properties, std::tuple<Rest...>>> {};
template <typename PropertiesT>
struct NoConflictingProperties
    : NoConflictingPropertiesHelper<PropertiesT, PropertiesT> {};

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
