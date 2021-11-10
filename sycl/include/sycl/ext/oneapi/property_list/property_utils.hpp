//==-- property_utils.hpp --- SYCL extended property list common utilities -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/property_helper.hpp>

#include <tuple>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace oneapi {

// Forward declaration
template <class PropertyT, class T, class... Ts> struct property_value;

namespace detail {

//******************************************************************************
// Misc
//******************************************************************************

// Checks if a type is a tuple.
template <typename T> struct IsTuple : std::false_type {};
template <typename... Ts> struct IsTuple<std::tuple<Ts...>> : std::true_type {};

// Gets the first type in a parameter pack of types.
template <typename... Ts>
using GetFirstType = typename std::tuple_element<0, std::tuple<Ts...>>::type;

// Prepends a value to a tuple.
template <typename T, typename Tuple> struct PrependTuple {};
template <typename T, typename... Ts>
struct PrependTuple<T, std::tuple<Ts...>> {
  using type = std::tuple<T, Ts...>;
};

// Checks if a type T has a static value member variable.
template <typename T, typename U = int> struct HasValue : std::false_type {};
template <typename T>
struct HasValue<T, decltype((void)T::value, 0)> : std::true_type {};

// Checks that all types in a tuple have unique PropertyID.
template <typename T> struct AllUnique {};
template <typename... Ts>
struct AllUnique<std::tuple<Ts...>> : std::true_type {};
template <typename T> struct AllUnique<std::tuple<T>> : std::true_type {};
template <typename L, typename R, typename... Rest>
struct AllUnique<std::tuple<L, R, Rest...>>
    : detail::conditional_t<L::PropertyID != R::PropertyID,
                            AllUnique<std::tuple<R, Rest...>>,
                            std::false_type> {};

//******************************************************************************
// Property identification
//******************************************************************************

// Checks if a type is a runtime property with data.
template <typename T> struct IsRuntimePropertyWithData {
  static constexpr bool value =
      is_property<T>::value && std::is_base_of<PropertyWithDataBase, T>::value;
};

// Checks if a type is a dataless runtime property.
template <typename T> struct IsRuntimeDatalessProperty {
  static constexpr bool value =
      is_property<T>::value && std::is_base_of<DataLessPropertyBase, T>::value;
};

// Checks if a type is a runtime property.
template <typename T> struct IsRuntimeProperty {
  static constexpr bool value =
      is_property<T>::value &&
      (std::is_base_of<DataLessPropertyBase, T>::value ||
       std::is_base_of<PropertyWithDataBase, T>::value);
};

// Checks if a type is either a runtime property (with or without data) or if
// it is a compile-time property (inside a property_value)
template <typename T> struct IntrospectiveIsProperty : is_property<T> {};
template <typename PropertyT, typename... PropertyValueTs>
struct IntrospectiveIsProperty<property_value<PropertyT, PropertyValueTs...>>
    : is_property<PropertyT> {};

// Checks that a type is a compile-time property. value is true if either T is
// a non-runtime property or if it is a property_value containing a non-runtime
// property.
template <typename T> struct IsCompileTimeProperty {
  static constexpr bool value =
      IntrospectiveIsProperty<T>::value && !IsRuntimeProperty<T>::value;
};

// Checks that all types in a tuple are valid properties.
template <typename T> struct AllProperties {};
template <typename... Ts>
struct AllProperties<std::tuple<Ts...>> : std::true_type {};
template <typename T, typename... Ts>
struct AllProperties<std::tuple<T, Ts...>>
    : detail::conditional_t<IntrospectiveIsProperty<T>::value,
                            AllProperties<std::tuple<Ts...>>, std::false_type> {
};

//******************************************************************************
// Property type sorting
//******************************************************************************

// Splits a tuple into head and tail if ShouldSplit is true. If ShouldSplit is
// false the head will be void and the tail will be the full tuple.
template <typename T1, bool ShouldSplit> struct HeadSplit {};
template <typename T, typename... Ts>
struct HeadSplit<std::tuple<T, Ts...>, true> {
  using htype = T;
  using ttype = std::tuple<Ts...>;
};
template <typename... Ts> struct HeadSplit<std::tuple<Ts...>, false> {
  using htype = void;
  using ttype = std::tuple<Ts...>;
};

// Selects the one of two types that is not void. This assumes that at least one
// of the two template arguemnts is void.
template <typename LHS, typename RHS> struct SelectNonVoid {};
template <typename LHS> struct SelectNonVoid<LHS, void> { using type = LHS; };
template <typename RHS> struct SelectNonVoid<void, RHS> { using type = RHS; };

// Merges two tuples by recursively extracting the type with the minimum
// PropertyID in the two tuples and prepending it to the merging of the
// remaining elements.
template <typename T1, typename T2> struct Merge {};
template <typename... LTs> struct Merge<std::tuple<LTs...>, std::tuple<>> {
  using type = std::tuple<LTs...>;
};
template <typename... RTs> struct Merge<std::tuple<>, std::tuple<RTs...>> {
  using type = std::tuple<RTs...>;
};
template <typename... LTs, typename... RTs>
struct Merge<std::tuple<LTs...>, std::tuple<RTs...>> {
  using l_head = GetFirstType<LTs...>;
  using r_head = GetFirstType<RTs...>;
  static constexpr bool left_has_min = l_head::PropertyID < r_head::PropertyID;
  using l_split = HeadSplit<std::tuple<LTs...>, left_has_min>;
  using r_split = HeadSplit<std::tuple<RTs...>, !left_has_min>;
  using min = typename SelectNonVoid<typename l_split::htype,
                                     typename r_split::htype>::type;
  using merge_tails =
      typename Merge<typename l_split::ttype, typename r_split::ttype>::type;
  using type = typename PrependTuple<min, merge_tails>::type;
};

// Creates pairs of tuples with a single element from a tuple with N elements.
// Resulting tuple will have ceil(N/2) elements.
template <typename...> struct CreateTuplePairs {
  using type = typename std::tuple<>;
};
template <typename T> struct CreateTuplePairs<T> {
  using type = typename std::tuple<std::pair<std::tuple<T>, std::tuple<>>>;
};
template <typename L, typename R, typename... Rest>
struct CreateTuplePairs<L, R, Rest...> {
  using type =
      typename PrependTuple<std::pair<std::tuple<L>, std::tuple<R>>,
                            typename CreateTuplePairs<Rest...>::type>::type;
};

// Merges pairs of tuples and creates new pairs of the merged pairs. Let N be
// the number of pairs in the supplied tuple, then the resulting tuple will
// contain ceil(N/2) pairs of tuples.
template <typename T> struct MergePairs { using type = std::tuple<>; };
template <typename... LTs, typename... RTs, typename... Rest>
struct MergePairs<
    std::tuple<std::pair<std::tuple<LTs...>, std::tuple<RTs...>>, Rest...>> {
  using merged = typename Merge<std::tuple<LTs...>, std::tuple<RTs...>>::type;
  using type = std::tuple<std::pair<merged, std::tuple<>>>;
};
template <typename... LLTs, typename... LRTs, typename... RLTs,
          typename... RRTs, typename... Rest>
struct MergePairs<
    std::tuple<std::pair<std::tuple<LLTs...>, std::tuple<LRTs...>>,
               std::pair<std::tuple<RLTs...>, std::tuple<RRTs...>>, Rest...>> {
  using lmerged =
      typename Merge<std::tuple<LLTs...>, std::tuple<LRTs...>>::type;
  using rmerged =
      typename Merge<std::tuple<RLTs...>, std::tuple<RRTs...>>::type;
  using type = typename PrependTuple<
      std::pair<lmerged, rmerged>,
      typename MergePairs<std::tuple<Rest...>>::type>::type;
};

// Recursively merges all pairs of tuples until only a single pair of tuples
// is left, where the right element of the pair is an empty tuple.
template <typename T> struct MergeAll {};
template <typename... Ts> struct MergeAll<std::tuple<Ts...>> {
  using type = std::tuple<Ts...>;
};
template <typename... Ts>
struct MergeAll<std::tuple<std::pair<std::tuple<Ts...>, std::tuple<>>>> {
  using type = std::tuple<Ts...>;
};
template <typename T, typename... Ts> struct MergeAll<std::tuple<T, Ts...>> {
  using reduced = typename MergePairs<std::tuple<T, Ts...>>::type;
  using type = typename MergeAll<reduced>::type;
};

// Performs merge-sort on types with PropertyID.
template <typename... Ts> struct Sorted {
  static_assert(detail::AllProperties<std::tuple<Ts...>>::value,
                "Unrecognized property in property list.");
  using split = typename CreateTuplePairs<Ts...>::type;
  using type = typename MergeAll<split>::type;
};

// Checks if the types in a tuple are sorted w.r.t. their PropertyID.
template <typename T> struct IsSorted {};
template <typename... Ts>
struct IsSorted<std::tuple<Ts...>> : std::true_type {};
template <typename T> struct IsSorted<std::tuple<T>> : std::true_type {};
template <typename L, typename R, typename... Rest>
struct IsSorted<std::tuple<L, R, Rest...>>
    : detail::conditional_t<L::PropertyID <= R::PropertyID,
                            IsSorted<std::tuple<R, Rest...>>, std::false_type> {
};

} // namespace detail
} // namespace oneapi
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
