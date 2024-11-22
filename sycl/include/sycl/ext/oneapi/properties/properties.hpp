//==---------- properties.hpp --- SYCL extended property list --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/is_device_copyable.hpp>
#include <sycl/ext/oneapi/properties/property.hpp>       // for IsRuntimePr...
#include <sycl/ext/oneapi/properties/property_utils.hpp> // for Sorted, Mer...
#include <sycl/ext/oneapi/properties/property_value.hpp> // for property_value

#include <tuple>       // for tuple, tupl...
#include <type_traits> // for enable_if_t
#include <variant>     // for tuple

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

template <typename properties_type_list_ty> class __SYCL_EBO properties;

namespace detail {

// NOTE: Meta-function to implement CTAD rules isn't allowed to return
// `properties<something>` and it's impossible to return a pack as well. As
// such, we're forced to have an extra level of `detail::properties_type_list`
// for the purpose of providing CTAD rules.
template <typename... property_tys> struct properties_type_list;

// This is used in a separate `properties` specialization to report friendlier
// errors.
template <typename... property_tys> struct invalid_properties_type_list {};

// Helper for reconstructing a properties type. This assumes that
// PropertyValueTs is sorted and contains only valid properties.
//
// It also allows us to hide details of `properties` implementation from the
// code that uses/defines them (with the exception of ESIMD which is extremely
// hacky in its own esimd::properties piggybacking on these ones).
template <typename... PropertyValueTs>
using properties_t =
    properties<detail::properties_type_list<PropertyValueTs...>>;

template <typename... property_tys>
inline constexpr bool properties_are_unique = []() constexpr {
  if constexpr (sizeof...(property_tys) == 0) {
    return true;
  } else {
    const std::array kinds = {PropertyID<property_tys>::value...};
    auto N = kinds.size();
    for (std::size_t i = 0; i < N; ++i)
      for (std::size_t j = i + 1; j < N; ++j)
        if (kinds[i] == kinds[j])
          return false;

    return true;
  }
}();

template <typename... property_tys>
inline constexpr bool properties_are_sorted = []() constexpr {
  if constexpr (sizeof...(property_tys) == 0) {
    return true;
  } else {
    const std::array kinds = {PropertyID<property_tys>::value...};
    // std::is_sorted isn't constexpr until C++20.
    for (std::size_t idx = 1; idx < kinds.size(); ++idx)
      if (kinds[idx - 1] >= kinds[idx])
        return false;
    return true;
  }
}();

template <typename... property_tys>
constexpr bool properties_are_valid_for_ctad = []() constexpr {
  // Need `if constexpr` to avoid hard error in "unique" check when querying
  // property kind if `property_tys` isn't a property.
  if constexpr (!((is_property_value_v<property_tys> && ...))) {
    return false;
  } else if constexpr (!detail::properties_are_unique<property_tys...>) {
    return false;
  } else {
    return true;
  }
}();

template <typename... property_tys> struct properties_sorter {
  // Not using "auto" due to MSVC bug in v19.36 and older. v19.37 and later is
  // able to compile "auto" just fine. See https://godbolt.org/z/eW3rjjs7n.
  static constexpr std::array<int, sizeof...(property_tys)> sorted_indices =
      []() constexpr {
        int idx = 0;
        int N = sizeof...(property_tys);
        // std::sort isn't constexpr until C++20. Also, it's possible there will
        // be a compiler builtin to sort types, in which case we should start
        // using that.
        std::array to_sort{
            std::pair{PropertyID<property_tys>::value, idx++}...};
        auto swap_pair = [](auto &x, auto &y) constexpr {
          auto tmp_first = x.first;
          auto tmp_second = x.second;
          x.first = y.first;
          x.second = y.second;
          y.first = tmp_first;
          y.second = tmp_second;
        };
        for (int i = 0; i < N; ++i)
          for (int j = i; j < N; ++j)
            if (to_sort[j].first < to_sort[i].first)
              swap_pair(to_sort[i], to_sort[j]);

        std::array<int, sizeof...(property_tys)> sorted_indices{};
        for (int i = 0; i < N; ++i)
          sorted_indices[i] = to_sort[i].second;

        return sorted_indices;
      }();

  template <typename> struct helper;
  template <int... IdxSeq>
  struct helper<std::integer_sequence<int, IdxSeq...>> {
    using type = properties_type_list<
        nth_type_t<sorted_indices[IdxSeq], property_tys...>...>;
  };

  using type = typename helper<
      std::make_integer_sequence<int, sizeof...(property_tys)>>::type;
};
// Specialization to avoid zero-size array creation.
template <> struct properties_sorter<> {
  using type = properties_type_list<>;
};

} // namespace detail

// Empty property list.
template <> class __SYCL_EBO properties<detail::properties_type_list<>> {
  template <typename T>
  static constexpr bool empty_properties_list_contains = false;

public:
  template <typename property_key_t> static constexpr bool has_property() {
    return false;
  }

  // Never exists for empty property list, provide this for a better error
  // message:
  template <typename T>
  static std::enable_if_t<empty_properties_list_contains<T>> get_property() {}
};

// Base implementation to provide nice user error in case of mis-use. Without it
// an error "base class '<property>' specified more than once as a direct base
// class" is reported prior to static_assert's error.
template <typename... property_tys>
class __SYCL_EBO
    properties<detail::invalid_properties_type_list<property_tys...>> {
public:
  properties(property_tys...) {
    if constexpr (!((is_property_value_v<property_tys> && ...))) {
      static_assert(((is_property_value_v<property_tys> && ...)),
                    "Non-property argument!");
    } else {
      // This is a separate specialization to report an error, we can afford
      // doing extra work to provide nice error message without sacrificing
      // compile time on non-exceptional path. Let's find *a* pair of properties
      // that failed the check. Note that there might be multiple duplicate
      // names, we're only reporting one instance. Once user addresses that, the
      // next pair will be reported.
      static constexpr auto conflict = []() constexpr {
        const std::array kinds = {detail::PropertyID<property_tys>::value...};
        auto N = kinds.size();
        for (int i = 0; i < N; ++i)
          for (int j = i + 1; j < N; ++j)
            if (kinds[i] == kinds[j])
              return std::pair{i, j};
      }();
      using first_type = detail::nth_type_t<conflict.first, property_tys...>;
      using second_type = detail::nth_type_t<conflict.second, property_tys...>;
      if constexpr (std::is_same_v<typename first_type::key_t,
                                   typename second_type::key_t>) {
        static_assert(!std::is_same_v<typename first_type::key_t,
                                      typename second_type::key_t>,
                      "Duplicate properties in property list.");
      } else {
        static_assert(
            detail::PropertyToKind<first_type>::Kind !=
                detail::PropertyToKind<second_type>::Kind,
            "Property Kind collision between different property keys!");
      }
    }
  }

  template <typename property_key_t> static constexpr bool has_property() {
    return false;
  }
};

template <typename... property_tys>
class __SYCL_EBO properties<detail::properties_type_list<property_tys...>>
    : private property_tys... {
  static_assert(detail::properties_are_sorted<property_tys...>,
                "Properties must be sorted!");
  using property_tys::get_property_impl...;

  template <typename> friend class __SYCL_EBO properties;

  template <typename prop_t> static constexpr bool is_valid_ctor_arg() {
    return ((std::is_same_v<prop_t, property_tys> || ...));
  }

  template <typename prop_t, typename... unsorted_property_tys>
  static constexpr bool can_be_constructed_from() {
    return std::is_default_constructible_v<prop_t> ||
           ((false || ... || std::is_same_v<prop_t, unsorted_property_tys>));
  }

  // It's possible it shouldn't be that complicated, but clang doesn't accept
  // simpler version: https://godbolt.org/z/oPff4h738, reported upstream at
  // https://github.com/llvm/llvm-project/issues/115547. Note that if the
  // `decltype(...)` is "inlined" then it has no issues with it, but that's too
  // verbose.
  struct helper : property_tys... {
    using property_tys::get_property_impl...;
  };
  template <typename property_key_t>
  using prop_t = decltype(std::declval<helper>().get_property_impl(
      detail::property_key_tag<property_key_t>{}));

public:
  // Definition is out-of-class so that `properties` would be complete there and
  // its interfaces could be used in `ConflictingProperties`' partial
  // specializations.
  template <
      typename... unsorted_property_tys,
      typename = std::enable_if_t<
          ((is_valid_ctor_arg<unsorted_property_tys>() && ...))>,
      typename = std::enable_if_t<
          ((can_be_constructed_from<property_tys, unsorted_property_tys...>() &&
            ...))>,
      typename = std::enable_if_t<
          detail::properties_are_unique<unsorted_property_tys...>>>
  constexpr properties(unsorted_property_tys... props);

  template <typename property_key_t> static constexpr bool has_property() {
    return std::is_base_of_v<detail::property_key_tag<property_key_t>,
                             properties>;
  }

  // Compile-time property.
  template <typename property_key_t>
  static constexpr auto
  get_property() -> std::enable_if_t<std::is_empty_v<prop_t<property_key_t>>,
                                     prop_t<property_key_t>> {
    return prop_t<property_key_t>{};
  }

  // Runtime property.
  // Extra operand to make MSVC happy as it complains otherwise:
  // https://godbolt.org/z/WGqdqrejj
  template <typename property_key_t>
  constexpr auto get_property(int = 0) const
      -> std::enable_if_t<!std::is_empty_v<prop_t<property_key_t>>,
                          prop_t<property_key_t>> {
    return get_property_impl(detail::property_key_tag<property_key_t>{});
  }
};

template <typename... property_tys>
template <typename... unsorted_property_tys, typename, typename, typename>
constexpr properties<detail::properties_type_list<property_tys...>>::properties(
    unsorted_property_tys... props)
    : unsorted_property_tys(props)... {
  static_assert(((!detail::ConflictingProperties<typename property_tys::key_t,
                                                 properties>::value &&
                  ...)),
                "Conflicting properties in property list.");
}

// Deduction guides
template <typename... unsorted_property_tys,
          typename = std::enable_if_t<
              detail::properties_are_valid_for_ctad<unsorted_property_tys...>>>
properties(unsorted_property_tys... props)
    -> properties<
        typename detail::properties_sorter<unsorted_property_tys...>::type>;

template <typename... unsorted_property_tys,
          typename = std::enable_if_t<
              !detail::properties_are_valid_for_ctad<unsorted_property_tys...>>>
properties(unsorted_property_tys... props)
    -> properties<
        detail::invalid_properties_type_list<unsorted_property_tys...>>;

using empty_properties_t = decltype(properties{});

namespace detail {

template <template <typename> typename predicate, typename... property_tys>
struct filter_properties_impl {
  static constexpr auto idx_info = []() constexpr {
    constexpr int N = sizeof...(property_tys);
    std::array<int, N> indexes{};
    int num_matched = 0;
    int idx = 0;
    (((predicate<property_tys>::value ? indexes[num_matched++] = idx++ : idx++),
      ...));

    return std::pair{indexes, num_matched};
  }();

  // Helper to convert constexpr indices values to an std::index_sequence type.
  // Values -> type is the key here.
  template <int... Idx>
  static constexpr auto idx_seq(std::integer_sequence<int, Idx...>) {
    return std::integer_sequence<int, idx_info.first[Idx]...>{};
  }

  using selected_idx_seq =
      decltype(idx_seq(std::make_integer_sequence<int, idx_info.second>{}));

  // Using prop_list_ty so that we don't need to explicitly spell out
  //  `properties` template parameters' implementation-details.
  template <typename prop_list_ty, int... Idxs>
  static constexpr auto apply_impl(const prop_list_ty &props,
                                   std::integer_sequence<int, Idxs...>) {
    return properties{props.template get_property<
        typename nth_type_t<Idxs, property_tys...>::key_t>()...};
  }

  template <typename prop_list_ty>
  static constexpr auto apply(const prop_list_ty &props) {
    return apply_impl(props, selected_idx_seq{});
  }
};

template <template <typename> typename predicate, typename... property_tys>
constexpr auto filter_properties(const properties_t<property_tys...> &props) {
  return filter_properties_impl<predicate, property_tys...>::apply(props);
}

template <typename... lhs_property_tys> struct merge_filter {
  template <typename rhs_property_ty>
  struct predicate
      : std::bool_constant<!((std::is_same_v<typename lhs_property_tys::key_t,
                                             typename rhs_property_ty::key_t> ||
                              ...))> {};
};

template <typename... lhs_property_tys, typename... rhs_property_tys>
constexpr auto merge_properties(const properties_t<lhs_property_tys...> &lhs,
                                const properties_t<rhs_property_tys...> &rhs) {
  auto rhs_unique_props =
      filter_properties<merge_filter<lhs_property_tys...>::template predicate>(
          rhs);
  if constexpr (std::is_same_v<std::decay_t<decltype(rhs)>,
                               std::decay_t<decltype(rhs_unique_props)>>) {
    // None of RHS properties share keys with LHS, no conflicts possible.
    return properties{
        lhs.template get_property<typename lhs_property_tys::key_t>()...,
        rhs.template get_property<typename rhs_property_tys::key_t>()...};
  } else {
    // Ensure no conflicts, then merge.
    constexpr auto has_conflict = [](auto *lhs_prop) constexpr {
      using lhs_property_ty = std::remove_pointer_t<decltype(lhs_prop)>;
      return (((std::is_same_v<typename lhs_property_ty::key_t,
                               typename rhs_property_tys::key_t> &&
                (!std::is_same_v<lhs_property_ty, rhs_property_tys> ||
                 !std::is_empty_v<lhs_property_ty>)) ||
               ...));
    };
    static_assert(
        !((has_conflict(static_cast<lhs_property_tys *>(nullptr)) || ...)),
        "Failed to merge property lists due to conflicting properties.");
    return merge_properties(lhs, rhs_unique_props);
  }
}

template <typename LHSPropertiesT, typename RHSPropertiesT>
using merged_properties_t = decltype(merge_properties(
    std::declval<LHSPropertiesT>(), std::declval<RHSPropertiesT>()));

template <typename property_key_t, typename prop_list_t, typename default_t>
constexpr auto get_property_or(default_t value, const prop_list_t &props) {
  if constexpr (prop_list_t::template has_property<property_key_t>())
    return props.template get_property<property_key_t>();
  else
    return value;
}
template <typename property_key_t, typename prop_list_t, typename default_t>
constexpr auto get_property_or(default_t value) {
  if constexpr (prop_list_t::template has_property<property_key_t>())
    return prop_list_t::template get_property<property_key_t>();
  else
    return value;
}

template <typename SyclT, typename PropListT>
struct all_are_properties_of : std::false_type /* not a properties list */ {};
template <typename SyclT, typename... Props>
struct all_are_properties_of<SyclT, properties_t<Props...>>
    : std::bool_constant<((is_property_value_of<Props, SyclT>::value && ...))> {
};
template <typename SyclT, typename PropListT>
inline constexpr bool all_are_properties_of_v =
    all_are_properties_of<SyclT, PropListT>::value;

} // namespace detail
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
