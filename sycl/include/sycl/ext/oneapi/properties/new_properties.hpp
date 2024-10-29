//==-------- new_properties.hpp --- SYCL extended property list ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <array>
#include <string_view>
#include <type_traits>
#include <utility>

#include <sycl/detail/defines_elementary.hpp>
#include <sycl/detail/type_traits.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {
namespace new_properties {
namespace detail {
template <typename... property_tys> struct properties_type_list;

// Is used to implement `is_property_v`.
struct property_key_tag_base {};
} // namespace detail

template <typename properties_type_list_ty, typename = void>
class __SYCL_EBO properties;

template <typename T> struct is_property_list : std::false_type {};
template <typename PropListTy>
struct is_property_list<properties<PropListTy>> : std::true_type {};
template <typename T>
inline constexpr bool is_property_list_v = is_property_list<T>::value;

template <typename T>
inline constexpr bool is_property_v =
    std::is_base_of_v<detail::property_key_tag_base, T> &&
    !is_property_list_v<T>;

namespace detail {

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

template <typename IdxSeq, typename... property_tys> struct properties_sorter;

// Specialization to avoid zero-size array creation.
template <> struct properties_sorter<std::integer_sequence<int>> {
  using type = properties_type_list<>;
};

template <int... IdxSeq, typename... property_tys>
struct properties_sorter<std::integer_sequence<int, IdxSeq...>,
                         property_tys...> {
  static constexpr auto sorted_indices = []() constexpr {
    int idx = 0;
    int N = sizeof...(property_tys);
    // std::sort isn't constexpr until C++20. Also, it's possible there will be
    // a compiler builtin to sort types, in which case we should start using it.
    std::array to_sort{std::pair{property_tys::property_name, idx++}...};
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

  using type = properties_type_list<
      nth_type_t<sorted_indices[IdxSeq], property_tys...>...>;
};

// Need "universal template"
// (https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p2989r2.pdf) to
// simplify this to a single template and support mixed type/non-type template
// parameters in properties.
template <typename> struct property_key_non_template : property_key_tag_base {};
template <template <auto...> typename>
struct property_key_value_template : property_key_tag_base {};
template <template <typename...> typename>
struct property_key_type_template : property_key_tag_base {};

// Helpers to provide "uniform" access to the key.
template <typename property>
constexpr auto key() {
  return property_key_non_template<property>{};
}
template <template <auto...> typename property>
constexpr auto key() {
  return property_key_value_template<property>{};
}
template <template <typename...> typename property>
constexpr auto key() {
  return property_key_type_template<property>{};
}

// NOTE: each property_t subclass must provide
//
// static constexpr std::string_view
// property_name{"<fully-qualified-name-of-the-property"}
//
// member that will be used to sort the properties in the property list. That
// way we can have a stable sorting order between device/custom-host compilers
// for both properties provided by the implementation and the ones introduced by
// users' applications/libraries.
template <typename property_t,
          typename property_key_t = property_key_non_template<property_t>>
struct property_base : property_key_t {
  using key_t = property_key_t;

protected:
  constexpr property_t get_property_impl(key_t = {}) const {
    return *static_cast<const property_t *>(this);
  }

  // For key_t access in error reporting specialization.
  template <typename, typename>
  friend class __SYCL_EBO new_properties::properties;

public:
  static constexpr const char *ir_attribute_name = "";
  static constexpr std::nullptr_t ir_attribute_value = nullptr;

  // this_property_t is to disable ADL - properties{property{}} is inherited
  // from property.

  template <typename this_property_t, typename other_property_t>
  friend constexpr std::enable_if_t<
      std::is_same_v<this_property_t, property_t> &&
          is_property_v<other_property_t>,
      decltype(properties{std::declval<const this_property_t>(),
                          std::declval<const other_property_t>()})>
  operator+(const this_property_t &lhs, const other_property_t &rhs) {
    return properties{lhs, rhs};
  }
};

template <typename... property_tys>
inline constexpr bool property_names_are_unique = []() constexpr {
  if constexpr (sizeof...(property_tys) == 0) {
    return true;
  } else {
    const std::array names = {property_tys::property_name...};
    auto N = names.size();
    for (int i = 0; i < N; ++i)
      for (int j = i + 1; j < N; ++j)
        if (names[i] == names[j])
          return false;

    return true;
  }
}();

template <typename... property_tys>
inline constexpr bool properties_are_sorted = []() constexpr {
  if constexpr (sizeof...(property_tys) == 0) {
    return true;
  } else {
    const std::array sort_names = {property_tys::property_name...};
    // std::is_sorted isn't constexpr until C++20.
    //
    // Sorting is an implementation detail while uniqueness of the
    // property_name's is an API restriction. This internal check actually
    // combines both conditions as we expect that user error is handled before
    // the internal `properties_are_sorted` assert is checked.
    for (std::size_t idx = 1; idx < sort_names.size(); ++idx)
      if (sort_names[idx - 1] >= sort_names[idx])
        return false;
    return true;
  }
}();

} // namespace detail

// Empty property list.
template <> class __SYCL_EBO properties<detail::properties_type_list<>, void> {
public:
  template <typename> static constexpr bool has_property() { return false; }
  template <template <auto...> typename> static constexpr bool has_property() {
    return false;
  }

  // TODO: How does this work without qualified name?
  template <typename other_property_t>
  friend constexpr std::enable_if_t<
      is_property_v<other_property_t>,
      properties<detail::properties_type_list<other_property_t>>>
  operator+(const properties &, const other_property_t &rhs) {
    return properties{rhs};
  }
};

// Base implementation to provide nice user error in case of mis-use. Without it
// an error "base class '<property>' specified more than once as a direct base
// class" is reported prior to static_assert's error.
template <typename... property_tys>
class __SYCL_EBO properties<
    detail::properties_type_list<property_tys...>,
    std::enable_if_t<!detail::property_names_are_unique<property_tys...>>> {
  static_assert((is_property_v<property_tys> && ...));

  // This is a separate specialization to report an error, we can afford doing
  // extra work to provide nice error message without sacrificing compile time
  // on non-exceptional path. Let's find *a* pair of properties that failed the
  // check. Note that there might be multiple duplicate names, we're only
  // reporting one instance. Once user addresses that, the next pair will be
  // reported.
  static constexpr auto conflict = []() constexpr {
    const std::array keys = {property_tys::property_name...};
    auto N = keys.size();
    for (int i = 0; i < N; ++i)
      for (int j = i + 1; j < N; ++j)
        if (keys[i] == keys[j])
          return std::pair{i, j};
  }();
  using first_type = detail::nth_type_t<conflict.first, property_tys...>;
  using second_type = detail::nth_type_t<conflict.second, property_tys...>;
  static_assert(
      !std::is_same_v<typename first_type::key_t, typename second_type::key_t>,
      "Duplicate property!");
  static_assert(first_type::property_name != second_type::property_name,
                "Property name collision between different property keys!");
  static_assert((is_property_v<property_tys> && ...));
};

// NOTE: Meta-function to implement CTAD rules isn't allowed to return
// `properties<something>` and it's impossible to return a pack as well. As
// such, we're forced to have an extra level of `detail::properties_type_list`
// for the purpose of providing CTAD rules.
template <typename... property_tys>
class __SYCL_EBO properties<
    detail::properties_type_list<property_tys...>,
    std::enable_if_t<detail::property_names_are_unique<property_tys...>>>
    : private property_tys... {
  static_assert((is_property_v<property_tys> && ...));
  static_assert(detail::properties_are_sorted<property_tys...>,
                "Properties must be sorted!");
  using property_tys::get_property_impl...;

  template <typename, typename> friend class __SYCL_EBO properties;

public:
  template <typename... unsorted_property_tys,
            typename = std::enable_if_t<
                ((is_property_v<unsorted_property_tys> && ...)) &&
                sizeof...(unsorted_property_tys) == sizeof...(property_tys)>>
  constexpr properties(unsorted_property_tys... props)
      : unsorted_property_tys(props)... {}

  // Until we have "universal template", `has_property`/`get_property`
  // implementations have to be duplicated for
  // non-templated/value-templated/type-templated cases.

  template <typename property> static constexpr bool has_property() {
    return std::is_base_of_v<decltype(detail::key<property>()), properties>;
  }

  template <template <auto...> typename property>
  static constexpr bool has_property() {
    return std::is_base_of_v<decltype(detail::key<property>()), properties>;
  }

  template <template <typename...> typename property>
  static constexpr bool has_property() {
    return std::is_base_of_v<decltype(detail::key<property>()), properties>;
  }

  // In addition to the duplication mentioned above, do SFINAE-based dispatch
  // between compile/run-time properties to enable "static" access when
  // possible.
  template <typename property>
  static constexpr auto get_property() -> std::enable_if_t<
      std::is_empty_v<decltype(std::declval<properties>().get_property_impl(
          detail::key<property>()))>,
      decltype(std::declval<properties>().get_property_impl(
          detail::key<property>()))> {
    return decltype(std::declval<properties>().get_property_impl(
        detail::key<property>())){};
  }

  template <typename property>
  constexpr auto get_property() const -> std::enable_if_t<
      !std::is_empty_v<decltype(std::declval<properties>().get_property_impl(
          detail::key<property>()))>,
      decltype(std::declval<properties>().get_property_impl(
          detail::key<property>()))> {
    return get_property_impl(detail::key<property>());
  }

  template <template <auto...> typename property>
  static constexpr auto get_property() -> std::enable_if_t<
      std::is_empty_v<decltype(std::declval<properties>().get_property_impl(
          detail::key<property>()))>,
      decltype(std::declval<properties>().get_property_impl(
          detail::key<property>()))> {
    return decltype(std::declval<properties>().get_property_impl(
        detail::key<property>())){};
  }

  template <template <auto...> typename property>
  constexpr auto get_property() const -> std::enable_if_t<
      !std::is_empty_v<decltype(std::declval<properties>().get_property_impl(
          detail::key<property>()))>,
      decltype(std::declval<properties>().get_property_impl(
          detail::key<property>()))> {
    return get_property_impl(detail::key<property>());
  }

  template <template <typename...> typename property>
  static constexpr auto get_property() -> std::enable_if_t<
      std::is_empty_v<decltype(std::declval<properties>().get_property_impl(
          detail::key<property>()))>,
      decltype(std::declval<properties>().get_property_impl(
          detail::key<property>()))> {
    return decltype(std::declval<properties>().get_property_impl(
        detail::key<property>())){};
  }

  template <template <typename...> typename property>
  constexpr auto get_property() const -> std::enable_if_t<
      !std::is_empty_v<decltype(std::declval<properties>().get_property_impl(
          detail::key<property>()))>,
      decltype(std::declval<properties>().get_property_impl(
          detail::key<property>()))> {
    return get_property_impl(detail::key<property>());
  }

  // TODO: Use more effective insert sort for single-property insertion.

  // Need to use qualified type to force CTAD instead of using *current*
  // properties instantiation.
  template <typename other_property_t>
  friend constexpr std::enable_if_t<
      is_property_v<other_property_t>,
      decltype(ext::oneapi::experimental::new_properties::properties{
          std::declval<property_tys>()..., std::declval<other_property_t>()})>
  operator+(const properties &lhs, const other_property_t &rhs) {
    return ext::oneapi::experimental::new_properties::properties{
        static_cast<const property_tys &>(lhs)..., rhs};
  }

  template <typename... other_property_tys>
  friend constexpr auto
  operator+(const properties &lhs,
            const ext::oneapi::experimental::new_properties::properties<
                detail::properties_type_list<other_property_tys...>> &rhs) {
    return ext::oneapi::experimental::new_properties::properties{
        static_cast<const property_tys &>(lhs)...,
        static_cast<const other_property_tys &>(rhs)...};
  }
};

template <typename... unsorted_property_tys,
          typename =
              std::enable_if_t<((is_property_v<unsorted_property_tys> && ...))>>
properties(unsorted_property_tys...)
    -> properties<typename detail::properties_sorter<
        std::make_integer_sequence<int, sizeof...(unsorted_property_tys)>,
        unsorted_property_tys...>::type>;

using empty_properties_t = decltype(properties{});
} // namespace new_properties
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
