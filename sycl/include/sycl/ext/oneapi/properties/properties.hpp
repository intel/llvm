//==---------- properties.hpp --- SYCL extended property list --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <array>
#include <type_traits>
#include <utility>
#include <string_view>

#include <sycl/detail/defines_elementary.hpp>


// For old properties:
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
namespace new_properties {

template <typename properties_type_list_ty, typename = void>
class __SYCL_EBO properties;

namespace detail {
template <typename... property_tys> struct properties_type_list;

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

// Is used to implement `is_property_v`.
struct property_key_tag_base {};

// We support incomplete property_key_t, so need to wrap it.
template <typename property_key_t>
struct property_key_tag : property_key_tag_base {};

template <typename property_t, typename property_key_t = property_t>
struct property_base : property_key_tag<property_key_t> {
protected:
  using key_t = property_key_t;
  constexpr property_t get_property(property_key_tag<key_t>) const {
    // In fact, `static_cast` below works just fine with clang/msvc but not with
    // gcc, see https://godbolt.org/z/MY6849jGh for a reduced test. However, we
    // need to support all ,so special case for compile-time properties (when
    // `is_empty_v` is true).
    if constexpr (std::is_empty_v<property_t>) {
      return property_t{};
    } else {
      return *static_cast<const property_t *>(this);
    }
  }

  // For key_t access in error reporting specialization.
  template <typename, typename>
  friend class __SYCL_EBO new_properties::properties;
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

template <typename T> struct is_property_list : std::false_type {};
template <typename PropListTy>
struct is_property_list<properties<PropListTy>> : std::true_type {};
template <typename T>
inline constexpr bool is_property_list_v = is_property_list<T>::value;

template <typename T>
inline constexpr bool is_property_v =
    std::is_base_of_v<detail::property_key_tag_base, T> &&
    !is_property_list_v<T>;

// Empty property list.
template <> class __SYCL_EBO properties<detail::properties_type_list<>, void> {
  template <typename> static constexpr bool has_property() { return false; }
};

// Base implementation to provide nice user error in case of mis-use. Without it
// an error "base class '<property>' specified more than once as a direct base
// class" is reported prior to static_assert's error.
template <typename... property_tys>
class __SYCL_EBO properties<
    detail::properties_type_list<property_tys...>,
    std::enable_if_t<!detail::property_names_are_unique<property_tys...>>> {

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

template <typename... property_tys>
class __SYCL_EBO
    properties<detail::properties_type_list<property_tys...>,
               std::enable_if_t<detail::properties_are_sorted<property_tys...>>>
    : public property_tys... {
  static_assert((is_property_v<property_tys> && ...));
  static_assert(
      detail::properties_are_sorted<property_tys...>,
      "Properties must be sorted!");
  using property_tys::get_property...;

public:
  template <typename... unsorted_property_tys,
            typename = std::enable_if_t<
                ((is_property_v<unsorted_property_tys> && ...))>>
  constexpr properties(unsorted_property_tys... props)
      : unsorted_property_tys(props)... {}

  template <
      typename... other_property_list_tys, typename... other_property_tys,
      typename = std::enable_if_t<((is_property_v<other_property_tys> && ...))>>
  constexpr properties(
      properties<detail::properties_type_list<other_property_list_tys...>>
          other_properties,
      other_property_tys... props)
      : other_property_list_tys(
            static_cast<other_property_list_tys &>(other_properties))...,
        other_property_tys(props)... {}

  // TODO: Do we need this? If so, is separate CTAD needed?
  // template <typename... unsorted_property_tys>
  // properties(unsorted_property_tys &&...props)
  //     : unsorted_property_tys(std::forward<unsorted_property_tys>(props))...
  //     {}

  template <typename property_key_t> static constexpr bool has_property() {
    return std::is_base_of_v<detail::property_key_tag<property_key_t>,
                             properties>;
  }

  // Two methods below do the following (pseudocode):
  //
  // template <property_key_t>
  // using ret_t = decltype(this->get_property(key_tag<property_key_t>{}));
  // static constexpr auto get_property() requires(is_empty_v<ret_t>) {
  //   return ret_t{};
  // }
  // constexpr auto get_property() requires(!is_empty_v<ret_t>) {
  //   return get_property(key_tag<property_key_t>{});
  // }
  template <typename property_key_t>
  static constexpr auto get_property() -> std::enable_if_t<
      std::is_empty_v<decltype(std::declval<properties>().get_property(
          detail::property_key_tag<property_key_t>{}))>,
      decltype(std::declval<properties>().get_property(
          detail::property_key_tag<property_key_t>{}))> {
    return decltype(std::declval<properties>().get_property(
        detail::property_key_tag<property_key_t>{})){};
  }

  template <typename property_key_t>
  constexpr auto get_property() const -> std::enable_if_t<
      !std::is_empty_v<decltype(std::declval<properties>().get_property(
          detail::property_key_tag<property_key_t>{}))>,
      decltype(std::declval<properties>().get_property(
          detail::property_key_tag<property_key_t>{}))> {
    return get_property(detail::property_key_tag<property_key_t>{});
  }

  template <typename property_key_t, typename default_property_t>
  constexpr auto
  get_property_or_default_to(default_property_t default_property) {
    if constexpr (has_property<property_key_t>())
      return get_property<property_key_t>();
    else
      return default_property;
  }
};

template <typename... unsorted_property_tys,
          typename =
              std::enable_if_t<((is_property_v<unsorted_property_tys> && ...))>>
properties(unsorted_property_tys...)
    -> properties<typename detail::properties_sorter<
        std::make_integer_sequence<int, sizeof...(unsorted_property_tys)>,
        unsorted_property_tys...>::type>;

template <
    typename... other_property_list_tys, typename... other_property_tys,
    typename = std::enable_if_t<((is_property_v<other_property_tys> && ...))>>
properties(properties<detail::properties_type_list<other_property_list_tys...>>,
           other_property_tys...)
    -> properties<typename detail::properties_sorter<
        std::make_integer_sequence<int, sizeof...(other_property_list_tys) +
                                            sizeof...(other_property_tys)>,
        other_property_list_tys..., other_property_tys...>::type>;

using empty_properties_t = decltype(properties{});

template <typename, typename> struct is_property_key_of : std::false_type {};
} // namespace new_properties
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

namespace detail {

// Checks if a tuple of properties contains a property.
template <typename PropT, typename PropertiesT>
struct ContainsProperty : std::false_type {};
template <typename PropT, typename T, typename... Ts>
struct ContainsProperty<PropT, std::tuple<T, Ts...>>
    : ContainsProperty<PropT, std::tuple<Ts...>> {};
template <typename PropT, typename... Rest>
struct ContainsProperty<PropT, std::tuple<PropT, Rest...>> : std::true_type {};
template <typename PropT, typename... PropValuesTs, typename... Rest>
struct ContainsProperty<
    PropT, std::tuple<property_value<PropT, PropValuesTs...>, Rest...>>
    : std::true_type {};

// Finds the full property_value type of a property in a tuple of properties.
// type is void if the type was not found in the tuple of properties.
template <typename CTPropertyT, typename PropertiesT = void>
struct FindCompileTimePropertyValueType {
  using type = void;
};
template <typename CTPropertyT, typename OtherProperty, typename... Rest>
struct FindCompileTimePropertyValueType<CTPropertyT,
                                        std::tuple<OtherProperty, Rest...>> {
  using type =
      typename FindCompileTimePropertyValueType<CTPropertyT,
                                                std::tuple<Rest...>>::type;
};
template <typename CTPropertyT, typename... CTPropertyValueTs, typename... Rest>
struct FindCompileTimePropertyValueType<
    CTPropertyT,
    std::tuple<property_value<CTPropertyT, CTPropertyValueTs...>, Rest...>> {
  using type = property_value<CTPropertyT, CTPropertyValueTs...>;
};

template <typename CTPropertyT, bool HasProperty, typename PropertiesT = void>
static constexpr std::enable_if_t<
    HasProperty,
    typename FindCompileTimePropertyValueType<CTPropertyT, PropertiesT>::type>
get_property() {
  return {};
}

template <typename CTPropertyT, bool HasProperty, typename PropertiesT = void>
static constexpr std::enable_if_t<!HasProperty, void> get_property() {
  return;
}

// Filters for all runtime properties with data in a tuple of properties.
// NOTE: We only need storage for runtime properties with data.
template <typename T> struct RuntimePropertyStorage {};
template <typename... Ts> struct RuntimePropertyStorage<std::tuple<Ts...>> {
  using type = std::tuple<>;
};
template <typename T, typename... Ts>
struct RuntimePropertyStorage<std::tuple<T, Ts...>>
    : std::conditional_t<IsRuntimeProperty<T>::value,
                         PrependTuple<T, typename RuntimePropertyStorage<
                                             std::tuple<Ts...>>::type>,
                         RuntimePropertyStorage<std::tuple<Ts...>>> {};

// Count occurrences of a type in a tuple.
template <typename T, typename Tuple> struct CountTypeInTuple;
template <typename T, typename... TupleTs>
struct CountTypeInTuple<T, std::tuple<TupleTs...>>
    : std::integral_constant<
          size_t, (0 + ... + static_cast<size_t>(std::is_same_v<T, TupleTs>))> {
};

// Helper for counting the number of properties that are also in PropertyArgsT.
template <typename PropertyArgsT, typename Props> struct CountContainedProps;
template <typename PropertyArgsT>
struct CountContainedProps<PropertyArgsT, std::tuple<>>
    : std::integral_constant<size_t, 0> {};
template <typename PropertyArgsT, typename PropertyT, typename... PropertyTs>
struct CountContainedProps<PropertyArgsT,
                           std::tuple<PropertyT, PropertyTs...>> {
  static constexpr size_t NumOccurrences =
      CountTypeInTuple<PropertyT, PropertyArgsT>::value;
  static_assert(NumOccurrences <= 1,
                "Duplicate occurrences of property in constructor arguments.");
  static constexpr size_t value =
      CountContainedProps<PropertyArgsT, std::tuple<PropertyTs...>>::value +
      NumOccurrences;
};

// Helper class to extract a subset of elements from a tuple.
// NOTES: This assumes no duplicate properties and that all properties in the
//        struct template argument appear in the tuple passed to Extract.
template <typename PropertyArgsT, typename PropertiesT>
struct ExtractProperties;
template <typename PropertyArgsT>
struct ExtractProperties<PropertyArgsT, std::tuple<>> {
  static constexpr std::tuple<> Extract(const PropertyArgsT &) {
    return std::tuple<>{};
  }
};
template <typename PropertyArgsT, typename PropertyT, typename... PropertiesTs>
struct ExtractProperties<PropertyArgsT,
                         std::tuple<PropertyT, PropertiesTs...>> {
  static constexpr std::tuple<PropertyT, PropertiesTs...>
  Extract(const PropertyArgsT &PropertyValues) {
    // TODO: NumOccurrences and checks should be moved out of the function once
    //       https://github.com/intel/llvm/issues/13677 has been fixed.
    constexpr size_t NumOccurrences =
        CountTypeInTuple<PropertyT, PropertyArgsT>::value;
    static_assert(
        NumOccurrences <= 1,
        "Duplicate occurrences of property in constructor arguments.");
    static_assert(NumOccurrences == 1 ||
                      std::is_default_constructible_v<PropertyT>,
                  "Each property in the property list must either be given an "
                  "argument in the constructor or be default-constructible.");

    auto NextExtractedProperties =
        ExtractProperties<PropertyArgsT, std::tuple<PropertiesTs...>>::Extract(
            PropertyValues);

    if constexpr (NumOccurrences == 1) {
      return std::tuple_cat(
          std::tuple<PropertyT>{std::get<PropertyT>(PropertyValues)},
          NextExtractedProperties);
    } else {
      return std::tuple_cat(std::tuple<PropertyT>{PropertyT{}},
                            NextExtractedProperties);
    }
  }
};

// Get the value of a property from a property list
template <typename PropKey, typename ConstType, typename DefaultPropVal,
          typename PropertiesT>
struct GetPropertyValueFromPropList {};

template <typename PropKey, typename ConstType, typename DefaultPropVal,
          typename... PropertiesT>
struct GetPropertyValueFromPropList<PropKey, ConstType, DefaultPropVal,
                                    std::tuple<PropertiesT...>> {
  using prop_val_t = std::conditional_t<
      ContainsProperty<PropKey, std::tuple<PropertiesT...>>::value,
      typename FindCompileTimePropertyValueType<
          PropKey, std::tuple<PropertiesT...>>::type,
      DefaultPropVal>;
  static constexpr ConstType value =
      PropertyMetaInfo<std::remove_const_t<prop_val_t>>::value;
};

} // namespace detail

template <typename PropertiesT> class properties {
  static_assert(detail::IsTuple<PropertiesT>::value,
                "Properties must be in a tuple.");
  static_assert(detail::AllPropertyValues<PropertiesT>::value,
                "Unrecognized property in property list.");
  static_assert(detail::IsSorted<PropertiesT>::value,
                "Properties in property list are not sorted.");
  static_assert(detail::SortedAllUnique<PropertiesT>::value,
                "Duplicate properties in property list.");
  static_assert(detail::NoConflictingProperties<PropertiesT>::value,
                "Conflicting properties in property list.");

public:
  template <typename... PropertyValueTs,
            std::enable_if_t<detail::AllPropertyValues<
                                 std::tuple<PropertyValueTs...>>::value,
                             int> = 0>
  constexpr properties(PropertyValueTs... props)
      : Storage(detail::ExtractProperties<std::tuple<PropertyValueTs...>,
                                          StorageT>::Extract({props...})) {
    // Default-constructible properties do not need to be in the arguments.
    // For properties with a storage, default-constructibility is checked in
    // ExtractProperties, while those without are so by default. As such, all
    // arguments must be a unique property type and must be in PropertiesT.
    constexpr size_t NumContainedProps =
        detail::CountContainedProps<std::tuple<PropertyValueTs...>,
                                    PropertiesT>::value;
    static_assert(NumContainedProps == sizeof...(PropertyValueTs),
                  "One or more property argument is not a property in the "
                  "property list.");
  }

  template <typename PropertyT>
  static constexpr std::enable_if_t<detail::IsProperty<PropertyT>::value, bool>
  has_property() {
    return detail::ContainsProperty<PropertyT, PropertiesT>::value;
  }

  template <typename PropertyT>
  typename std::enable_if_t<detail::IsRuntimeProperty<PropertyT>::value &&
                                has_property<PropertyT>(),
                            PropertyT>
  get_property() const {
    return std::get<PropertyT>(Storage);
  }

  template <typename PropertyT>
  typename std::enable_if_t<detail::IsRuntimeProperty<PropertyT>::value &&
                                !has_property<PropertyT>(),
                            void>
  get_property() const {
    static_assert(has_property<PropertyT>(),
                  "Property list does not contain the requested property.");
    return;
  }

  template <typename PropertyT>
  static constexpr auto get_property(
      typename std::enable_if_t<detail::IsCompileTimeProperty<PropertyT>::value>
          * = 0) {
    static_assert(has_property<PropertyT>(),
                  "Property list does not contain the requested property.");
    return detail::get_property<PropertyT, has_property<PropertyT>(),
                                PropertiesT>();
  }

private:
  using StorageT = typename detail::RuntimePropertyStorage<PropertiesT>::type;

  StorageT Storage;
};

#ifdef __cpp_deduction_guides
// Deduction guides
template <typename... PropertyValueTs>
properties(PropertyValueTs... props)
    -> properties<typename detail::Sorted<PropertyValueTs...>::type>;
#endif

using empty_properties_t = properties<std::tuple<>>;

// Property list traits
template <typename propertiesT> struct is_property_list : std::false_type {};
template <typename... PropertyValueTs>
struct is_property_list<properties<std::tuple<PropertyValueTs...>>>
    : std::is_same<
          properties<std::tuple<PropertyValueTs...>>,
          properties<typename detail::Sorted<PropertyValueTs...>::type>> {};

#if __cplusplus > 201402L
template <typename propertiesT>
inline constexpr bool is_property_list_v = is_property_list<propertiesT>::value;
#endif

namespace detail {

// Helper for reconstructing a properties type. This assumes that
// PropertyValueTs is sorted and contains only valid properties.
template <typename... PropertyValueTs>
using properties_t = properties<std::tuple<PropertyValueTs...>>;

// Helper for merging two property lists;
template <typename LHSPropertiesT, typename RHSPropertiesT>
struct merged_properties;
template <typename... LHSPropertiesTs, typename... RHSPropertiesTs>
struct merged_properties<properties_t<LHSPropertiesTs...>,
                         properties_t<RHSPropertiesTs...>> {
  using type = properties<typename MergeProperties<
      std::tuple<LHSPropertiesTs...>, std::tuple<RHSPropertiesTs...>>::type>;
};
template <typename LHSPropertiesT, typename RHSPropertiesT>
using merged_properties_t =
    typename merged_properties<LHSPropertiesT, RHSPropertiesT>::type;

template <typename Properties, typename PropertyKey, typename Cond = void>
struct ValueOrDefault {
  template <typename ValT> static constexpr ValT get(ValT Default) {
    return Default;
  }
};

template <typename Properties, typename PropertyKey>
struct ValueOrDefault<
    Properties, PropertyKey,
    std::enable_if_t<is_property_list_v<Properties> &&
                     Properties::template has_property<PropertyKey>()>> {
  template <typename ValT> static constexpr ValT get(ValT) {
    return Properties::template get_property<PropertyKey>().value;
  }
};

// helper: check_all_props_are_keys_of
template <typename SyclT> constexpr bool check_all_props_are_keys_of() {
  return true;
}

template <typename SyclT, typename FirstProp, typename... RestProps>
constexpr bool check_all_props_are_keys_of() {
  return ext::oneapi::experimental::is_property_key_of<FirstProp,
                                                       SyclT>::value &&
         check_all_props_are_keys_of<SyclT, RestProps...>();
}

// all_props_are_keys_of
template <typename SyclT, typename PropertiesT>
struct all_props_are_keys_of : std::false_type {};

template <typename SyclT>
struct all_props_are_keys_of<SyclT,
                             ext::oneapi::experimental::empty_properties_t>
    : std::true_type {};

template <typename SyclT, typename PropT>
struct all_props_are_keys_of<
    SyclT, ext::oneapi::experimental::detail::properties_t<PropT>>
    : std::bool_constant<
          ext::oneapi::experimental::is_property_key_of<PropT, SyclT>::value> {
};

template <typename SyclT, typename... Props>
struct all_props_are_keys_of<
    SyclT, ext::oneapi::experimental::detail::properties_t<Props...>>
    : std::bool_constant<check_all_props_are_keys_of<SyclT, Props...>()> {};

} // namespace detail
} // namespace ext::oneapi::experimental

template <typename PropertiesT>
struct is_device_copyable<ext::oneapi::experimental::properties<PropertiesT>>
    : is_device_copyable<PropertiesT> {};
} // namespace _V1
} // namespace sycl
