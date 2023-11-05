//==----------- annotated_ptr.hpp - SYCL annotated_ptr extension -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines.hpp>
#include <sycl/ext/intel/experimental/fpga_annotated_properties.hpp>
#include <sycl/ext/oneapi/experimental/annotated_ptr/annotated_ptr_properties.hpp>
#include <sycl/ext/oneapi/experimental/common_annotated_properties/properties.hpp>
#include <sycl/ext/oneapi/properties/properties.hpp>
#include <sycl/ext/oneapi/properties/property.hpp>
#include <sycl/ext/oneapi/properties/property_value.hpp>

#include <cstddef>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

namespace sycl {
inline namespace _V1 {
namespace ext {
namespace oneapi {
namespace experimental {

namespace {
#define PROPAGATE_OP(op)                                                       \
  T operator op##=(const T &rhs) const { return *this = *this op rhs; }

// compare strings on compile time
constexpr bool compareStrs(const char *Str1, const char *Str2) {
  return std::string_view(Str1) == Str2;
}

// filter properties with AllowedPropsTuple via name checking
template <typename TestProps, typename AllowedPropsTuple>
struct PropertiesAreAllowed {};

template <typename TestProps, typename... AllowedProps>
struct PropertiesAreAllowed<TestProps, std::tuple<const AllowedProps...>> {
  static constexpr const bool allowed =
      (compareStrs(detail::PropertyMetaInfo<TestProps>::name,
                   detail::PropertyMetaInfo<AllowedProps>::name) ||
       ...);
};

template <typename... Ts>
using tuple_cat_t = decltype(std::tuple_cat(std::declval<Ts>()...));

template <typename AllowedPropTuple, typename... Props>
struct PropertiesFilter {
  using tuple = tuple_cat_t<typename std::conditional<
      PropertiesAreAllowed<Props, AllowedPropTuple>::allowed, std::tuple<Props>,
      std::tuple<>>::type...>;
};
} // namespace
template <typename T, typename PropertyListT = empty_properties_t>
class annotated_ref {
  // This should always fail when instantiating the unspecialized version.
  static constexpr bool is_valid_property_list =
      is_property_list<PropertyListT>::value;
  static_assert(is_valid_property_list, "Property list is invalid.");
};

template <typename T, typename... Props>
class annotated_ref<T, detail::properties_t<Props...>> {
  using property_list_t = detail::properties_t<Props...>;

private:
  T *m_Ptr;
  annotated_ref(T *Ptr) : m_Ptr(Ptr) {}

public:
  annotated_ref(const annotated_ref &) = delete;

  operator T() const {
#ifdef __SYCL_DEVICE_ONLY__
    return *__builtin_intel_sycl_ptr_annotation(
        m_Ptr, detail::PropertyMetaInfo<Props>::name...,
        detail::PropertyMetaInfo<Props>::value...);
#else
    return *m_Ptr;
#endif
  }

  T operator=(const T &Obj) const {
#ifdef __SYCL_DEVICE_ONLY__
    *__builtin_intel_sycl_ptr_annotation(
        m_Ptr, detail::PropertyMetaInfo<Props>::name...,
        detail::PropertyMetaInfo<Props>::value...) = Obj;
#else
    *m_Ptr = Obj;
#endif
    return Obj;
  }

  T operator=(const annotated_ref &Ref) const { return *this = T(Ref); }

  PROPAGATE_OP(+)
  PROPAGATE_OP(-)
  PROPAGATE_OP(*)
  PROPAGATE_OP(/)
  PROPAGATE_OP(%)
  PROPAGATE_OP(^)
  PROPAGATE_OP(&)
  PROPAGATE_OP(|)
  PROPAGATE_OP(<<)
  PROPAGATE_OP(>>)

  T operator++() { return *this += 1; }

  T operator++(int) {
    const T t = *this;
    *this = (t + 1);
    return t;
  }

  T operator--() { return *this -= 1; }

  T operator--(int) {
    const T t = *this;
    *this = (t - 1);
    return t;
  }

  template <class T2, class P2> friend class annotated_ptr;
};

#undef PROPAGATE_OP

#ifdef __cpp_deduction_guides
template <typename T, typename... Args>
annotated_ptr(T *, Args...)
    -> annotated_ptr<T, typename detail::DeducedProperties<Args...>::type>;

template <typename T, typename old, typename... ArgT>
annotated_ptr(annotated_ptr<T, old>, properties<std::tuple<ArgT...>>)
    -> annotated_ptr<
        T, detail::merged_properties_t<old, detail::properties_t<ArgT...>>>;
#endif // __cpp_deduction_guides

template <typename T, typename PropertyListT = empty_properties_t>
class annotated_ptr {
  // This should always fail when instantiating the unspecialized version.
  static constexpr bool is_valid_property_list =
      is_property_list<PropertyListT>::value;
  static_assert(is_valid_property_list, "Property list is invalid.");
};

template <typename T, typename... Props>
class __SYCL_SPECIAL_CLASS
__SYCL_TYPE(annotated_ptr) annotated_ptr<T, detail::properties_t<Props...>> {
  using property_list_t = detail::properties_t<Props...>;

  // buffer_location and alignment are allowed for annotated_ref
  using allowed_properties =
      std::tuple<decltype(ext::intel::experimental::buffer_location<0>),
                 decltype(ext::oneapi::experimental::alignment<0>)>;
  using filtered_properties =
      typename PropertiesFilter<allowed_properties, Props...>::tuple;

  // template unpack helper
  template <typename... FilteredProps> struct unpack {};

  template <typename... FilteredProps>
  struct unpack<std::tuple<FilteredProps...>> {
    using type = detail::properties_t<FilteredProps...>;
  };

  using reference = sycl::ext::oneapi::experimental::annotated_ref<
      T, typename unpack<filtered_properties>::type>;

#ifdef __SYCL_DEVICE_ONLY__
  using global_pointer_t = typename decorated_global_ptr<T>::pointer;
#else
  using global_pointer_t = T *;
#endif

  global_pointer_t m_Ptr;

  template <typename T2, typename PropertyListT> friend class annotated_ptr;

#ifdef __SYCL_DEVICE_ONLY__
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(
      detail::PropertyMetaInfo<Props>::name...,
      detail::PropertyMetaInfo<Props>::value...)]] global_pointer_t Ptr) {
    m_Ptr = Ptr;
  }
#endif

public:
  static constexpr bool is_valid_property_list =
      is_property_list<property_list_t>::value;
  static_assert(is_valid_property_list, "Property list is invalid.");
  static constexpr bool contains_valid_properties =
      check_property_list<T *, Props...>::value;
  static_assert(contains_valid_properties,
                "The property list contains invalid property.");
  // check the set if FPGA specificed properties are used
  static constexpr bool hasValidFPGAProperties =
      detail::checkValidFPGAPropertySet<Props...>::value;
  static_assert(hasValidFPGAProperties,
                "FPGA Interface properties (i.e. awidth, dwidth, etc.)"
                "can only be set with BufferLocation together.");
  // check if conduit and register_map properties are specified together
  static constexpr bool hasConduitAndRegisterMapProperties =
      detail::checkHasConduitAndRegisterMap<Props...>::value;
  static_assert(hasConduitAndRegisterMapProperties,
                "The properties conduit and register_map cannot be "
                "specified at the same time.");

  annotated_ptr() noexcept = default;
  annotated_ptr(const annotated_ptr &) = default;
  annotated_ptr &operator=(const annotated_ptr &) = default;

  explicit annotated_ptr(T *Ptr,
                         const property_list_t & = properties{}) noexcept
      : m_Ptr(global_pointer_t(Ptr)) {}

  // Constructs an annotated_ptr object from a raw pointer and variadic
  // properties. The new property set contains all properties of the input
  // variadic properties. The same property in `Props...` and
  // `PropertyValueTs...` must have the same property value.
  template <typename... PropertyValueTs>
  explicit annotated_ptr(T *Ptr, const PropertyValueTs &...props) noexcept
      : m_Ptr(global_pointer_t(Ptr)) {
    static constexpr bool has_same_properties = std::is_same<
        property_list_t,
        detail::merged_properties_t<property_list_t,
                                    decltype(properties{props...})>>::value;
    static_assert(
        has_same_properties,
        "The property list must contain all properties of the input of the "
        "constructor");
  }

  // Constructs an annotated_ptr object from another annotated_ptr object.
  // The new property set contains all properties of the input
  // annotated_ptr object. The same property in `Props...` and `PropertyList2`
  // must have the same property value.
  template <typename T2, typename PropertyList2>
  explicit annotated_ptr(const annotated_ptr<T2, PropertyList2> &other) noexcept
      : m_Ptr(other.m_Ptr) {
    static constexpr bool is_input_convertible =
        std::is_convertible<T2 *, T *>::value;
    static_assert(
        is_input_convertible,
        "The underlying pointer type of the input annotated_ptr is not "
        "convertible to the target pointer type");

    static constexpr bool has_same_properties = std::is_same<
        property_list_t,
        detail::merged_properties_t<property_list_t, PropertyList2>>::value;
    static_assert(
        has_same_properties,
        "The constructed annotated_ptr type must contain all the properties "
        "of the input annotated_ptr");
  }

  // Constructs an annotated_ptr object from another annotated_ptr object and
  // a property list. The new property set is the union of property lists
  // `PropertyListU` and `PropertyListV`. The same property in `PropertyListU`
  // and `PropertyListV` must have the same property value.
  template <typename T2, typename PropertyListU, typename PropertyListV>
  explicit annotated_ptr(const annotated_ptr<T2, PropertyListU> &other,
                         const PropertyListV &) noexcept
      : m_Ptr(other.m_Ptr) {
    static constexpr bool is_input_convertible =
        std::is_convertible<T2 *, T *>::value;
    static_assert(
        is_input_convertible,
        "The underlying pointer type of the input annotated_ptr is not "
        "convertible to the target pointer type");

    static constexpr bool has_same_properties = std::is_same<
        property_list_t,
        detail::merged_properties_t<PropertyListU, PropertyListV>>::value;
    static_assert(
        has_same_properties,
        "The property list of constructed annotated_ptr type must be the "
        "union of the input property lists");
  }

  reference operator*() const noexcept { return reference(m_Ptr); }

  reference operator[](std::ptrdiff_t idx) const noexcept {
    return reference(m_Ptr + idx);
  }

  annotated_ptr operator+(size_t offset) const noexcept {
    return annotated_ptr<T, property_list_t>(m_Ptr + offset);
  }

  std::ptrdiff_t operator-(annotated_ptr other) const noexcept {
    return m_Ptr - other.m_Ptr;
  }

  explicit operator bool() const noexcept { return m_Ptr != nullptr; }

  operator T *() const noexcept = delete;

  T *get() const noexcept { return m_Ptr; }

  annotated_ptr &operator++() noexcept {
    m_Ptr += 1;
    return *this;
  }

  annotated_ptr operator++(int) noexcept {
    auto tmp = *this;
    m_Ptr += 1;
    return tmp;
  }

  annotated_ptr &operator--() noexcept {
    m_Ptr -= 1;
    return *this;
  }

  annotated_ptr operator--(int) noexcept {
    auto tmp = *this;
    m_Ptr -= 1;
    return tmp;
  }

  template <typename PropertyT> static constexpr bool has_property() {
    return property_list_t::template has_property<PropertyT>();
  }

  template <typename PropertyT> static constexpr auto get_property() {
    return property_list_t::template get_property<PropertyT>();
  }
};

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl
