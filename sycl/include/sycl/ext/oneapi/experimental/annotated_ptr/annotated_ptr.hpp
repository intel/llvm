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

//===----------------------------------------------------------------------===//
//        Specific properties of annotated_ptr
//===----------------------------------------------------------------------===//
struct alignment_key {
  template <int K>
  using value_t = property_value<alignment_key, std::integral_constant<int, K>>;
};

template <int K> inline constexpr alignment_key::value_t<K> alignment;

template <> struct is_property_key<alignment_key> : std::true_type {};

template <typename T, int W>
struct is_valid_property<T, alignment_key::value_t<W>>
    : std::bool_constant<std::is_pointer<T>::value> {};

template <typename T, typename PropertyListT>
struct is_property_key_of<alignment_key, annotated_ptr<T, PropertyListT>>
    : std::true_type {};

namespace detail {

template <> struct PropertyToKind<alignment_key> {
  static constexpr PropKind Kind = PropKind::Alignment;
};

template <> struct IsCompileTimeProperty<alignment_key> : std::true_type {};

template <int N> struct PropertyMetaInfo<alignment_key::value_t<N>> {
  static constexpr const char *name = "sycl-alignment";
  static constexpr int value = N;
};

} // namespace detail

namespace {
#define PROPAGATE_OP(op)                                                       \
  annotated_ref operator op(const T &rhs) {                                    \
    (*m_Ptr) op rhs;                                                           \
    return *this;                                                              \
  }

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

template <typename T, typename PropertyListT = detail::empty_properties_t>
class annotated_ref {
  // This should always fail when instantiating the unspecialized version.
  static_assert(is_property_list<PropertyListT>::value,
                "Property list is invalid.");
};

template <typename T, typename... Props>
class annotated_ref<T, detail::properties_t<Props...>> {
  using property_list_t = detail::properties_t<Props...>;

private:
  T *m_Ptr;

public:
  annotated_ref(T *Ptr) : m_Ptr(Ptr) {}
  annotated_ref(const annotated_ref &) = default;

  operator T() const {
#ifdef __SYCL_DEVICE_ONLY__
    return *__builtin_intel_sycl_ptr_annotation(
        m_Ptr, detail::PropertyMetaInfo<Props>::name...,
        detail::PropertyMetaInfo<Props>::value...);
#else
    return *m_Ptr;
#endif
  }

  annotated_ref &operator=(const T &Obj) {
#ifdef __SYCL_DEVICE_ONLY__
    *__builtin_intel_sycl_ptr_annotation(
        m_Ptr, detail::PropertyMetaInfo<Props>::name...,
        detail::PropertyMetaInfo<Props>::value...) = Obj;
#else
    *m_Ptr = Obj;
#endif
    return *this;
  }

  annotated_ref &operator=(const annotated_ref &) = default;

  PROPAGATE_OP(+=)
  PROPAGATE_OP(-=)
  PROPAGATE_OP(*=)
  PROPAGATE_OP(/=)
  PROPAGATE_OP(%=)
  PROPAGATE_OP(^=)
  PROPAGATE_OP(&=)
  PROPAGATE_OP(|=)
};

#undef PROPAGATE_OP
} // namespace

#ifdef __cpp_deduction_guides
template <typename T, typename... Args>
annotated_ptr(T *, Args...)
    -> annotated_ptr<T, typename detail::DeducedProperties<Args...>::type>;

template <typename T, typename old, typename... ArgT>
annotated_ptr(annotated_ptr<T, old>, properties<std::tuple<ArgT...>>)
    -> annotated_ptr<
        T, detail::merged_properties_t<old, detail::properties_t<ArgT...>>>;
#endif // __cpp_deduction_guides

template <typename T, typename PropertyListT = detail::empty_properties_t>
class annotated_ptr {
  // This should always fail when instantiating the unspecialized version.
  static_assert(is_property_list<PropertyListT>::value,
                "Property list is invalid.");
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
  static_assert(is_property_list<property_list_t>::value,
                "Property list is invalid.");
  static_assert(check_property_list<T *, Props...>::value,
                "The property list contains invalid property.");
  // check the set if FPGA specificed properties are used
  static_assert(detail::checkValidFPGAPropertySet<Props...>::value,
                "FPGA Interface properties (i.e. awidth, dwidth, etc.)"
                "can only be set with BufferLocation together.");

  annotated_ptr() noexcept = default;
  annotated_ptr(const annotated_ptr &) = default;
  annotated_ptr &operator=(annotated_ptr &) = default;

  annotated_ptr(T *Ptr, const property_list_t & = properties{}) noexcept
      : m_Ptr(global_pointer_t(Ptr)) {}

  // Constructs an annotated_ptr object from a raw pointer and variadic
  // properties. The new property set contains all properties of the input
  // variadic properties. The same property in `Props...` and
  // `PropertyValueTs...` must have the same property value.
  template <typename... PropertyValueTs>
  annotated_ptr(T *Ptr, const PropertyValueTs &...props) noexcept
      : m_Ptr(global_pointer_t(Ptr)) {
    static_assert(
        std::is_same<
            property_list_t,
            detail::merged_properties_t<property_list_t,
                                        decltype(properties{props...})>>::value,
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
    static_assert(
        std::is_convertible<T2 *, T *>::value,
        "The underlying pointer type of the input annotated_ptr is not "
        "convertible to the target pointer type");

    static_assert(
        std::is_same<
            property_list_t,
            detail::merged_properties_t<property_list_t, PropertyList2>>::value,
        "The constructed annotated_ptr type must contain all the properties "
        "of "
        "the input annotated_ptr");
  }

  // Constructs an annotated_ptr object from another annotated_ptr object and
  // a property list. The new property set is the union of property lists
  // `PropertyListU` and `PropertyListV`. The same property in `PropertyListU`
  // and `PropertyListV` must have the same property value.
  template <typename T2, typename PropertyListU, typename PropertyListV>
  explicit annotated_ptr(const annotated_ptr<T2, PropertyListU> &other,
                         const PropertyListV &) noexcept
      : m_Ptr(other.m_Ptr) {
    static_assert(
        std::is_convertible<T2 *, T *>::value,
        "The underlying pointer type of the input annotated_ptr is not "
        "convertible to the target pointer type");

    static_assert(
        std::is_same<property_list_t, detail::merged_properties_t<
                                          PropertyListU, PropertyListV>>::value,
        "The property list of constructed annotated_ptr type must be the "
        "union "
        "of the input property lists");
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

  operator T *() noexcept = delete;
  operator T *() const = delete;

  T *get() const noexcept { return m_Ptr; }

  annotated_ptr &operator=(T *) noexcept {
    return annotated_ptr<T, property_list_t>(m_Ptr);
  }

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
