//==----------- annotated_arg.hpp - SYCL annotated_arg extension -----------==//
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
#include <sycl/pointers.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>
#include <variant>

namespace sycl {

// device_copyable trait
template <typename T, typename PropertyList>
struct is_device_copyable<
    ext::oneapi::experimental::annotated_arg<T, PropertyList>>
    : is_device_copyable<T> {};

inline namespace _V1 {
namespace ext {
namespace oneapi {
namespace experimental {

namespace detail {

// Type-trait for checking if a type defines `operator[]`.
template <typename T>
struct HasSubscriptOperator
    : std::bool_constant<
          !std::is_void_v<decltype(std::declval<T>().operator[](0))>> {};

template <class T> struct is_ann_arg_impl : std::false_type {};
template <class T, class P>
struct is_ann_arg_impl<annotated_arg<T, P>> : std::true_type {};
template <class T, class P>
struct is_ann_arg_impl<const annotated_arg<T, P>> : std::true_type {};
template <class T>
constexpr bool is_ann_arg_v =
    is_ann_arg_impl<std::remove_reference_t<T>>::value;

template <class T> struct GetUnderlyingTImpl;
template <class T, class P> struct GetUnderlyingTImpl<annotated_arg<T, P>> {
  using type = T;
};
template <class T>
using GetUnderlyingT = typename GetUnderlyingTImpl<
    std::remove_cv_t<std::remove_reference_t<T>>>::type;

} // namespace detail

// Deduction guide
template <typename T, typename... Args>
annotated_arg(T, Args...)
    -> annotated_arg<T, typename detail::DeducedProperties<Args...>::type>;

template <typename T, typename old, typename... ArgT>
annotated_arg(annotated_arg<T, old>, properties<std::tuple<ArgT...>>)
    -> annotated_arg<
        T, detail::merged_properties_t<old, detail::properties_t<ArgT...>>>;

template <typename T, typename PropertyListT = empty_properties_t>
class annotated_arg {
  // This should always fail when instantiating the unspecialized version.
  static constexpr bool is_valid_property_list =
      is_property_list<PropertyListT>::value;
  static_assert(is_valid_property_list, "Property list is invalid.");
};

// Partial specialization for pointer type
template <typename T, typename... Props>
class __SYCL_SPECIAL_CLASS
__SYCL_TYPE(annotated_arg) annotated_arg<T *, detail::properties_t<Props...>> {
  using property_list_t = detail::properties_t<Props...>;

  T *obj;

  template <typename T2, typename PropertyListT> friend class annotated_arg;

#ifdef __ENABLE_USM_ADDR_SPACE__
  using global_pointer_t = std::conditional_t<
      detail::IsUsmKindDevice<property_list_t>::value,
      typename sycl::ext::intel::decorated_device_ptr<T>::pointer,
      std::conditional_t<
          detail::IsUsmKindHost<property_list_t>::value,
          typename sycl::ext::intel::decorated_host_ptr<T>::pointer,
          typename decorated_global_ptr<T>::pointer>>;
#else  // __ENABLE_USM_ADDR_SPACE__
  using global_pointer_t = typename decorated_global_ptr<T>::pointer;
#endif // __ENABLE_USM_ADDR_SPACE__

#ifdef __SYCL_DEVICE_ONLY__
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(
      detail::PropertyMetaInfo<Props>::name...,
      detail::PropertyMetaInfo<Props>::value...)]] global_pointer_t _obj) {
    obj = _obj;
  }
#endif

public:
  annotated_arg() noexcept = default;
  annotated_arg(const annotated_arg &) = default;
  annotated_arg &operator=(annotated_arg &) = default;

  annotated_arg(T *_ptr,
                const property_list_t &PropList = property_list_t{}) noexcept
      : obj(_ptr) {
    (void)PropList;
  }

  // Constructs an annotated_arg object from a raw pointer and variadic
  // properties. The new property set contains all properties of the input
  // variadic properties. The same property in `Props...` and
  // `PropertyValueTs...` must have the same property value.
  template <typename... PropertyValueTs>
  annotated_arg(T *_ptr, const PropertyValueTs &...props) noexcept : obj(_ptr) {
    static constexpr bool has_same_properties = std::is_same<
        property_list_t,
        detail::merged_properties_t<property_list_t,
                                    decltype(properties{props...})>>::value;
    static_assert(
        has_same_properties,
        "The property list must contain all properties of the input of the "
        "constructor");
  }

  // Constructs an annotated_arg object from another annotated_arg object.
  // The new property set contains all properties of the input
  // annotated_arg object. The same property in `Props...` and `PropertyList2`
  // must have the same property value.
  template <typename T2, typename PropertyList2>
  explicit annotated_arg(const annotated_arg<T2, PropertyList2> &other) noexcept
      : obj(other.obj) {
    static constexpr bool is_input_convertible =
        std::is_convertible<T2, T *>::value;
    static_assert(is_input_convertible,
                  "The underlying data type of the input annotated_arg is not "
                  "compatible");

    static constexpr bool has_same_properties = std::is_same<
        property_list_t,
        detail::merged_properties_t<property_list_t, PropertyList2>>::value;
    static_assert(
        has_same_properties,
        "The constructed annotated_arg type must contain all the properties "
        "of the input annotated_arg");
  }

  // Constructs an annotated_arg object from another annotated_arg object and a
  // property list. The new property set is the union of property lists
  // `PropertyListU` and `PropertyListV`. The same property in `PropertyListU`
  // and `PropertyListV` must have the same property value.
  template <typename T2, typename PropertyListU, typename PropertyListV>
  explicit annotated_arg(const annotated_arg<T2, PropertyListU> &other,
                         const PropertyListV &proplist) noexcept
      : obj(other.obj) {
    (void)proplist;
    static constexpr bool is_input_convertible =
        std::is_convertible<T2, T *>::value;
    static_assert(is_input_convertible,
                  "The underlying data type of the input annotated_arg is not "
                  "compatible");

    static constexpr bool has_same_properties = std::is_same<
        property_list_t,
        detail::merged_properties_t<PropertyListU, PropertyListV>>::value;
    static_assert(
        has_same_properties,
        "The property list of constructed annotated_arg type must be the union "
        "of the input property lists");
  }

  operator T *() noexcept { return obj; }
  operator T *() const noexcept { return obj; }

  T &operator[](std::ptrdiff_t idx) const noexcept { return obj[idx]; }

  T *operator->() const noexcept { return obj; }

  template <typename PropertyT> static constexpr bool has_property() {
    return property_list_t::template has_property<PropertyT>();
  }

  template <typename PropertyT> static constexpr auto get_property() {
    return property_list_t::template get_property<PropertyT>();
  }

  // *************************************************************************
  // All static error checking is added here instead of placing inside neat
  // functions to minimize the number lines printed out when an assert
  // is triggered.
  // static constexprs are used to ensure that the triggered assert prints
  // a message that is very readable. Without these, the assert will
  // print out long templated names
  // *************************************************************************
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
                "FPGA Interface properties (i.e. awidth, dwidth, etc.) "
                "can only be set with BufferLocation together.");
  // check if conduit and register_map properties are specified together
  static constexpr bool hasConduitAndRegisterMapProperties =
      detail::checkHasConduitAndRegisterMap<Props...>::value;
  static_assert(hasConduitAndRegisterMapProperties,
                "The properties conduit and register_map cannot be "
                "specified at the same time.");
};

// Partial specialization for non-pointer type
template <typename T, typename... Props>
class __SYCL_SPECIAL_CLASS
__SYCL_TYPE(annotated_arg) annotated_arg<T, detail::properties_t<Props...>> {
  using property_list_t = detail::properties_t<Props...>;
  using UnderlyingT = T;

  template <typename T2, typename PropertyListT> friend class annotated_arg;

  T obj;

#ifdef __SYCL_DEVICE_ONLY__
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(
      detail::PropertyMetaInfo<Props>::name...,
      detail::PropertyMetaInfo<Props>::value...)]] T _obj) {
    obj = _obj;
  }
#endif

public:
  annotated_arg() noexcept = default;
  annotated_arg(const annotated_arg &) = default;
  annotated_arg &operator=(annotated_arg &) = default;

  annotated_arg(const T &_obj,
                const property_list_t &PropList = property_list_t{}) noexcept
      : obj(_obj) {
    (void)PropList;
  }

  // Constructs an annotated_arg object from a raw pointer and variadic
  // properties. The new property set contains all properties of the input
  // variadic properties. The same property in `Props...` and
  // `PropertyValueTs...` must have the same property value.
  template <typename... PropertyValueTs>
  annotated_arg(const T &_obj, PropertyValueTs... props) noexcept : obj(_obj) {
    static constexpr bool has_same_properties = std::is_same<
        property_list_t,
        detail::merged_properties_t<property_list_t,
                                    decltype(properties{props...})>>::value;
    static_assert(
        has_same_properties,
        "The property list must contain all properties of the input of the "
        "constructor");
  }

  // Constructs an annotated_arg object from another annotated_arg object.
  // The new property set contains all properties of the input
  // annotated_arg object. The same property in `Props...` and `PropertyList2`
  // must have the same property value.
  template <typename T2, typename PropertyList2>
  explicit annotated_arg(const annotated_arg<T2, PropertyList2> &other) noexcept
      : obj(other.obj) {
    static constexpr bool is_input_convertible =
        std::is_convertible<T2, T>::value;
    static_assert(is_input_convertible,
                  "The underlying data type of the input annotated_arg is not "
                  "compatible");

    static constexpr bool has_same_properties = std::is_same<
        property_list_t,
        detail::merged_properties_t<property_list_t, PropertyList2>>::value;
    static_assert(
        has_same_properties,
        "The constructed annotated_arg type must contain all the properties "
        "of the input annotated_arg");
  }

  // Constructs an annotated_arg object from another annotated_arg object and a
  // property list. The new property set is the union of property lists
  // `PropertyListU` and `PropertyListV`. The same property in `PropertyListU`
  // and `PropertyListV` must have the same property value.
  template <typename T2, typename PropertyListU, typename PropertyListV>
  explicit annotated_arg(const annotated_arg<T2, PropertyListU> &other,
                         const PropertyListV &proplist) noexcept
      : obj(other.obj) {
    (void)proplist;
    static constexpr bool is_input_convertible =
        std::is_convertible<T2, T>::value;
    static_assert(is_input_convertible,
                  "The underlying data type of the input annotated_arg is not "
                  "compatible");

    static constexpr bool has_same_properties = std::is_same<
        property_list_t,
        detail::merged_properties_t<PropertyListU, PropertyListV>>::value;
    static_assert(
        has_same_properties,
        "The property list of constructed annotated_arg type must be the union "
        "of the input property lists");
  }

  operator T() noexcept { return obj; }
  operator T() const noexcept { return obj; }

  template <class RelayT = T>
  std::enable_if_t<detail::HasSubscriptOperator<RelayT>::value,
                   decltype(std::declval<RelayT>().operator[](0))> &
  operator[](std::ptrdiff_t idx) const noexcept {
    return obj.operator[](idx);
  }

  template <typename PropertyT> static constexpr bool has_property() {
    return property_list_t::template has_property<PropertyT>();
  }

  template <typename PropertyT> static constexpr auto get_property() {
    return property_list_t::template get_property<PropertyT>();
  }

// propagate binary operators
#define PROPAGATE_OP(op)                                                       \
  template <class O> friend auto operator op(O &&a, const annotated_arg &b) {  \
    if constexpr (!detail::is_ann_arg_v<O>)                                    \
      return std::forward<O>(a) op b.operator T();                             \
    else                                                                       \
      return a.operator detail::GetUnderlyingT<O>() op b.operator T();         \
  }                                                                            \
  template <class O, typename = std::enable_if_t<!detail::is_ann_arg_v<O>>>    \
  friend auto operator op(const annotated_arg &a, O &&b)                       \
      -> decltype(std::declval<T>() op std::forward<O>(b)) {                   \
    return a.operator T() op std::forward<O>(b);                               \
  }
  PROPAGATE_OP(+)
  PROPAGATE_OP(-)
  PROPAGATE_OP(*)
  PROPAGATE_OP(/)
  PROPAGATE_OP(%)
  PROPAGATE_OP(|)
  PROPAGATE_OP(&)
  PROPAGATE_OP(^)
  PROPAGATE_OP(<<)
  PROPAGATE_OP(>>)
  PROPAGATE_OP(<)
  PROPAGATE_OP(<=)
  PROPAGATE_OP(>)
  PROPAGATE_OP(>=)
  PROPAGATE_OP(==)
  PROPAGATE_OP(!=)
  PROPAGATE_OP(&&)
  PROPAGATE_OP(||)
#undef PROPAGATE_OP

// Propagate unary operators
// by setting a default template we get SFINAE to kick in
#define PROPAGATE_OP(op)                                                       \
  template <typename O = T>                                                    \
  auto operator op() const -> decltype(op std::declval<O>()) {                 \
    return op this->operator O();                                              \
  }
  PROPAGATE_OP(+)
  PROPAGATE_OP(-)
  PROPAGATE_OP(!)
  PROPAGATE_OP(~)
#undef PROPAGATE_OP

  // *************************************************************************
  // All static error checking is added here instead of placing inside neat
  // functions to minimize the number lines printed out when an assert
  // is triggered.
  // static constexprs are used to ensure that the triggered assert prints
  // a message that is very readable. Without these, the assert will
  // print out long templated names
  // *************************************************************************
  static constexpr bool is_device_copyable = is_device_copyable_v<T>;
  static_assert(is_device_copyable, "Type T must be device copyable.");

  // check if invalid properties are specified for non pointer type
  static constexpr bool has_buffer_location =
      has_property<buffer_location_key>();
  static_assert(!has_buffer_location,
                "Property buffer_location cannot be specified for "
                "annotated_arg<T> when T is a non pointer type.");

  static constexpr bool has_awidth = has_property<awidth_key>();
  static_assert(!has_awidth, "Property awidth cannot be specified for "
                             "annotated_arg<T> when T is a non pointer type.");

  static constexpr bool has_dwidth = has_property<dwidth_key>();
  static_assert(!has_dwidth, "Property dwidth cannot be specified for "
                             "annotated_arg<T> when T is a non pointer type.");

  static constexpr bool has_latency = has_property<latency_key>();
  static_assert(!has_latency, "Property latency cannot be specified for "
                              "annotated_arg<T> when T is a non pointer type.");

  static constexpr bool has_read_write_mode =
      has_property<read_write_mode_key>();
  static_assert(!has_read_write_mode,
                "Property read_write_mode cannot be specified for "
                "annotated_arg<T> when T is a non pointer type.");

  static constexpr bool has_maxburst = has_property<maxburst_key>();
  static_assert(!has_maxburst,
                "Property maxburst cannot be specified for "
                "annotated_arg<T> when T is a non pointer type.");

  static constexpr bool has_wait_request = has_property<wait_request_key>();
  static_assert(!has_wait_request,
                "Property wait_request cannot be specified for "
                "annotated_arg<T> when T is a non pointer type.");

  static constexpr bool has_alignment = has_property<alignment_key>();
  static_assert(!has_alignment,
                "Property alignment cannot be specified for "
                "annotated_arg<T> when T is a non pointer type.");

  static constexpr bool has_usm_kind = has_property<usm_kind_key>();
  static_assert(!has_usm_kind,
                "Property usm_kind cannot be specified for "
                "annotated_arg<T> when T is a non pointer type.");

  static constexpr bool is_valid_property_list =
      is_property_list<property_list_t>::value;
  static_assert(is_valid_property_list, "Property list is invalid.");
  static constexpr bool contains_valid_properties =
      check_property_list<T, Props...>::value;
  static_assert(contains_valid_properties,
                "The property list contains invalid property.");
  // check the set if FPGA specificed properties are used
  static constexpr bool hasValidFPGAProperties =
      detail::checkValidFPGAPropertySet<Props...>::value;
  static_assert(hasValidFPGAProperties,
                "FPGA Interface properties (i.e. awidth, dwidth, etc.) "
                "can only be set with BufferLocation together.");
  // check if conduit and register_map properties are specified together
  static constexpr bool hasConduitAndRegisterMapProperties =
      detail::checkHasConduitAndRegisterMap<Props...>::value;
  static_assert(hasConduitAndRegisterMapProperties,
                "The properties conduit and register_map cannot be "
                "specified at the same time.");
};

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl
