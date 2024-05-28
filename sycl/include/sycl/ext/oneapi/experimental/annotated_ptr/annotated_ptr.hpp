//
//==----------- annotated_ptr.hpp - SYCL annotated_ptr extension -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines.hpp>
#include <sycl/ext/intel/experimental/cache_control_properties.hpp>
#include <sycl/ext/intel/experimental/fpga_annotated_properties.hpp>
#include <sycl/ext/oneapi/experimental/annotated_ptr/annotated_ptr_properties.hpp>
#include <sycl/ext/oneapi/experimental/common_annotated_properties/properties.hpp>
#include <sycl/ext/oneapi/properties/properties.hpp>
#include <sycl/ext/oneapi/properties/property.hpp>
#include <sycl/ext/oneapi/properties/property_value.hpp>
#include <sycl/pointers.hpp>

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

template <typename T, typename PropertyListT = empty_properties_t>
class annotated_ref {
  // This should always fail when instantiating the unspecialized version.
  static constexpr bool is_valid_property_list =
      is_property_list<PropertyListT>::value;
  static_assert(is_valid_property_list, "Property list is invalid.");
};

namespace detail {
template <class T> struct is_ann_ref_impl : std::false_type {};
template <class T, class P>
struct is_ann_ref_impl<annotated_ref<T, P>> : std::true_type {};
template <class T, class P>
struct is_ann_ref_impl<const annotated_ref<T, P>> : std::true_type {};
template <class T>
constexpr bool is_ann_ref_v =
    is_ann_ref_impl<std::remove_reference_t<T>>::value;

template <typename... Ts>
using contains_alignment =
    detail::ContainsProperty<alignment_key, std::tuple<Ts...>>;

// properties filter
template <typename property_list, template <class...> typename filter>
using PropertiesFilter =
    sycl::detail::boost::mp11::mp_copy_if<property_list, filter>;

// filter properties that are applied on annotations
template <typename... Props>
using annotation_filter = properties<
    PropertiesFilter<std::tuple<Props...>, propagateToPtrAnnotation>>;
} // namespace detail

template <typename I, typename P> struct annotationHelper {};

// unpack properties to varadic template
template <typename I, typename... P>
struct annotationHelper<I, detail::properties_t<P...>> {
  static I *annotate(I *ptr) {
    return __builtin_intel_sycl_ptr_annotation(
        ptr, detail::PropertyMetaInfo<P>::name...,
        detail::PropertyMetaInfo<P>::value...);
  }

  static I load(I *ptr) {
    return *__builtin_intel_sycl_ptr_annotation(
        ptr, detail::PropertyMetaInfo<P>::name...,
        detail::PropertyMetaInfo<P>::value...);
  }

  template <class O> static I store(I *ptr, O &&Obj) {
    return *__builtin_intel_sycl_ptr_annotation(
               ptr, detail::PropertyMetaInfo<P>::name...,
               detail::PropertyMetaInfo<P>::value...) = std::forward<O>(Obj);
  }
};

template <typename T, typename... Props>
class annotated_ref<T, detail::properties_t<Props...>> {
  using property_list_t = detail::properties_t<Props...>;

  static_assert(
      std::is_trivially_copyable_v<T>,
      "annotated_ref can only encapsulate a trivially-copyable type!");

private:
  T *m_Ptr;
  explicit annotated_ref(T *Ptr) : m_Ptr(Ptr) {}

public:
  annotated_ref(const annotated_ref &) = delete;

  // implicit conversion with annotaion
  operator T() const {
#ifdef __SYCL_DEVICE_ONLY__
    return annotationHelper<T, detail::annotation_filter<Props...>>::load(
        m_Ptr);
#else
    return *m_Ptr;
#endif
  }

  // assignment operator with annotaion
  template <class O, typename = std::enable_if_t<!detail::is_ann_ref_v<O>>>
  T operator=(O &&Obj) const {
#ifdef __SYCL_DEVICE_ONLY__
    return annotationHelper<T, detail::annotation_filter<Props...>>::store(
        m_Ptr, Obj);
#else
    return *m_Ptr = std::forward<O>(Obj);
#endif
  }

  template <class O, class P>
  T operator=(const annotated_ref<O, P> &Ref) const {
    O t2 = Ref.operator O();
    return *this = t2;
  }

  // propagate compound operators
#define PROPAGATE_OP(op)                                                       \
  template <class O, typename = std::enable_if_t<!detail::is_ann_ref_v<O>>>    \
  T operator op(O &&rhs) const {                                               \
    T t = this->operator T();                                                  \
    t op std::forward<O>(rhs);                                                 \
    *this = t;                                                                 \
    return t;                                                                  \
  }                                                                            \
  template <class O, class P>                                                  \
  T operator op(const annotated_ref<O, P> &rhs) const {                        \
    T t = this->operator T();                                                  \
    O t2 = rhs.operator T();                                                   \
    t op t2;                                                                   \
    *this = t;                                                                 \
    return t;                                                                  \
  }
  PROPAGATE_OP(+=)
  PROPAGATE_OP(-=)
  PROPAGATE_OP(*=)
  PROPAGATE_OP(/=)
  PROPAGATE_OP(%=)
  PROPAGATE_OP(^=)
  PROPAGATE_OP(&=)
  PROPAGATE_OP(|=)
  PROPAGATE_OP(<<=)
  PROPAGATE_OP(>>=)
#undef PROPAGATE_OP

  // propagate binary operators
#define PROPAGATE_OP(op)                                                       \
  template <class O>                                                           \
  friend auto operator op(O &&a, const annotated_ref &b)                       \
      ->decltype(std::forward<O>(a) op std::declval<T>()) {                    \
    return std::forward<O>(a) op b.operator T();                               \
  }                                                                            \
  template <class O, typename = std::enable_if_t<!detail::is_ann_ref_v<O>>>    \
  friend auto operator op(const annotated_ref &a, O &&b)                       \
      ->decltype(std::declval<T>() op std::forward<O>(b)) {                    \
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
  auto operator op() const->decltype(op std::declval<O>()) {                   \
    return op this->operator O();                                              \
  }
  PROPAGATE_OP(+)
  PROPAGATE_OP(-)
  PROPAGATE_OP(!)
  PROPAGATE_OP(~)
#undef PROPAGATE_OP

  // Propagate inc/dec operators
  T operator++() const {
    T t = this->operator T();
    ++t;
    *this = t;
    return t;
  }

  T operator++(int) const {
    T t1 = this->operator T();
    T t2 = t1;
    t2++;
    *this = t2;
    return t1;
  }

  T operator--() const {
    T t = this->operator T();
    --t;
    *this = t;
    return t;
  }

  T operator--(int) const {
    T t1 = this->operator T();
    T t2 = t1;
    t2--;
    *this = t2;
    return t1;
  }

  template <class T2, class P2> friend class annotated_ptr;
};

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

  static_assert(std::is_same_v<T, void> || std::is_trivially_copyable_v<T>,
                "annotated_ptr can only encapsulate either "
                "a trivially-copyable type "
                "or void!");

  using property_list_t = detail::properties_t<Props...>;

  // annotated_ref type
  using reference =
      sycl::ext::oneapi::experimental::annotated_ref<T, property_list_t>;

#ifdef __ENABLE_USM_ADDR_SPACE__
  using global_pointer_t = std::conditional_t<
      detail::IsUsmKindDevice<property_list_t>::value,
      typename sycl::ext::intel::decorated_device_ptr<T>::pointer,
      std::conditional_t<
          detail::IsUsmKindHost<property_list_t>::value,
          typename sycl::ext::intel::decorated_host_ptr<T>::pointer,
          typename decorated_global_ptr<T>::pointer>>;
#else
  using global_pointer_t = typename decorated_global_ptr<T>::pointer;
#endif // __ENABLE_USM_ADDR_SPACE__

  T *m_Ptr;

  template <typename T2, typename PropertyListT> friend class annotated_ptr;

#ifdef __SYCL_DEVICE_ONLY__
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(
      detail::PropertyMetaInfo<Props>::name...,
      detail::PropertyMetaInfo<Props>::value...)]] global_pointer_t Ptr) {
    m_Ptr = Ptr;
  }
#endif

public:
  annotated_ptr() noexcept = default;
  annotated_ptr(const annotated_ptr &) = default;
  annotated_ptr &operator=(const annotated_ptr &) = default;

  explicit annotated_ptr(T *Ptr,
                         const property_list_t & = property_list_t{}) noexcept
      : m_Ptr(Ptr) {}

  // Constructs an annotated_ptr object from a raw pointer and variadic
  // properties. The new property set contains all properties of the input
  // variadic properties. The same property in `Props...` and
  // `PropertyValueTs...` must have the same property value.
  template <typename... PropertyValueTs>
  explicit annotated_ptr(T *Ptr, const PropertyValueTs &...props) noexcept
      : m_Ptr(Ptr) {

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

  std::ptrdiff_t operator-(annotated_ptr other) const noexcept {
    return m_Ptr - other.m_Ptr;
  }

  explicit operator bool() const noexcept { return m_Ptr != nullptr; }

  operator T *() const noexcept = delete;

  T *get() const noexcept {
#ifdef __SYCL_DEVICE_ONLY__
    return annotationHelper<T, detail::annotation_filter<Props...>>::annotate(
        m_Ptr);
#else
    return m_Ptr;
#endif
  }

  // When the properties contain alignment, operator '[]', '+', '++' and '--'
  // (both post- and prefix) are disabled. Calling these operators when
  // alignment is present causes a compile error. Note that clang format is
  // turned off for these operators to make sure the complete error notes are
  // printed
  // clang-format off
  template <bool has_alignment = detail::contains_alignment<Props...>::value,
            class = std::enable_if_t<!has_alignment>>
  reference operator[](std::ptrdiff_t idx) const noexcept {
    return reference(m_Ptr + idx);
  }

  template <bool has_alignment = detail::contains_alignment<Props...>::value,
            class = std::enable_if_t<has_alignment>>
  auto operator[](std::ptrdiff_t idx) const noexcept -> decltype("operator[] is not available when alignment is specified!") = delete;

  template <bool has_alignment = detail::contains_alignment<Props...>::value,
            class = std::enable_if_t<!has_alignment>>
  annotated_ptr operator+(size_t offset) const noexcept {
    return annotated_ptr(m_Ptr + offset);
  }

  template <bool has_alignment = detail::contains_alignment<Props...>::value,
            class = std::enable_if_t<has_alignment>>
  auto operator+(size_t offset) const noexcept -> decltype("operator+ is not available when alignment is specified!") = delete;

  template <bool has_alignment = detail::contains_alignment<Props...>::value,
            class = std::enable_if_t<!has_alignment>>
  annotated_ptr &operator++() noexcept {
    m_Ptr += 1;
    return *this;
  }

  template <bool has_alignment = detail::contains_alignment<Props...>::value,
            class = std::enable_if_t<has_alignment>>
  auto operator++() noexcept -> decltype("operator++ is not available when alignment is specified!") = delete;

  template <bool has_alignment = detail::contains_alignment<Props...>::value,
            class = std::enable_if_t<!has_alignment>>
  annotated_ptr operator++(int) noexcept {
    auto tmp = *this;
    m_Ptr += 1;
    return tmp;
  }

  template <bool has_alignment = detail::contains_alignment<Props...>::value,
            class = std::enable_if_t<has_alignment>>
  auto operator++(int) noexcept -> decltype("operator++ is not available when alignment is specified!") = delete;

  template <bool has_alignment = detail::contains_alignment<Props...>::value,
            class = std::enable_if_t<!has_alignment>>
  annotated_ptr &operator--() noexcept {
    m_Ptr -= 1;
    return *this;
  }

  template <bool has_alignment = detail::contains_alignment<Props...>::value,
            class = std::enable_if_t<has_alignment>>
  auto operator--() noexcept -> decltype("operator-- is not available when alignment is specified!") = delete;

  template <bool has_alignment = detail::contains_alignment<Props...>::value,
            class = std::enable_if_t<!has_alignment>>
  annotated_ptr operator--(int) noexcept {
    auto tmp = *this;
    m_Ptr -= 1;
    return tmp;
  }

  template <bool has_alignment = detail::contains_alignment<Props...>::value,
            class = std::enable_if_t<has_alignment>>
  auto operator--(int) noexcept -> decltype("operator-- is not available when alignment is specified!") = delete;

  // clang-format on

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

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl
