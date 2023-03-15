//==----------- annotated_ptr.hpp - SYCL annotated_ptr extension -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstddef>
#include <type_traits>

#include <sycl/detail/stl_type_traits.hpp>
#include <sycl/exception.hpp>
#include <sycl/ext/oneapi/annotated_arg/properties.hpp>
#include <sycl/ext/oneapi/properties/properties.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext {
namespace oneapi {
namespace experimental {

namespace {
template <typename T, typename PropertyListT = detail::empty_properties_t>
class annotated_ref {
  // This should always fail when instantiating the unspecialized version.
  static_assert(is_property_list<PropertyListT>::value,
                "Property list is invalid.");
};

template <typename T, typename... Props>
class annotated_ref<T, detail::properties_t<Props...>> {
  using property_list_t = detail::properties_t<Props...>;
  using this_t = annotated_ref<T, property_list_t>;

private:
  T *m_Ptr
#ifdef __SYCL_DEVICE_ONLY__
      [[__sycl_detail__::add_ir_annotations_member(
          detail::PropertyMetaInfo<Props>::name...,
          detail::PropertyMetaInfo<Props>::value...)]]
#endif
      ;

public:
  annotated_ref(T *Ptr) : m_Ptr(Ptr) {}
  annotated_ref(const annotated_ref &) = default;

  operator T() const { return *m_Ptr; }

  this_t &operator=(const T &Obj) {
    *m_Ptr = Obj;
    return *this;
  }

  template <typename... OtherProps>
  this_t &
  operator=(const annotated_ref<T, detail::properties_t<OtherProps...>> &Obj) {
    *m_Ptr = *Obj.m_Ptr;
    return *this;
  }

  this_t &operator++() {
    ++(*m_Ptr);
    return *this;
  }

  this_t &operator++(int) {
    (*m_Ptr)++;
    return *this;
  }

  this_t &operator--() {
    --(*m_Ptr);
    return *this;
  }

  this_t &operator--(int) {
    (*m_Ptr)--;
    return *this;
  }

  T *operator->() const { return m_Ptr; }

#define PROPAGATE_BINARY_OP(op)                                                \
  template <typename Other> T operator op(const Other &rhs) {                  \
    return (*m_Ptr)op rhs;                                                     \
  }                                                                            \
  template <typename... OtherProps>                                            \
  T operator op(                                                               \
      const annotated_ref<T, detail::properties_t<OtherProps...>> &rhs) {      \
    return (*m_Ptr)op(*rhs.m_Ptr);                                             \
  }

#define PROPAGATE_ASSIGN_OP(op)                                                \
  template <typename Other> this_t &operator op(const Other &rhs) {            \
    (*m_Ptr) op rhs;                                                           \
    return *this;                                                              \
  }                                                                            \
  template <typename... OtherProps>                                            \
  this_t &operator op(                                                         \
      const annotated_ref<T, detail::properties_t<OtherProps...>> &rhs) {      \
    (*m_Ptr) op(*rhs.m_Ptr);                                                   \
    return *this;                                                              \
  }

#define PROPAGATE_UNARY_OP(op)                                                 \
  T operator op() { return op(*m_Ptr); }

#define PROPAGATE_LOGICAL_OP(op)                                               \
  template <typename Other> inline bool operator op(const Other &rhs) const {  \
    return (*m_Ptr)op rhs;                                                     \
  }                                                                            \
  template <typename... OtherProps>                                            \
  inline bool operator op(                                                     \
      const annotated_ref<T, detail::properties_t<OtherProps...>> &rhs)        \
      const {                                                                  \
    return (*m_Ptr)op(*rhs.m_Ptr);                                             \
  }

  /* Note. Operator || && do not implement short-circut evaluation */
  PROPAGATE_BINARY_OP(+)
  PROPAGATE_BINARY_OP(-)
  PROPAGATE_BINARY_OP(*)
  PROPAGATE_BINARY_OP(/)
  PROPAGATE_BINARY_OP(%)
  PROPAGATE_BINARY_OP(^)
  PROPAGATE_BINARY_OP(<<)
  PROPAGATE_BINARY_OP(>>)
  PROPAGATE_BINARY_OP(|)
  PROPAGATE_BINARY_OP(&)
  PROPAGATE_UNARY_OP(~)
  PROPAGATE_UNARY_OP(-)
  PROPAGATE_ASSIGN_OP(+=)
  PROPAGATE_ASSIGN_OP(-=)
  PROPAGATE_ASSIGN_OP(*=)
  PROPAGATE_ASSIGN_OP(/=)
  PROPAGATE_ASSIGN_OP(%=)
  PROPAGATE_ASSIGN_OP(^=)
  PROPAGATE_ASSIGN_OP(&=)
  PROPAGATE_ASSIGN_OP(|=)
  PROPAGATE_ASSIGN_OP(<<=)
  PROPAGATE_ASSIGN_OP(>>=)
  PROPAGATE_LOGICAL_OP(||)
  PROPAGATE_LOGICAL_OP(&&)
  PROPAGATE_LOGICAL_OP(<)
  PROPAGATE_LOGICAL_OP(>)
  PROPAGATE_LOGICAL_OP(==)
  PROPAGATE_LOGICAL_OP(!=)
  PROPAGATE_LOGICAL_OP(<=)
  PROPAGATE_LOGICAL_OP(>=)
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
  using reference =
      sycl::ext::oneapi::experimental::annotated_ref<T, property_list_t>;

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

  reference operator->() const { return reference(m_Ptr); }

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
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
