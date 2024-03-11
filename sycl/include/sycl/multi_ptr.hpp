//==------------ multi_ptr.hpp - SYCL multi_ptr class ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/__spirv/spirv_ops.hpp>           // for __spirv_ocl_prefetch
#include <sycl/access/access.hpp>             // for address_space, decorated
#include <sycl/aliases.hpp>                   // for half
#include <sycl/detail/defines_elementary.hpp> // for __SYCL2020_DEPRECATED
#include <sycl/detail/type_traits.hpp>        // for const_if_const_AS
#include <sycl/half_type.hpp>                 // for BIsRepresentationT

#include <cstddef>     // for nullptr_t, ptrdiff_t, size_t
#include <iterator>    // for random_access_iterator_tag
#include <type_traits> // for enable_if_t, add_pointer_t

namespace sycl {
inline namespace _V1 {

namespace detail {

// Helper to avoid instantiations of invalid non-legacy multi_ptr types.
template <typename ElementType, access::address_space Space>
struct LegacyPointerTypes {
  using pointer_t =
      typename multi_ptr<ElementType, Space, access::decorated::yes>::pointer;
  using const_pointer_t = typename multi_ptr<const ElementType, Space,
                                             access::decorated::yes>::pointer;
};

// Specialization for constant_space to avoid creating a non-legacy multi_ptr
// with the unsupported space.
template <typename ElementType>
struct LegacyPointerTypes<ElementType, access::address_space::constant_space> {
  using decorated_type = typename detail::DecoratedType<
      ElementType, access::address_space::constant_space>::type;
  using pointer_t = decorated_type *;
  using const_pointer_t = decorated_type const *;
};

// Helper to avoid instantiations of invalid non-legacy multi_ptr types.
template <typename ElementType, access::address_space Space>
struct LegacyReferenceTypes {
  using reference_t =
      typename multi_ptr<ElementType, Space, access::decorated::yes>::reference;
  using const_reference_t =
      typename multi_ptr<const ElementType, Space,
                         access::decorated::yes>::reference;
};

// Specialization for constant_space to avoid creating a non-legacy multi_ptr
// with the unsupported space.
template <typename ElementType>
struct LegacyReferenceTypes<ElementType,
                            access::address_space::constant_space> {
  using decorated_type = typename detail::DecoratedType<
      ElementType, access::address_space::constant_space>::type;
  using reference_t = decorated_type &;
  using const_reference_t = decorated_type &;
};
} // namespace detail

// Forward declarations
template <typename dataT, int dimensions, access::mode accessMode,
          access::target accessTarget, access::placeholder isPlaceholder,
          typename PropertyListT>
class accessor;
template <typename dataT, int dimensions> class local_accessor;

/// Provides constructors for address space qualified and non address space
/// qualified pointers to allow interoperability between plain C++ and OpenCL C.
///
/// \ingroup sycl_api
// TODO: Default value for DecorateAddress is for backwards compatiblity. It
//       should be removed.
template <typename ElementType, access::address_space Space,
          access::decorated DecorateAddress = access::decorated::legacy>
class __SYCL_TYPE(multi_ptr) multi_ptr {
private:
  using decorated_type =
      typename detail::DecoratedType<ElementType, Space>::type;

public:
  static constexpr bool is_decorated =
      DecorateAddress == access::decorated::yes;
  static constexpr access::address_space address_space = Space;

  using value_type = ElementType;
  using pointer = std::conditional_t<is_decorated, decorated_type *,
                                     std::add_pointer_t<value_type>>;
  using reference = std::conditional_t<is_decorated, decorated_type &,
                                       std::add_lvalue_reference_t<value_type>>;
  using iterator_category = std::random_access_iterator_tag;
  using difference_type = std::ptrdiff_t;

  static_assert(std::is_same_v<remove_decoration_t<pointer>,
                               std::add_pointer_t<value_type>>);
  static_assert(std::is_same_v<remove_decoration_t<reference>,
                               std::add_lvalue_reference_t<value_type>>);
  // Legacy has a different interface.
  static_assert(DecorateAddress != access::decorated::legacy);

  // Constructors
  multi_ptr() : m_Pointer(nullptr) {}
  multi_ptr(const multi_ptr &) = default;
  multi_ptr(multi_ptr &&) = default;
  explicit multi_ptr(typename multi_ptr<ElementType, Space,
                                        access::decorated::yes>::pointer ptr)
      : m_Pointer(ptr) {}
  multi_ptr(std::nullptr_t) : m_Pointer(nullptr) {}

  // Only if Space is in
  // {global_space, ext_intel_global_device_space, generic_space}
  template <
      int Dimensions, access::mode Mode, access::placeholder isPlaceholder,
      typename PropertyListT, access::address_space RelaySpace = Space,
      typename = typename std::enable_if_t<
          RelaySpace == Space &&
          (Space == access::address_space::generic_space ||
           Space == access::address_space::global_space ||
           Space == access::address_space::ext_intel_global_device_space)>>
  multi_ptr(accessor<ElementType, Dimensions, Mode, target::device,
                     isPlaceholder, PropertyListT>
                Accessor)
      : multi_ptr(detail::cast_AS<decorated_type *>(
            Accessor.template get_multi_ptr<DecorateAddress>()
                .get_decorated())) {}

  // Only if Space == local_space || generic_space
  template <int Dimensions, access::mode Mode,
            access::placeholder isPlaceholder, typename PropertyListT,
            access::address_space RelaySpace = Space,
            typename = typename std::enable_if_t<
                RelaySpace == Space &&
                (Space == access::address_space::generic_space ||
                 Space == access::address_space::local_space)>>
  __SYCL2020_DEPRECATED(
      "multi_ptr construction using target::local specialized "
      "accessor is deprecated since SYCL 2020")
  multi_ptr(accessor<ElementType, Dimensions, Mode, target::local,
                     isPlaceholder, PropertyListT>
                Accessor)
      : multi_ptr(Accessor.get_pointer().get()) {}

  // Only if Space == local_space || generic_space
  template <int Dimensions, access::address_space RelaySpace = Space,
            typename = typename std::enable_if_t<
                RelaySpace == Space &&
                (Space == access::address_space::generic_space ||
                 Space == access::address_space::local_space)>>
  multi_ptr(local_accessor<ElementType, Dimensions> Accessor)
      : multi_ptr(Accessor.template get_multi_ptr<DecorateAddress>()
                      .get_decorated()) {}

  // Only if Space == constant_space
  template <
      int dimensions, access::placeholder isPlaceholder, typename PropertyListT,
      access::address_space _Space = Space,
      typename = typename std::enable_if_t<
          _Space == Space && Space == access::address_space::constant_space>>
  __SYCL2020_DEPRECATED(
      "multi_ptr construction using target::constant_buffer specialized "
      "accessor is deprecated since SYCL 2020")
  multi_ptr(accessor<ElementType, dimensions, access_mode::read,
                     target::constant_buffer, isPlaceholder, PropertyListT>
                Accessor)
      : multi_ptr(Accessor.template get_multi_ptr<DecorateAddress>()
                      .get_decorated()) {}

  // The following constructors are necessary to create multi_ptr<const
  // ElementType, Space, DecorateAddress> from accessor<ElementType, ...>.
  // Constructors above could not be used for this purpose because it will
  // require 2 implicit conversions of user types which is not allowed by C++:
  //    1. from accessor<ElementType, ...> to
  //       multi_ptr<ElementType, Space, DecorateAddress>
  //    2. from multi_ptr<ElementType, Space, DecorateAddress> to
  //       multi_ptr<const ElementType, Space, DecorateAddress>

  // Only if Space is in
  // {global_space, ext_intel_global_device_space, generic_space} and element
  // type is const
  template <
      int Dimensions, access::mode Mode, access::placeholder isPlaceholder,
      typename PropertyListT, access::address_space _Space = Space,
      typename RelayElementType = ElementType,
      typename = typename std::enable_if_t<
          _Space == Space &&
          (Space == access::address_space::generic_space ||
           Space == access::address_space::global_space ||
           Space == access::address_space::ext_intel_global_device_space) &&
          std::is_const_v<RelayElementType> &&
          std::is_same_v<RelayElementType, ElementType>>>
  multi_ptr(accessor<typename std::remove_const_t<RelayElementType>, Dimensions,
                     Mode, target::device, isPlaceholder, PropertyListT>
                Accessor)
      : m_Pointer(detail::cast_AS<decorated_type *>(
            Accessor.template get_multi_ptr<DecorateAddress>()
                .get_decorated())) {}

  // Only if Space == local_space || generic_space and element type is const
  template <int Dimensions, access::mode Mode,
            access::placeholder isPlaceholder, typename PropertyListT,
            access::address_space RelaySpace = Space,
            typename RelayElementType = ElementType,
            typename = typename std::enable_if_t<
                RelaySpace == Space &&
                (Space == access::address_space::generic_space ||
                 Space == access::address_space::local_space) &&
                std::is_const_v<RelayElementType> &&
                std::is_same_v<RelayElementType, ElementType>>>
  __SYCL2020_DEPRECATED(
      "multi_ptr construction using target::local specialized "
      "accessor is deprecated since SYCL 2020")
  multi_ptr(accessor<typename std::remove_const_t<RelayElementType>, Dimensions,
                     Mode, target::local, isPlaceholder, PropertyListT>
                Accessor)
      : multi_ptr(Accessor.get_pointer().get()) {}

  // Only if Space == local_space || generic_space and element type is const
  template <int Dimensions, access::address_space RelaySpace = Space,
            typename RelayElementType = ElementType,
            typename = typename std::enable_if_t<
                RelaySpace == Space &&
                (Space == access::address_space::generic_space ||
                 Space == access::address_space::local_space) &&
                std::is_const_v<RelayElementType> &&
                std::is_same_v<RelayElementType, ElementType>>>
  multi_ptr(
      local_accessor<typename std::remove_const_t<RelayElementType>, Dimensions>
          Accessor)
      // Not having get_decorated() results in facing issue represented in
      // https://github.com/intel/llvm/issues/9745.
      // TODO: would be good to simplify it in future without facing above issue
      : multi_ptr(Accessor.template get_multi_ptr<DecorateAddress>()
                      .get_decorated()) {}

  // Only if Space == constant_space and element type is const
  template <
      int dimensions, access::placeholder isPlaceholder, typename PropertyListT,
      access::address_space _Space = Space,
      typename RelayElementType = ElementType,
      typename = typename std::enable_if_t<
          _Space == Space && Space == access::address_space::constant_space &&
          std::is_const_v<RelayElementType> &&
          std::is_same_v<RelayElementType, ElementType>>>
  __SYCL2020_DEPRECATED(
      "multi_ptr construction using target::constant_buffer specialized "
      "accessor is deprecated since SYCL 2020")
  multi_ptr(accessor<typename std::remove_const_t<RelayElementType>, dimensions,
                     access_mode::read, target::constant_buffer, isPlaceholder,
                     PropertyListT>
                Accessor)
      : multi_ptr(Accessor.template get_multi_ptr<DecorateAddress>()
                      .get_decorated()) {}

  // Assignment and access operators
  multi_ptr &operator=(const multi_ptr &) = default;
  multi_ptr &operator=(multi_ptr &&) = default;
  multi_ptr &operator=(std::nullptr_t) {
    m_Pointer = nullptr;
    return *this;
  }
  template <
      access::address_space OtherSpace, access::decorated OtherIsDecorated,
      typename =
          std::enable_if_t<Space == access::address_space::generic_space &&
                           OtherSpace != access::address_space::constant_space>>
  multi_ptr &
  operator=(const multi_ptr<value_type, OtherSpace, OtherIsDecorated> &Other) {
    m_Pointer = detail::cast_AS<decorated_type *>(Other.get_decorated());
    return *this;
  }
  template <
      access::address_space OtherSpace, access::decorated OtherIsDecorated,
      typename =
          std::enable_if_t<Space == access::address_space::generic_space &&
                           OtherSpace != access::address_space::constant_space>>
  multi_ptr &
  operator=(multi_ptr<value_type, OtherSpace, OtherIsDecorated> &&Other) {
    m_Pointer = detail::cast_AS<decorated_type *>(std::move(Other.m_Pointer));
    return *this;
  }

  reference operator*() const { return *m_Pointer; }
  pointer operator->() const { return get(); }
  reference operator[](difference_type index) const { return m_Pointer[index]; }

  pointer get() const { return detail::cast_AS<pointer>(m_Pointer); }
  decorated_type *get_decorated() const { return m_Pointer; }
  std::add_pointer_t<value_type> get_raw() const {
    return reinterpret_cast<std::add_pointer_t<value_type>>(get());
  }

  __SYCL2020_DEPRECATED("Conversion to pointer type is deprecated since SYCL "
                        "2020. Please use get() instead.")
  operator pointer() const { return get(); }

  template <access::address_space OtherSpace,
            access::decorated OtherIsDecorated,
            access::address_space RelaySpace = Space,
            typename = typename std::enable_if_t<
                RelaySpace == Space &&
                RelaySpace == access::address_space::generic_space &&
                (OtherSpace == access::address_space::private_space ||
                 OtherSpace == access::address_space::global_space ||
                 OtherSpace == access::address_space::local_space)>>
  explicit
  operator multi_ptr<value_type, OtherSpace, OtherIsDecorated>() const {
    return multi_ptr<value_type, OtherSpace, OtherIsDecorated>{
        detail::cast_AS<typename multi_ptr<value_type, OtherSpace,
                                           access::decorated::yes>::pointer>(
            get_decorated())};
  }

  template <access::address_space OtherSpace,
            access::decorated OtherIsDecorated,
            typename RelayElementType = ElementType,
            access::address_space RelaySpace = Space,
            typename = typename std::enable_if_t<
                std::is_same_v<RelayElementType, ElementType> &&
                !std::is_const_v<RelayElementType> && RelaySpace == Space &&
                RelaySpace == access::address_space::generic_space &&
                (OtherSpace == access::address_space::private_space ||
                 OtherSpace == access::address_space::global_space ||
                 OtherSpace == access::address_space::local_space)>>
  explicit
  operator multi_ptr<const value_type, OtherSpace, OtherIsDecorated>() const {
    return multi_ptr<const value_type, OtherSpace, OtherIsDecorated>{
        detail::cast_AS<typename multi_ptr<const value_type, OtherSpace,
                                           access::decorated::yes>::pointer>(
            get_decorated())};
  }

  template <access::decorated ConvIsDecorated,
            typename RelayElementType = ElementType,
            typename = typename std::enable_if_t<
                std::is_same_v<RelayElementType, ElementType> &&
                !std::is_const_v<RelayElementType>>>
  operator multi_ptr<void, Space, ConvIsDecorated>() const {
    return multi_ptr<void, Space, ConvIsDecorated>{detail::cast_AS<
        typename multi_ptr<void, Space, access::decorated::yes>::pointer>(
        get_decorated())};
  }

  template <access::decorated ConvIsDecorated,
            typename RelayElementType = ElementType,
            typename = typename std::enable_if_t<
                std::is_same_v<RelayElementType, ElementType> &&
                std::is_const_v<RelayElementType>>>
  operator multi_ptr<const void, Space, ConvIsDecorated>() const {
    return multi_ptr<const void, Space, ConvIsDecorated>{detail::cast_AS<
        typename multi_ptr<const void, Space, access::decorated::yes>::pointer>(
        get_decorated())};
  }

  template <access::decorated ConvIsDecorated>
  operator multi_ptr<const value_type, Space, ConvIsDecorated>() const {
    return multi_ptr<const value_type, Space, ConvIsDecorated>{
        detail::cast_AS<typename multi_ptr<const value_type, Space,
                                           access::decorated::yes>::pointer>(
            get_decorated())};
  }

  operator multi_ptr<value_type, Space,
                     detail::NegateDecorated<DecorateAddress>::value>() const {
    return multi_ptr<value_type, Space,
                     detail::NegateDecorated<DecorateAddress>::value>{
        get_decorated()};
  }

  // Explicit conversion to global_space
  // Only available if Space == address_space::ext_intel_global_device_space ||
  // Space == address_space::ext_intel_global_host_space
  template <
      access::address_space GlobalSpace = access::address_space::global_space,
      access::address_space RelaySpace = Space,
      typename = typename std::enable_if_t<
          RelaySpace == Space &&
          GlobalSpace == access::address_space::global_space &&
          (Space == access::address_space::ext_intel_global_device_space ||
           Space == access::address_space::ext_intel_global_host_space)>>
  explicit
  operator multi_ptr<ElementType, GlobalSpace, DecorateAddress>() const {
    using global_pointer_t =
        typename multi_ptr<ElementType, GlobalSpace,
                           access::decorated::yes>::pointer;
    return multi_ptr<ElementType, GlobalSpace, DecorateAddress>(
        detail::cast_AS<global_pointer_t>(get_decorated()));
  }

  // Only if Space == global_space
  template <
      access::address_space _Space = Space,
      typename = typename std::enable_if_t<
          _Space == Space && Space == access::address_space::global_space>>
  void prefetch(size_t NumElements) const {
    size_t NumBytes = NumElements * sizeof(ElementType);
    using ptr_t = typename detail::DecoratedType<char, Space>::type const *;
    __spirv_ocl_prefetch(reinterpret_cast<ptr_t>(get_decorated()), NumBytes);
  }

  // Arithmetic operators
  multi_ptr &operator++() {
    m_Pointer += (difference_type)1;
    return *this;
  }
  multi_ptr operator++(int) {
    multi_ptr result(*this);
    ++(*this);
    return result;
  }
  multi_ptr &operator--() {
    m_Pointer -= (difference_type)1;
    return *this;
  }
  multi_ptr operator--(int) {
    multi_ptr result(*this);
    --(*this);
    return result;
  }
  multi_ptr &operator+=(difference_type r) {
    m_Pointer += r;
    return *this;
  }
  multi_ptr &operator-=(difference_type r) {
    m_Pointer -= r;
    return *this;
  }
  multi_ptr operator+(difference_type r) const {
    return multi_ptr(get_decorated() + r);
  }
  multi_ptr operator-(difference_type r) const {
    return multi_ptr(get_decorated() - r);
  }

private:
  decorated_type *m_Pointer;
};

/// Specialization of multi_ptr for const void.
template <access::address_space Space, access::decorated DecorateAddress>
class __SYCL_TYPE(multi_ptr) multi_ptr<const void, Space, DecorateAddress> {
private:
  using decorated_type =
      typename detail::DecoratedType<const void, Space>::type;

public:
  static constexpr bool is_decorated =
      DecorateAddress == access::decorated::yes;
  static constexpr access::address_space address_space = Space;

  using value_type = const void;
  using pointer = std::conditional_t<is_decorated, decorated_type *,
                                     std::add_pointer_t<value_type>>;
  using difference_type = std::ptrdiff_t;

  static_assert(std::is_same_v<remove_decoration_t<pointer>,
                               std::add_pointer_t<value_type>>);
  // Legacy has a different interface.
  static_assert(DecorateAddress != access::decorated::legacy);

  // Constructors
  multi_ptr() : m_Pointer(nullptr) {}
  multi_ptr(const multi_ptr &) = default;
  multi_ptr(multi_ptr &&) = default;
  explicit multi_ptr(typename multi_ptr<const void, Space,
                                        access::decorated::yes>::pointer ptr)
      : m_Pointer(ptr) {}
  multi_ptr(std::nullptr_t) : m_Pointer(nullptr) {}

  // Only if Space is in
  // {global_space, ext_intel_global_device_space}
  template <
      typename ElementType, int Dimensions, access::mode Mode,
      access::placeholder isPlaceholder, typename PropertyListT,
      access::address_space RelaySpace = Space,
      typename = typename std::enable_if_t<
          RelaySpace == Space &&
          (Space == access::address_space::global_space ||
           Space == access::address_space::generic_space ||
           Space == access::address_space::ext_intel_global_device_space)>>
  multi_ptr(accessor<ElementType, Dimensions, Mode, target::device,
                     isPlaceholder, PropertyListT>
                Accessor)
      : multi_ptr(detail::cast_AS<decorated_type *>(
            Accessor.template get_multi_ptr<DecorateAddress>()
                .get_decorated())) {}

  // Only if Space == local_space
  template <
      typename ElementType, int Dimensions, access::mode Mode,
      access::placeholder isPlaceholder, typename PropertyListT,
      access::address_space RelaySpace = Space,
      typename = typename std::enable_if_t<
          RelaySpace == Space && Space == access::address_space::local_space>>
  __SYCL2020_DEPRECATED(
      "multi_ptr construction using target::local specialized "
      "accessor is deprecated since SYCL 2020")
  multi_ptr(accessor<ElementType, Dimensions, Mode, target::local,
                     isPlaceholder, PropertyListT>
                Accessor)
      : multi_ptr(Accessor.get_pointer().get()) {}

  // Only if Space == local_space
  template <typename ElementType, int Dimensions,
            access::address_space RelaySpace = Space,
            typename = typename std::enable_if_t<
                RelaySpace == Space &&
                (Space == access::address_space::local_space ||
                 Space == access::address_space::generic_space)>>
  multi_ptr(local_accessor<ElementType, Dimensions> Accessor)
      : multi_ptr(Accessor.template get_multi_ptr<DecorateAddress>()
                      .get_decorated()) {}

  // Only if Space == constant_space
  template <
      typename ElementType, int dimensions, typename PropertyListT,
      access::address_space _Space = Space,
      typename = typename std::enable_if_t<
          _Space == Space && Space == access::address_space::constant_space>>
  __SYCL2020_DEPRECATED(
      "multi_ptr construction using target::constant_buffer specialized "
      "accessor is deprecated since SYCL 2020")
  multi_ptr(accessor<ElementType, dimensions, access_mode::read,
                     target::constant_buffer, access::placeholder::false_t,
                     PropertyListT>
                Accessor)
      : multi_ptr(Accessor.template get_multi_ptr<DecorateAddress>()
                      .get_decorated()) {}

  // Assignment operators
  multi_ptr &operator=(const multi_ptr &) = default;
  multi_ptr &operator=(multi_ptr &&) = default;
  multi_ptr &operator=(std::nullptr_t) {
    m_Pointer = nullptr;
    return *this;
  }

  pointer get() const { return detail::cast_AS<pointer>(m_Pointer); }

  // Conversion to the underlying pointer type
  __SYCL2020_DEPRECATED("Conversion to pointer type is deprecated since SYCL "
                        "2020. Please use get() instead.")
  operator pointer() const { return get(); }

  // Explicit conversion to a multi_ptr<ElementType>
  template <typename ElementType,
            typename = typename std::enable_if_t<std::is_const_v<ElementType>>>
  explicit operator multi_ptr<ElementType, Space, DecorateAddress>() const {
    return multi_ptr<ElementType, Space, DecorateAddress>{
        detail::cast_AS<typename multi_ptr<ElementType, Space,
                                           access::decorated::yes>::pointer>(
            m_Pointer)};
  }

  // Implicit conversion to the negated decoration version of multi_ptr.
  operator multi_ptr<value_type, Space,
                     detail::NegateDecorated<DecorateAddress>::value>() const {
    return multi_ptr<value_type, Space,
                     detail::NegateDecorated<DecorateAddress>::value>{
        m_Pointer};
  }

  // Explicit conversion to global_space
  // Only available if Space == address_space::ext_intel_global_device_space ||
  // Space == address_space::ext_intel_global_host_space
  template <
      access::address_space GlobalSpace = access::address_space::global_space,
      access::address_space RelaySpace = Space,
      typename = typename std::enable_if_t<
          RelaySpace == Space &&
          GlobalSpace == access::address_space::global_space &&
          (Space == access::address_space::ext_intel_global_device_space ||
           Space == access::address_space::ext_intel_global_host_space)>>
  explicit
  operator multi_ptr<const void, GlobalSpace, DecorateAddress>() const {
    using global_pointer_t =
        typename multi_ptr<const void, GlobalSpace,
                           access::decorated::yes>::pointer;
    return multi_ptr<const void, GlobalSpace, DecorateAddress>(
        detail::cast_AS<global_pointer_t>(m_Pointer));
  }

private:
  decorated_type *m_Pointer;
};

// Specialization of multi_ptr for void.
template <access::address_space Space, access::decorated DecorateAddress>
class __SYCL_TYPE(multi_ptr) multi_ptr<void, Space, DecorateAddress> {
private:
  using decorated_type = typename detail::DecoratedType<void, Space>::type;

public:
  static constexpr bool is_decorated =
      DecorateAddress == access::decorated::yes;
  static constexpr access::address_space address_space = Space;

  using value_type = void;
  using pointer = std::conditional_t<is_decorated, decorated_type *,
                                     std::add_pointer_t<value_type>>;
  using difference_type = std::ptrdiff_t;

  static_assert(std::is_same_v<remove_decoration_t<pointer>,
                               std::add_pointer_t<value_type>>);
  // Legacy has a different interface.
  static_assert(DecorateAddress != access::decorated::legacy);
  // constant_space is only supported in legacy multi_ptr.
  static_assert(Space != access::address_space::constant_space,
                "SYCL 2020 multi_ptr does not support the deprecated "
                "constant_space address space.");

  // Constructors
  multi_ptr() : m_Pointer(nullptr) {}
  multi_ptr(const multi_ptr &) = default;
  multi_ptr(multi_ptr &&) = default;
  explicit multi_ptr(
      typename multi_ptr<void, Space, access::decorated::yes>::pointer ptr)
      : m_Pointer(ptr) {}
  multi_ptr(std::nullptr_t) : m_Pointer(nullptr) {}

  // Only if Space is in
  // {global_space, ext_intel_global_device_space}
  template <
      typename ElementType, int Dimensions, access::mode Mode,
      access::placeholder isPlaceholder, typename PropertyListT,
      access::address_space RelaySpace = Space,
      typename = typename std::enable_if_t<
          RelaySpace == Space &&
          (Space == access::address_space::global_space ||
           Space == access::address_space::generic_space ||
           Space == access::address_space::ext_intel_global_device_space)>>
  multi_ptr(accessor<ElementType, Dimensions, Mode, target::device,
                     isPlaceholder, PropertyListT>
                Accessor)
      : multi_ptr(detail::cast_AS<decorated_type *>(
            Accessor.template get_multi_ptr<DecorateAddress>()
                .get_decorated())) {}

  // Only if Space == local_space
  template <
      typename ElementType, int Dimensions, access::mode Mode,
      access::placeholder isPlaceholder, typename PropertyListT,
      access::address_space RelaySpace = Space,
      typename = typename std::enable_if_t<
          RelaySpace == Space && Space == access::address_space::local_space>>
  __SYCL2020_DEPRECATED(
      "multi_ptr construction using target::local specialized "
      "accessor is deprecated since SYCL 2020")
  multi_ptr(accessor<ElementType, Dimensions, Mode, target::local,
                     isPlaceholder, PropertyListT>
                Accessor)
      : multi_ptr(Accessor.get_pointer().get()) {}

  // Only if Space == local_space
  template <typename ElementType, int Dimensions,
            access::address_space RelaySpace = Space,
            typename = typename std::enable_if_t<
                RelaySpace == Space &&
                (Space == access::address_space::local_space ||
                 Space == access::address_space::generic_space)>>
  multi_ptr(local_accessor<ElementType, Dimensions> Accessor)
      : multi_ptr(Accessor.template get_multi_ptr<DecorateAddress>()
                      .get_decorated()) {}

  // Only if Space == constant_space
  template <
      typename ElementType, int dimensions, typename PropertyListT,
      access::address_space _Space = Space,
      typename = typename std::enable_if_t<
          _Space == Space && Space == access::address_space::constant_space>>
  __SYCL2020_DEPRECATED(
      "multi_ptr construction using target::constant_buffer specialized "
      "accessor is deprecated since SYCL 2020")
  multi_ptr(accessor<ElementType, dimensions, access_mode::read,
                     target::constant_buffer, access::placeholder::false_t,
                     PropertyListT>
                Accessor)
      : multi_ptr(Accessor.template get_multi_ptr<DecorateAddress>()
                      .get_decorated()) {}

  // Assignment operators
  multi_ptr &operator=(const multi_ptr &) = default;
  multi_ptr &operator=(multi_ptr &&) = default;
  multi_ptr &operator=(std::nullptr_t) {
    m_Pointer = nullptr;
    return *this;
  }

  pointer get() const { return detail::cast_AS<pointer>(m_Pointer); }

  // Conversion to the underlying pointer type
  __SYCL2020_DEPRECATED("Conversion to pointer type is deprecated since SYCL "
                        "2020. Please use get() instead.")
  operator pointer() const { return get(); }

  // Explicit conversion to a multi_ptr<ElementType>
  template <typename ElementType>
  explicit operator multi_ptr<ElementType, Space, DecorateAddress>() const {
    return multi_ptr<ElementType, Space, DecorateAddress>{
        detail::cast_AS<typename multi_ptr<ElementType, Space,
                                           access::decorated::yes>::pointer>(
            m_Pointer)};
  }

  // Implicit conversion to the negated decoration version of multi_ptr.
  operator multi_ptr<value_type, Space,
                     detail::NegateDecorated<DecorateAddress>::value>() const {
    return multi_ptr<value_type, Space,
                     detail::NegateDecorated<DecorateAddress>::value>{
        m_Pointer};
  }

  // Explicit conversion to global_space
  // Only available if Space == address_space::ext_intel_global_device_space ||
  // Space == address_space::ext_intel_global_host_space
  template <
      access::address_space GlobalSpace = access::address_space::global_space,
      access::address_space RelaySpace = Space,
      typename = typename std::enable_if_t<
          RelaySpace == Space &&
          GlobalSpace == access::address_space::global_space &&
          (Space == access::address_space::ext_intel_global_device_space ||
           Space == access::address_space::ext_intel_global_host_space)>>
  explicit operator multi_ptr<void, GlobalSpace, DecorateAddress>() const {
    using global_pointer_t =
        typename multi_ptr<void, GlobalSpace, access::decorated::yes>::pointer;
    return multi_ptr<void, GlobalSpace, DecorateAddress>(
        detail::cast_AS<global_pointer_t>(m_Pointer));
  }

private:
  decorated_type *m_Pointer;
};

// Legacy specialization of multi_ptr.
// TODO: Add deprecation warning here when possible.
template <typename ElementType, access::address_space Space>
class __SYCL2020_DEPRECATED(
    "decorated::legacy multi_ptr specialization is deprecated since SYCL 2020.")
    multi_ptr<ElementType, Space, access::decorated::legacy> {
public:
  using value_type = ElementType;
  using element_type =
      std::conditional_t<std::is_same_v<ElementType, half>,
                         sycl::detail::half_impl::BIsRepresentationT,
                         ElementType>;
  using difference_type = std::ptrdiff_t;

  // Implementation defined pointer and reference types that correspond to
  // SYCL/OpenCL interoperability types for OpenCL C functions
  using pointer_t =
      typename detail::LegacyPointerTypes<ElementType, Space>::pointer_t;
  using const_pointer_t =
      typename detail::LegacyPointerTypes<ElementType, Space>::const_pointer_t;
  using reference_t =
      typename detail::LegacyReferenceTypes<ElementType, Space>::reference_t;
  using const_reference_t =
      typename detail::LegacyReferenceTypes<ElementType,
                                            Space>::const_reference_t;

  static constexpr access::address_space address_space = Space;

  // Constructors
  multi_ptr() : m_Pointer(nullptr) {}
  multi_ptr(const multi_ptr &rhs) = default;
  multi_ptr(multi_ptr &&) = default;
#ifdef __SYCL_DEVICE_ONLY__
  // The generic address space have no corresponding 'opencl_...' attribute and
  // this constructor is considered as a duplicate for the
  // multi_ptr(ElementType *pointer) one, so the check is required.
  template <
      access::address_space _Space = Space,
      typename = typename std::enable_if_t<
          _Space == Space && Space != access::address_space::generic_space>>
  multi_ptr(pointer_t pointer) : m_Pointer(pointer) {}
#endif

  multi_ptr(ElementType *pointer)
      : m_Pointer(detail::cast_AS<pointer_t>(pointer)) {
    // TODO An implementation should reject an argument if the deduced
    // address space is not compatible with Space.
  }
#if defined(RESTRICT_WRITE_ACCESS_TO_CONSTANT_PTR)
  template <typename = typename detail::const_if_const_AS<Space, ElementType>>
  multi_ptr(const ElementType *pointer)
      : m_Pointer(detail::cast_AS<pointer_t>(pointer)) {}
#endif

  multi_ptr(std::nullptr_t) : m_Pointer(nullptr) {}
  ~multi_ptr() = default;

  // Assignment and access operators
  multi_ptr &operator=(const multi_ptr &) = default;
  multi_ptr &operator=(multi_ptr &&) = default;

#ifdef __SYCL_DEVICE_ONLY__
  // The generic address space have no corresponding 'opencl_...' attribute and
  // this operator is considered as a duplicate for the
  // multi_ptr &operator=(ElementType *pointer) one, so the check is required.
  template <
      access::address_space _Space = Space,
      typename = typename std::enable_if_t<
          _Space == Space && Space != access::address_space::generic_space>>
  multi_ptr &operator=(pointer_t pointer) {
    m_Pointer = pointer;
    return *this;
  }
#endif

  multi_ptr &operator=(ElementType *pointer) {
    // TODO An implementation should reject an argument if the deduced
    // address space is not compatible with Space.
    m_Pointer = detail::cast_AS<pointer_t>(pointer);
    return *this;
  }

  multi_ptr &operator=(std::nullptr_t) {
    m_Pointer = nullptr;
    return *this;
  }

  using ReturnPtr = detail::const_if_const_AS<Space, ElementType> *;
  using ReturnRef = detail::const_if_const_AS<Space, ElementType> &;
  using ReturnConstRef = const ElementType &;

  ReturnRef operator*() const {
    return *reinterpret_cast<ReturnPtr>(m_Pointer);
  }

  ReturnPtr operator->() const {
    return reinterpret_cast<ReturnPtr>(m_Pointer);
  }

  ReturnRef operator[](difference_type index) {
    return reinterpret_cast<ReturnPtr>(m_Pointer)[index];
  }

  ReturnConstRef operator[](difference_type index) const {
    return reinterpret_cast<ReturnPtr>(m_Pointer)[index];
  }

  // Only if Space is in
  // {global_space, ext_intel_global_device_space, generic_space}
  template <
      int dimensions, access::mode Mode, access::placeholder isPlaceholder,
      typename PropertyListT, access::address_space _Space = Space,
      typename = typename std::enable_if_t<
          _Space == Space &&
          (Space == access::address_space::generic_space ||
           Space == access::address_space::global_space ||
           Space == access::address_space::ext_intel_global_device_space)>>
  multi_ptr(accessor<ElementType, dimensions, Mode, target::device,
                     isPlaceholder, PropertyListT>
                Accessor)
      : multi_ptr(detail::cast_AS<pointer_t>(Accessor.get_pointer().get())) {}

  // Only if Space == local_space || generic_space
  template <
      int dimensions, access::mode Mode, access::placeholder isPlaceholder,
      typename PropertyListT, access::address_space _Space = Space,
      typename = typename std::enable_if_t<
          _Space == Space && (Space == access::address_space::generic_space ||
                              Space == access::address_space::local_space)>>
  multi_ptr(accessor<ElementType, dimensions, Mode, target::local,
                     isPlaceholder, PropertyListT>
                Accessor)
      : multi_ptr(Accessor.get_pointer()) {}

  // Only if Space == local_space || generic_space
  template <int dimensions>
  multi_ptr(local_accessor<ElementType, dimensions> Accessor)
      : multi_ptr(Accessor.get_pointer()) {}

  // Only if Space == constant_space
  template <
      int dimensions, access::mode Mode, access::placeholder isPlaceholder,
      typename PropertyListT, access::address_space _Space = Space,
      typename = typename std::enable_if_t<
          _Space == Space && Space == access::address_space::constant_space>>
  multi_ptr(accessor<ElementType, dimensions, Mode, target::constant_buffer,
                     isPlaceholder, PropertyListT>
                Accessor)
      : multi_ptr(Accessor.get_pointer()) {}

  // The following constructors are necessary to create multi_ptr<const
  // ElementType, Space, access::decorated::legacy> from
  // accessor<ElementType, ...>. Constructors above could not be used for this
  // purpose because it will require 2 implicit conversions of user types which
  // is not allowed by C++:
  //    1. from accessor<ElementType, ...> to
  //       multi_ptr<ElementType, Space, access::decorated::legacy>
  //    2. from multi_ptr<ElementType, Space, access::decorated::legacy> to
  //       multi_ptr<const ElementType, Space, access::decorated::legacy>

  // Only if Space is in
  // {global_space, ext_intel_global_device_space, generic_space} and element
  // type is const
  template <
      int dimensions, access::mode Mode, access::placeholder isPlaceholder,
      typename PropertyListT, access::address_space _Space = Space,
      typename ET = ElementType,
      typename = typename std::enable_if_t<
          _Space == Space &&
          (Space == access::address_space::generic_space ||
           Space == access::address_space::global_space ||
           Space == access::address_space::ext_intel_global_device_space) &&
          std::is_const_v<ET> && std::is_same_v<ET, ElementType>>>
  multi_ptr(accessor<typename std::remove_const_t<ET>, dimensions, Mode,
                     target::device, isPlaceholder, PropertyListT>
                Accessor)
      : multi_ptr(Accessor.get_pointer()) {}

  // Only if Space == local_space || generic_space and element type is const
  template <int dimensions, access::mode Mode,
            access::placeholder isPlaceholder, typename PropertyListT,
            access::address_space _Space = Space, typename ET = ElementType,
            typename = typename std::enable_if_t<
                _Space == Space &&
                (Space == access::address_space::generic_space ||
                 Space == access::address_space::local_space) &&
                std::is_const_v<ET> && std::is_same_v<ET, ElementType>>>
  multi_ptr(accessor<typename std::remove_const_t<ET>, dimensions, Mode,
                     target::local, isPlaceholder, PropertyListT>
                Accessor)
      : multi_ptr(Accessor.get_pointer()) {}

  // Only if Space == local_space || generic_space and element type is const
  template <int dimensions, access::address_space _Space = Space,
            typename ET = ElementType,
            typename = typename std::enable_if_t<
                _Space == Space &&
                (Space == access::address_space::generic_space ||
                 Space == access::address_space::local_space) &&
                std::is_const_v<ET> && std::is_same_v<ET, ElementType>>>
  multi_ptr(
      local_accessor<typename std::remove_const_t<ET>, dimensions> Accessor)
      : multi_ptr(Accessor.get_pointer()) {}

  // Only if Space == constant_space and element type is const
  template <
      int dimensions, access::mode Mode, access::placeholder isPlaceholder,
      typename PropertyListT, access::address_space _Space = Space,
      typename ET = ElementType,
      typename = typename std::enable_if_t<
          _Space == Space && Space == access::address_space::constant_space &&
          std::is_const_v<ET> && std::is_same_v<ET, ElementType>>>
  multi_ptr(accessor<typename std::remove_const_t<ET>, dimensions, Mode,
                     target::constant_buffer, isPlaceholder, PropertyListT>
                Accessor)
      : multi_ptr(Accessor.get_pointer()) {}

  // TODO: This constructor is the temporary solution for the existing problem
  // with conversions from multi_ptr<ElementType, Space,
  // access::decorated::legacy> to multi_ptr<const ElementType, Space,
  // access::decorated::legacy>. Without it the compiler fails due to having 3
  // different same rank paths available.
  // Constructs multi_ptr<const ElementType, Space, access::decorated::legacy>:
  //   multi_ptr<ElementType, Space, access::decorated::legacy> ->
  //     multi_ptr<const ElementTYpe, Space, access::decorated::legacy>
  template <typename ET = ElementType>
  multi_ptr(typename std::enable_if_t<
            std::is_const_v<ET> && std::is_same_v<ET, ElementType>,
            const multi_ptr<typename std::remove_const_t<ET>, Space,
                            access::decorated::legacy>> &ETP)
      : m_Pointer(ETP.get()) {}

  // Returns the underlying OpenCL C pointer
  pointer_t get() const { return m_Pointer; }
  pointer_t get_decorated() const { return m_Pointer; }
  std::add_pointer_t<element_type> get_raw() const {
    return reinterpret_cast<std::add_pointer_t<element_type>>(get());
  }

  // Implicit conversion to the underlying pointer type
  operator ReturnPtr() const { return reinterpret_cast<ReturnPtr>(m_Pointer); }

  // Implicit conversion to a multi_ptr<void>
  // Only available when ElementType is not const-qualified
  template <typename ET = ElementType>
  operator multi_ptr<
      typename std::enable_if_t<
          std::is_same_v<ET, ElementType> && !std::is_const_v<ET>, void>::type,
      Space, access::decorated::legacy>() const {
    using ptr_t = typename detail::DecoratedType<void, Space> *;
    return multi_ptr<void, Space, access::decorated::legacy>(
        reinterpret_cast<ptr_t>(m_Pointer));
  }

  // Implicit conversion to a multi_ptr<const void>
  // Only available when ElementType is const-qualified
  template <typename ET = ElementType>
  operator multi_ptr<typename std::enable_if_t<
                         std::is_same_v<ET, ElementType> && std::is_const_v<ET>,
                         const void>::type,
                     Space, access::decorated::legacy>() const {
    using ptr_t = typename detail::DecoratedType<const void, Space> *;
    return multi_ptr<const void, Space, access::decorated::legacy>(
        reinterpret_cast<ptr_t>(m_Pointer));
  }

  // Implicit conversion to multi_ptr<const ElementType, Space,
  // access::decorated::legacy>
  operator multi_ptr<const ElementType, Space, access::decorated::legacy>()
      const {
    using ptr_t =
        typename detail::DecoratedType<const ElementType, Space>::type *;
    return multi_ptr<const ElementType, Space, access::decorated::legacy>(
        reinterpret_cast<ptr_t>(m_Pointer));
  }

  // Arithmetic operators
  multi_ptr &operator++() {
    m_Pointer += (difference_type)1;
    return *this;
  }
  multi_ptr operator++(int) {
    multi_ptr result(*this);
    ++(*this);
    return result;
  }
  multi_ptr &operator--() {
    m_Pointer -= (difference_type)1;
    return *this;
  }
  multi_ptr operator--(int) {
    multi_ptr result(*this);
    --(*this);
    return result;
  }
  multi_ptr &operator+=(difference_type r) {
    m_Pointer += r;
    return *this;
  }
  multi_ptr &operator-=(difference_type r) {
    m_Pointer -= r;
    return *this;
  }
  multi_ptr operator+(difference_type r) const {
    return multi_ptr(m_Pointer + r);
  }
  multi_ptr operator-(difference_type r) const {
    return multi_ptr(m_Pointer - r);
  }

#ifdef __ENABLE_USM_ADDR_SPACE__
  // Explicit conversion to global_space
  // Only available if Space == address_space::ext_intel_global_device_space ||
  // Space == address_space::ext_intel_global_host_space
  template <
      access::address_space _Space = Space,
      typename = typename std::enable_if_t<
          _Space == Space &&
          (Space == access::address_space::ext_intel_global_device_space ||
           Space == access::address_space::ext_intel_global_host_space)>>
  explicit operator multi_ptr<ElementType, access::address_space::global_space,
                              access::decorated::legacy>() const {
    using global_pointer_t = typename detail::DecoratedType<
        ElementType, access::address_space::global_space>::type *;
    return multi_ptr<ElementType, access::address_space::global_space,
                     access::decorated::legacy>(
        reinterpret_cast<global_pointer_t>(m_Pointer));
  }
#endif // __ENABLE_USM_ADDR_SPACE__

  // Only if Space == global_space
  template <
      access::address_space _Space = Space,
      typename = typename std::enable_if_t<
          _Space == Space && Space == access::address_space::global_space>>
  void prefetch(size_t NumElements) const {
    size_t NumBytes = NumElements * sizeof(ElementType);
    using ptr_t = typename detail::DecoratedType<char, Space>::type const *;
    __spirv_ocl_prefetch(reinterpret_cast<ptr_t>(m_Pointer), NumBytes);
  }

private:
  pointer_t m_Pointer;
};

// Legacy specialization of multi_ptr for void.
// TODO: Add deprecation warning here when possible.
template <access::address_space Space>
class __SYCL2020_DEPRECATED(
    "decorated::legacy multi_ptr specialization is deprecated since SYCL 2020.")
    multi_ptr<void, Space, access::decorated::legacy> {
public:
  using value_type = void;
  using element_type = void;
  using difference_type = std::ptrdiff_t;

  // Implementation defined pointer types that correspond to
  // SYCL/OpenCL interoperability types for OpenCL C functions
  using pointer_t = typename detail::LegacyPointerTypes<void, Space>::pointer_t;
  using const_pointer_t =
      typename detail::LegacyPointerTypes<const void, Space>::pointer_t;

  static constexpr access::address_space address_space = Space;

  // Constructors
  multi_ptr() : m_Pointer(nullptr) {}
  multi_ptr(const multi_ptr &) = default;
  multi_ptr(multi_ptr &&) = default;
  multi_ptr(pointer_t pointer) : m_Pointer(pointer) {}
#ifdef __SYCL_DEVICE_ONLY__
  template <
      typename RelayPointerT = pointer_t,
      typename = std::enable_if_t<std::is_same_v<RelayPointerT, pointer_t> &&
                                  !std::is_same_v<RelayPointerT, void *>>>
  multi_ptr(void *pointer) : m_Pointer(detail::cast_AS<pointer_t>(pointer)) {
    // TODO An implementation should reject an argument if the deduced
    // address space is not compatible with Space.
  }
#if defined(RESTRICT_WRITE_ACCESS_TO_CONSTANT_PTR)
  template <typename = typename detail::const_if_const_AS<Space, void>>
  multi_ptr(const void *pointer)
      : m_Pointer(detail::cast_AS<pointer_t>(pointer)) {}
#endif
#endif
  multi_ptr(std::nullptr_t) : m_Pointer(nullptr) {}
  ~multi_ptr() = default;

  // TODO: This constructor is the temporary solution for the existing problem
  // with conversions from multi_ptr<ElementType, Space,
  // access::decorated::legacy> to multi_ptr<void, Space,
  // access::decorated::legacy>. Without it the compiler fails due to having 3
  // different same rank paths available.
  template <typename ElementType>
  multi_ptr(const multi_ptr<ElementType, Space, access::decorated::legacy> &ETP)
      : m_Pointer(ETP.get()) {}

  // Assignment operators
  multi_ptr &operator=(const multi_ptr &) = default;
  multi_ptr &operator=(multi_ptr &&) = default;
  multi_ptr &operator=(pointer_t pointer) {
    m_Pointer = pointer;
    return *this;
  }
#ifdef __SYCL_DEVICE_ONLY__
  template <
      typename RelayPointerT = pointer_t,
      typename = std::enable_if_t<std::is_same_v<RelayPointerT, pointer_t> &&
                                  !std::is_same_v<RelayPointerT, void *>>>
  multi_ptr &operator=(void *pointer) {
    // TODO An implementation should reject an argument if the deduced
    // address space is not compatible with Space.
    m_Pointer = detail::cast_AS<pointer_t>(pointer);
    return *this;
  }
#endif
  multi_ptr &operator=(std::nullptr_t) {
    m_Pointer = nullptr;
    return *this;
  }

  // Only if Space is in
  // {global_space, ext_intel_global_device_space, generic_space}
  template <
      typename ElementType, int dimensions, access::mode Mode,
      typename PropertyListT, access::address_space _Space = Space,
      typename = typename std::enable_if_t<
          _Space == Space &&
          (Space == access::address_space::generic_space ||
           Space == access::address_space::global_space ||
           Space == access::address_space::ext_intel_global_device_space)>>
  multi_ptr(accessor<ElementType, dimensions, Mode, target::device,
                     access::placeholder::false_t, PropertyListT>
                Accessor)
      : multi_ptr(Accessor.get_pointer()) {}

  // Only if Space == local_space || generic_space
  template <
      typename ElementType, int dimensions, access::mode Mode,
      typename PropertyListT, access::address_space _Space = Space,
      typename = typename std::enable_if_t<
          _Space == Space && (Space == access::address_space::generic_space ||
                              Space == access::address_space::local_space)>>
  multi_ptr(accessor<ElementType, dimensions, Mode, target::local,
                     access::placeholder::false_t, PropertyListT>
                Accessor)
      : multi_ptr(Accessor.get_pointer()) {}

  // Only if Space == local_space || generic_space
  template <
      typename ElementType, int dimensions,
      access::address_space _Space = Space,
      typename = typename std::enable_if_t<
          _Space == Space && (Space == access::address_space::generic_space ||
                              Space == access::address_space::local_space)>>
  multi_ptr(local_accessor<ElementType, dimensions> Accessor)
      : multi_ptr(Accessor.get_pointer()) {}

  // Only if Space == constant_space
  template <
      typename ElementType, int dimensions, access::mode Mode,
      typename PropertyListT, access::address_space _Space = Space,
      typename = typename std::enable_if_t<
          _Space == Space && Space == access::address_space::constant_space>>
  multi_ptr(accessor<ElementType, dimensions, Mode, target::constant_buffer,
                     access::placeholder::false_t, PropertyListT>
                Accessor)
      : multi_ptr(Accessor.get_pointer()) {}

  using ReturnPtr = detail::const_if_const_AS<Space, void> *;
  // Returns the underlying OpenCL C pointer
  pointer_t get() const { return m_Pointer; }
  pointer_t get_decorated() const { return m_Pointer; }
  std::add_pointer_t<element_type> get_raw() const {
    return reinterpret_cast<std::add_pointer_t<element_type>>(get());
  }

  // Implicit conversion to the underlying pointer type
  operator ReturnPtr() const { return reinterpret_cast<ReturnPtr>(m_Pointer); };

  // Explicit conversion to a multi_ptr<ElementType>
  template <typename ElementType>
  explicit
  operator multi_ptr<ElementType, Space, access::decorated::legacy>() const {
    using elem_pointer_t =
        typename detail::DecoratedType<ElementType, Space>::type *;
    return multi_ptr<ElementType, Space, access::decorated::legacy>(
        static_cast<elem_pointer_t>(m_Pointer));
  }

  // Implicit conversion to multi_ptr<const void, Space>
  operator multi_ptr<const void, Space, access::decorated::legacy>() const {
    using ptr_t = typename detail::DecoratedType<const void, Space>::type *;
    return multi_ptr<const void, Space, access::decorated::legacy>(
        reinterpret_cast<ptr_t>(m_Pointer));
  }

private:
  pointer_t m_Pointer;
};

// Legacy specialization of multi_ptr for const void.
// TODO: Add deprecation warning here when possible.
template <access::address_space Space>
class __SYCL2020_DEPRECATED(
    "decorated::legacy multi_ptr specialization is deprecated since SYCL 2020.")
    multi_ptr<const void, Space, access::decorated::legacy> {
public:
  using value_type = const void;
  using element_type = const void;
  using difference_type = std::ptrdiff_t;

  // Implementation defined pointer types that correspond to
  // SYCL/OpenCL interoperability types for OpenCL C functions
  using pointer_t =
      typename detail::LegacyPointerTypes<const void, Space>::pointer_t;
  using const_pointer_t =
      typename detail::LegacyPointerTypes<const void, Space>::pointer_t;

  static constexpr access::address_space address_space = Space;

  // Constructors
  multi_ptr() : m_Pointer(nullptr) {}
  multi_ptr(const multi_ptr &) = default;
  multi_ptr(multi_ptr &&) = default;
  multi_ptr(pointer_t pointer) : m_Pointer(pointer) {}
#ifdef __SYCL_DEVICE_ONLY__
  template <
      typename RelayPointerT = pointer_t,
      typename = std::enable_if_t<std::is_same_v<RelayPointerT, pointer_t> &&
                                  !std::is_same_v<RelayPointerT, const void *>>>
  multi_ptr(const void *pointer)
      : m_Pointer(detail::cast_AS<pointer_t>(pointer)) {
    // TODO An implementation should reject an argument if the deduced
    // address space is not compatible with Space.
  }
#if defined(RESTRICT_WRITE_ACCESS_TO_CONSTANT_PTR)
  template <typename = typename detail::const_if_const_AS<Space, void>>
  multi_ptr(const void *pointer)
      : m_Pointer(detail::cast_AS<pointer_t>(pointer)) {}
#endif
#endif
  multi_ptr(std::nullptr_t) : m_Pointer(nullptr) {}
  ~multi_ptr() = default;

  // TODO: This constructor is the temporary solution for the existing problem
  // with conversions from multi_ptr<ElementType, Space,
  // access::decorated::legacy> to multi_ptr<const void, Space,
  // access::decorated::legacy>. Without it the compiler fails due to having 3
  // different same rank paths available.
  template <typename ElementType>
  multi_ptr(const multi_ptr<ElementType, Space, access::decorated::legacy> &ETP)
      : m_Pointer(ETP.get()) {}

  // Assignment operators
  multi_ptr &operator=(const multi_ptr &) = default;
  multi_ptr &operator=(multi_ptr &&) = default;
  multi_ptr &operator=(pointer_t pointer) {
    m_Pointer = pointer;
    return *this;
  }
#ifdef __SYCL_DEVICE_ONLY__
  template <
      typename RelayPointerT = pointer_t,
      typename = std::enable_if_t<std::is_same_v<RelayPointerT, pointer_t> &&
                                  !std::is_same_v<RelayPointerT, const void *>>>
  multi_ptr &operator=(const void *pointer) {
    // TODO An implementation should reject an argument if the deduced
    // address space is not compatible with Space.
    m_Pointer = detail::cast_AS<pointer_t>(pointer);
    return *this;
  }
#endif
  multi_ptr &operator=(std::nullptr_t) {
    m_Pointer = nullptr;
    return *this;
  }

  // Only if Space is in
  // {global_space, ext_intel_global_device_space, generic_space}
  template <
      typename ElementType, int dimensions, access::mode Mode,
      typename PropertyListT, access::address_space _Space = Space,
      typename = typename std::enable_if_t<
          _Space == Space &&
          (Space == access::address_space::generic_space ||
           Space == access::address_space::global_space ||
           Space == access::address_space::ext_intel_global_device_space)>>
  multi_ptr(accessor<ElementType, dimensions, Mode, target::device,
                     access::placeholder::false_t, PropertyListT>
                Accessor)
      : multi_ptr(Accessor.get_pointer()) {}

  // Only if Space == local_space || generic_space
  template <
      typename ElementType, int dimensions, access::mode Mode,
      typename PropertyListT, access::address_space _Space = Space,
      typename = typename std::enable_if_t<
          _Space == Space && (Space == access::address_space::generic_space ||
                              Space == access::address_space::local_space)>>
  multi_ptr(accessor<ElementType, dimensions, Mode, target::local,
                     access::placeholder::false_t, PropertyListT>
                Accessor)
      : multi_ptr(Accessor.get_pointer()) {}

  // Only if Space == local_space || generic_space
  template <
      typename ElementType, int dimensions,
      access::address_space _Space = Space,
      typename = typename std::enable_if_t<
          _Space == Space && (Space == access::address_space::generic_space ||
                              Space == access::address_space::local_space)>>
  multi_ptr(local_accessor<ElementType, dimensions> Accessor)
      : multi_ptr(Accessor.get_pointer()) {}

  // Only if Space == constant_space
  template <
      typename ElementType, int dimensions, access::mode Mode,
      typename PropertyListT, access::address_space _Space = Space,
      typename = typename std::enable_if_t<
          _Space == Space && Space == access::address_space::constant_space>>
  multi_ptr(accessor<ElementType, dimensions, Mode, target::constant_buffer,
                     access::placeholder::false_t, PropertyListT>
                Accessor)
      : multi_ptr(Accessor.get_pointer()) {}

  // Returns the underlying OpenCL C pointer
  pointer_t get() const { return m_Pointer; }
  pointer_t get_decorated() const { return m_Pointer; }
  std::add_pointer_t<element_type> get_raw() const {
    return reinterpret_cast<std::add_pointer_t<element_type>>(get());
  }

  // Implicit conversion to the underlying pointer type
  operator const void *() const {
    return reinterpret_cast<const void *>(m_Pointer);
  };

  // Explicit conversion to a multi_ptr<const ElementType>
  // multi_ptr<const void, Space, access::decorated::legacy> ->
  //   multi_ptr<const void, Space, access::decorated::legacy>
  // The result type must have const specifier.
  template <typename ElementType>
  explicit
  operator multi_ptr<const ElementType, Space, access::decorated::legacy>()
      const {
    using elem_pointer_t =
        typename detail::DecoratedType<const ElementType, Space>::type *;
    return multi_ptr<const ElementType, Space, access::decorated::legacy>(
        static_cast<elem_pointer_t>(m_Pointer));
  }

private:
  pointer_t m_Pointer;
};

#ifdef __cpp_deduction_guides
template <class T, int dimensions, access::placeholder isPlaceholder,
          typename PropertyListT>
multi_ptr(accessor<T, dimensions, access::mode::read, target::device,
                   isPlaceholder, PropertyListT>)
    -> multi_ptr<const T, access::address_space::global_space,
                 access::decorated::no>;
template <class T, int dimensions, access::placeholder isPlaceholder,
          typename PropertyListT>
multi_ptr(accessor<T, dimensions, access::mode::write, target::device,
                   isPlaceholder, PropertyListT>)
    -> multi_ptr<T, access::address_space::global_space, access::decorated::no>;
template <class T, int dimensions, access::placeholder isPlaceholder,
          typename PropertyListT>
multi_ptr(accessor<T, dimensions, access::mode::read_write, target::device,
                   isPlaceholder, PropertyListT>)
    -> multi_ptr<T, access::address_space::global_space, access::decorated::no>;
template <class T, int dimensions, access::placeholder isPlaceholder,
          typename PropertyListT>
multi_ptr(accessor<T, dimensions, access_mode::read, target::constant_buffer,
                   isPlaceholder, PropertyListT>)
    -> multi_ptr<const T, access::address_space::constant_space,
                 access::decorated::no>;
template <class T, int dimensions, access::mode Mode,
          access::placeholder isPlaceholder, typename PropertyListT>
multi_ptr(
    accessor<T, dimensions, Mode, target::local, isPlaceholder, PropertyListT>)
    -> multi_ptr<T, access::address_space::local_space, access::decorated::no>;
template <typename T, int dimensions>
multi_ptr(local_accessor<T, dimensions>)
    -> multi_ptr<T, access::address_space::local_space, access::decorated::no>;
#endif

template <access::address_space Space, access::decorated DecorateAddress,
          typename ElementType>
multi_ptr<ElementType, Space, DecorateAddress>
address_space_cast(ElementType *pointer) {
  // TODO An implementation should reject an argument if the deduced address
  // space is not compatible with Space.
  // Use LegacyPointerTypes here to also allow constant_space
  return multi_ptr<ElementType, Space, DecorateAddress>(
      detail::cast_AS<
          typename detail::LegacyPointerTypes<ElementType, Space>::pointer_t>(
          pointer));
}

template <
    typename ElementType, access::address_space Space,
    access::decorated DecorateAddress = access::decorated::legacy,
    typename = std::enable_if_t<DecorateAddress == access::decorated::legacy>>
__SYCL2020_DEPRECATED("make_ptr is deprecated since SYCL 2020. Please use "
                      "address_space_cast instead.")
multi_ptr<ElementType, Space, DecorateAddress> make_ptr(
    typename multi_ptr<ElementType, Space, DecorateAddress>::pointer_t
        pointer) {
  return {pointer};
}

template <
    typename ElementType, access::address_space Space,
    access::decorated DecorateAddress,
    typename = std::enable_if_t<DecorateAddress != access::decorated::legacy>>
__SYCL2020_DEPRECATED("make_ptr is deprecated since SYCL 2020. Please use "
                      "address_space_cast instead.")
multi_ptr<ElementType, Space, DecorateAddress> make_ptr(
    typename multi_ptr<ElementType, Space, DecorateAddress>::pointer pointer) {
  return address_space_cast<Space, DecorateAddress>(pointer);
}

#ifdef __SYCL_DEVICE_ONLY__
// An implementation should reject an argument if the deduced address space
// is not compatible with Space.
// This is guaranteed by the c'tor.
template <typename ElementType, access::address_space Space,
          access::decorated DecorateAddress = access::decorated::legacy>
__SYCL2020_DEPRECATED("make_ptr is deprecated since SYCL 2020. Please use "
                      "address_space_cast instead.")
multi_ptr<ElementType, Space, DecorateAddress> make_ptr(ElementType *pointer) {
  return address_space_cast<Space, DecorateAddress>(pointer);
}
#if defined(RESTRICT_WRITE_ACCESS_TO_CONSTANT_PTR)
template <typename ElementType, access::address_space Space,
          access::decorated DecorateAddress = access::decorated::legacy,
          typename = typename detail::const_if_const_AS<Space, ElementType>>
__SYCL2020_DEPRECATED("make_ptr is deprecated since SYCL 2020. Please use "
                      "address_space_cast instead.")
multi_ptr<ElementType, Space, DecorateAddress> make_ptr(
    const ElementType *pointer) {
  return multi_ptr<ElementType, Space, DecorateAddress>(pointer);
}
#endif // RESTRICT_WRITE_ACCESS_TO_CONSTANT_PTR
#endif // // __SYCL_DEVICE_ONLY__

template <typename ElementType, access::address_space Space,
          access::decorated DecorateAddress>
bool operator==(const multi_ptr<ElementType, Space, DecorateAddress> &lhs,
                const multi_ptr<ElementType, Space, DecorateAddress> &rhs) {
  return lhs.get() == rhs.get();
}

template <typename ElementType, access::address_space Space,
          access::decorated DecorateAddress>
bool operator!=(const multi_ptr<ElementType, Space, DecorateAddress> &lhs,
                const multi_ptr<ElementType, Space, DecorateAddress> &rhs) {
  return lhs.get() != rhs.get();
}

template <typename ElementType, access::address_space Space,
          access::decorated DecorateAddress>
bool operator<(const multi_ptr<ElementType, Space, DecorateAddress> &lhs,
               const multi_ptr<ElementType, Space, DecorateAddress> &rhs) {
  return lhs.get() < rhs.get();
}

template <typename ElementType, access::address_space Space,
          access::decorated DecorateAddress>
bool operator>(const multi_ptr<ElementType, Space, DecorateAddress> &lhs,
               const multi_ptr<ElementType, Space, DecorateAddress> &rhs) {
  return lhs.get() > rhs.get();
}

template <typename ElementType, access::address_space Space,
          access::decorated DecorateAddress>
bool operator<=(const multi_ptr<ElementType, Space, DecorateAddress> &lhs,
                const multi_ptr<ElementType, Space, DecorateAddress> &rhs) {
  return lhs.get() <= rhs.get();
}

template <typename ElementType, access::address_space Space,
          access::decorated DecorateAddress>
bool operator>=(const multi_ptr<ElementType, Space, DecorateAddress> &lhs,
                const multi_ptr<ElementType, Space, DecorateAddress> &rhs) {
  return lhs.get() >= rhs.get();
}

template <typename ElementType, access::address_space Space,
          access::decorated DecorateAddress>
bool operator!=(const multi_ptr<ElementType, Space, DecorateAddress> &lhs,
                std::nullptr_t) {
  return lhs.get() != nullptr;
}

template <typename ElementType, access::address_space Space,
          access::decorated DecorateAddress>
bool operator!=(std::nullptr_t,
                const multi_ptr<ElementType, Space, DecorateAddress> &rhs) {
  return rhs.get() != nullptr;
}

template <typename ElementType, access::address_space Space,
          access::decorated DecorateAddress>
bool operator==(const multi_ptr<ElementType, Space, DecorateAddress> &lhs,
                std::nullptr_t) {
  return lhs.get() == nullptr;
}

template <typename ElementType, access::address_space Space,
          access::decorated DecorateAddress>
bool operator==(std::nullptr_t,
                const multi_ptr<ElementType, Space, DecorateAddress> &rhs) {
  return rhs.get() == nullptr;
}

template <typename ElementType, access::address_space Space,
          access::decorated DecorateAddress>
bool operator>(const multi_ptr<ElementType, Space, DecorateAddress> &lhs,
               std::nullptr_t) {
  return lhs.get() != nullptr;
}

template <typename ElementType, access::address_space Space,
          access::decorated DecorateAddress>
bool operator>(std::nullptr_t,
               const multi_ptr<ElementType, Space, DecorateAddress> &) {
  return false;
}

template <typename ElementType, access::address_space Space,
          access::decorated DecorateAddress>
bool operator<(const multi_ptr<ElementType, Space, DecorateAddress> &,
               std::nullptr_t) {
  return false;
}

template <typename ElementType, access::address_space Space,
          access::decorated DecorateAddress>
bool operator<(std::nullptr_t,
               const multi_ptr<ElementType, Space, DecorateAddress> &rhs) {
  return rhs.get() != nullptr;
}

template <typename ElementType, access::address_space Space,
          access::decorated DecorateAddress>
bool operator>=(const multi_ptr<ElementType, Space, DecorateAddress> &,
                std::nullptr_t) {
  return true;
}

template <typename ElementType, access::address_space Space,
          access::decorated DecorateAddress>
bool operator>=(std::nullptr_t,
                const multi_ptr<ElementType, Space, DecorateAddress> &rhs) {
  return rhs.get() == nullptr;
}

template <typename ElementType, access::address_space Space,
          access::decorated DecorateAddress>
bool operator<=(const multi_ptr<ElementType, Space, DecorateAddress> &lhs,
                std::nullptr_t) {
  return lhs.get() == nullptr;
}

template <typename ElementType, access::address_space Space,
          access::decorated DecorateAddress>
bool operator<=(std::nullptr_t,
                const multi_ptr<ElementType, Space, DecorateAddress> &) {
  return true;
}

} // namespace _V1
} // namespace sycl
