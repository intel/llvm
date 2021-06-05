//==------------ multi_ptr.hpp - SYCL multi_ptr class ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/__spirv/spirv_ops.hpp>
#include <CL/sycl/access/access.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/type_traits.hpp>
#include <cassert>
#include <cstddef>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
// Forward declaration
template <typename dataT, int dimensions, access::mode accessMode,
          access::target accessTarget, access::placeholder isPlaceholder,
          typename PropertyListT>
class accessor;

/// Provides constructors for address space qualified and non address space
/// qualified pointers to allow interoperability between plain C++ and OpenCL C.
///
/// \ingroup sycl_api
template <typename ElementType, access::address_space Space> class multi_ptr {
public:
  using element_type =
      detail::conditional_t<std::is_same<ElementType, half>::value,
                    cl::sycl::detail::half_impl::BIsRepresentationT,
                    ElementType>;
  using difference_type = std::ptrdiff_t;

  // Implementation defined pointer and reference types that correspond to
  // SYCL/OpenCL interoperability types for OpenCL C functions
  using pointer_t = typename detail::DecoratedType<ElementType, Space>::type *;
  using const_pointer_t =
      typename detail::DecoratedType<ElementType, Space>::type const *;
  using reference_t =
      typename detail::DecoratedType<ElementType, Space>::type &;
  using const_reference_t =
      typename detail::DecoratedType<ElementType, Space>::type &;

  static constexpr access::address_space address_space = Space;

  // Constructors
  multi_ptr() : m_Pointer(nullptr) {}
  multi_ptr(const multi_ptr &rhs) = default;
  multi_ptr(multi_ptr &&) = default;
#ifdef __SYCL_DEVICE_ONLY__
  multi_ptr(pointer_t pointer) : m_Pointer(pointer) {}
#endif

  multi_ptr(ElementType *pointer) : m_Pointer((pointer_t)(pointer)) {
    // TODO An implementation should reject an argument if the deduced
    // address space is not compatible with Space.
  }
#if defined(RESTRICT_WRITE_ACCESS_TO_CONSTANT_PTR)
  template <typename = typename detail::const_if_const_AS<Space, ElementType>>
  multi_ptr(const ElementType *pointer) : m_Pointer((pointer_t)(pointer)) {}
#endif

  multi_ptr(std::nullptr_t) : m_Pointer(nullptr) {}
  ~multi_ptr() = default;

  // Assignment and access operators
  multi_ptr &operator=(const multi_ptr &) = default;
  multi_ptr &operator=(multi_ptr &&) = default;

#ifdef __SYCL_DEVICE_ONLY__
  multi_ptr &operator=(pointer_t pointer) {
    m_Pointer = pointer;
    return *this;
  }
#endif

  multi_ptr &operator=(ElementType *pointer) {
    // TODO An implementation should reject an argument if the deduced
    // address space is not compatible with Space.
    m_Pointer = (pointer_t)pointer;
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

  // Only if Space == global_space || global_device_space
  template <int dimensions, access::mode Mode,
            access::placeholder isPlaceholder, typename PropertyListT,
            access::address_space _Space = Space,
            typename = typename detail::enable_if_t<
                _Space == Space &&
                (Space == access::address_space::global_space ||
                 Space == access::address_space::global_device_space)>>
  multi_ptr(
      accessor<ElementType, dimensions, Mode, access::target::global_buffer,
               isPlaceholder, PropertyListT>
          Accessor) {
    m_Pointer = (pointer_t)(Accessor.get_pointer().get());
  }

  // Only if Space == local_space
  template <int dimensions, access::mode Mode,
            access::placeholder isPlaceholder, typename PropertyListT,
            access::address_space _Space = Space,
            typename = typename detail::enable_if_t<
                _Space == Space && Space == access::address_space::local_space>>
  multi_ptr(accessor<ElementType, dimensions, Mode, access::target::local,
                     isPlaceholder, PropertyListT>
                Accessor)
      : multi_ptr(Accessor.get_pointer()) {}

  // Only if Space == constant_space
  template <
      int dimensions, access::mode Mode, access::placeholder isPlaceholder,
      typename PropertyListT, access::address_space _Space = Space,
      typename = typename detail::enable_if_t<
          _Space == Space && Space == access::address_space::constant_space>>
  multi_ptr(
      accessor<ElementType, dimensions, Mode, access::target::constant_buffer,
               isPlaceholder, PropertyListT>
          Accessor)
      : multi_ptr(Accessor.get_pointer()) {}

  // The following constructors are necessary to create multi_ptr<const
  // ElementType, Space> from accessor<ElementType, ...>. Constructors above
  // could not be used for this purpose because it will require 2 implicit
  // conversions of user types which is not allowed by C++:
  //    1. from accessor<ElementType, ...> to multi_ptr<ElementType, Space>
  //    2. from multi_ptr<ElementType, Space> to multi_ptr<const ElementType,
  //    Space>

  // Only if Space == global_space || global_device_space and element type is
  // const
  template <
      int dimensions, access::mode Mode, access::placeholder isPlaceholder,
      typename PropertyListT, access::address_space _Space = Space,
      typename ET = ElementType,
      typename = typename detail::enable_if_t<
          _Space == Space &&
          (Space == access::address_space::global_space ||
           Space == access::address_space::global_device_space) &&
          std::is_const<ET>::value && std::is_same<ET, ElementType>::value>>
  multi_ptr(
      accessor<typename detail::remove_const_t<ET>, dimensions, Mode,
               access::target::global_buffer, isPlaceholder, PropertyListT>
          Accessor)
      : multi_ptr(Accessor.get_pointer()) {}

  // Only if Space == local_space and element type is const
  template <
      int dimensions, access::mode Mode, access::placeholder isPlaceholder,
      typename PropertyListT, access::address_space _Space = Space,
      typename ET = ElementType,
      typename = typename detail::enable_if_t<
          _Space == Space && Space == access::address_space::local_space &&
          std::is_const<ET>::value && std::is_same<ET, ElementType>::value>>
  multi_ptr(accessor<typename detail::remove_const_t<ET>, dimensions, Mode,
                     access::target::local, isPlaceholder, PropertyListT>
                Accessor)
      : multi_ptr(Accessor.get_pointer()) {}

  // Only if Space == constant_space and element type is const
  template <
      int dimensions, access::mode Mode, access::placeholder isPlaceholder,
      typename PropertyListT, access::address_space _Space = Space,
      typename ET = ElementType,
      typename = typename detail::enable_if_t<
          _Space == Space && Space == access::address_space::constant_space &&
          std::is_const<ET>::value && std::is_same<ET, ElementType>::value>>
  multi_ptr(
      accessor<typename detail::remove_const_t<ET>, dimensions, Mode,
               access::target::constant_buffer, isPlaceholder, PropertyListT>
          Accessor)
      : multi_ptr(Accessor.get_pointer()) {}

  // TODO: This constructor is the temporary solution for the existing problem
  // with conversions from multi_ptr<ElementType, Space> to
  // multi_ptr<const ElementType, Space>. Without it the compiler
  // fails due to having 3 different same rank paths available.
  // Constructs multi_ptr<const ElementType, Space>:
  //   multi_ptr<ElementType, Space> -> multi_ptr<const ElementTYpe, Space>
  template <typename ET = ElementType>
  multi_ptr(typename detail::enable_if_t<
            std::is_const<ET>::value && std::is_same<ET, ElementType>::value,
            const multi_ptr<typename detail::remove_const_t<ET>, Space>> &ETP)
      : m_Pointer(ETP.get()) {}

  // Returns the underlying OpenCL C pointer
  pointer_t get() const { return m_Pointer; }

  // Implicit conversion to the underlying pointer type
  operator ReturnPtr() const { return reinterpret_cast<ReturnPtr>(m_Pointer); }

  // Implicit conversion to a multi_ptr<void>
  // Only available when ElementType is not const-qualified
  template <typename ET = ElementType>
  operator multi_ptr<
      typename detail::enable_if_t<std::is_same<ET, ElementType>::value &&
                                       !std::is_const<ET>::value,
                                   void>::type,
      Space>() const {
    using ptr_t = typename detail::DecoratedType<void, Space> *;
    return multi_ptr<void, Space>(reinterpret_cast<ptr_t>(m_Pointer));
  }

  // Implicit conversion to a multi_ptr<const void>
  // Only available when ElementType is const-qualified
  template <typename ET = ElementType>
  operator multi_ptr<
      typename detail::enable_if_t<std::is_same<ET, ElementType>::value &&
                                       std::is_const<ET>::value,
                                   const void>::type,
      Space>() const {
    using ptr_t = typename detail::DecoratedType<const void, Space> *;
    return multi_ptr<const void, Space>(reinterpret_cast<ptr_t>(m_Pointer));
  }

  // Implicit conversion to multi_ptr<const ElementType, Space>
  operator multi_ptr<const ElementType, Space>() const {
    using ptr_t =
        typename detail::DecoratedType<const ElementType, Space>::type *;
    return multi_ptr<const ElementType, Space>(
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
  // Only available if Space == address_space::global_device_space ||
  // Space == address_space::global_host_space
  template <access::address_space _Space = Space,
            typename = typename detail::enable_if_t<
                _Space == Space &&
                (Space == access::address_space::global_device_space ||
                 Space == access::address_space::global_host_space)>>
  explicit
  operator multi_ptr<ElementType, access::address_space::global_space>() const {
    using global_pointer_t = typename detail::DecoratedType<
        ElementType, access::address_space::global_space>::type *;
    return multi_ptr<ElementType, access::address_space::global_space>(
        reinterpret_cast<global_pointer_t>(m_Pointer));
  }
#endif // __ENABLE_USM_ADDR_SPACE__

  // Only if Space == global_space
  template <
      access::address_space _Space = Space,
      typename = typename detail::enable_if_t<
          _Space == Space && Space == access::address_space::global_space>>
  void prefetch(size_t NumElements) const {
    size_t NumBytes = NumElements * sizeof(ElementType);
    using ptr_t = typename detail::DecoratedType<char, Space>::type const *;
    __spirv_ocl_prefetch(reinterpret_cast<ptr_t>(m_Pointer), NumBytes);
  }

private:
  pointer_t m_Pointer;
};

// Specialization of multi_ptr for void
template <access::address_space Space> class multi_ptr<void, Space> {
public:
  using element_type = void;
  using difference_type = std::ptrdiff_t;

  // Implementation defined pointer types that correspond to
  // SYCL/OpenCL interoperability types for OpenCL C functions
  using pointer_t = typename detail::DecoratedType<void, Space>::type *;
  using const_pointer_t =
      typename detail::DecoratedType<void, Space>::type const *;

  static constexpr access::address_space address_space = Space;

  // Constructors
  multi_ptr() : m_Pointer(nullptr) {}
  multi_ptr(const multi_ptr &) = default;
  multi_ptr(multi_ptr &&) = default;
  multi_ptr(pointer_t pointer) : m_Pointer(pointer) {}
#ifdef __SYCL_DEVICE_ONLY__
  multi_ptr(void *pointer) : m_Pointer((pointer_t)pointer) {
    // TODO An implementation should reject an argument if the deduced
    // address space is not compatible with Space.
  }
#if defined(RESTRICT_WRITE_ACCESS_TO_CONSTANT_PTR)
  template <typename = typename detail::const_if_const_AS<Space, void>>
  multi_ptr(const void *pointer) : m_Pointer((pointer_t)(pointer)) {}
#endif
#endif
  multi_ptr(std::nullptr_t) : m_Pointer(nullptr) {}
  ~multi_ptr() = default;

  // TODO: This constructor is the temporary solution for the existing problem
  // with conversions from multi_ptr<ElementType, Space> to
  // multi_ptr<void, Space>. Without it the compiler
  // fails due to having 3 different same rank paths available.
  template <typename ElementType>
  multi_ptr(const multi_ptr<ElementType, Space> &ETP) : m_Pointer(ETP.get()) {}

  // Assignment operators
  multi_ptr &operator=(const multi_ptr &) = default;
  multi_ptr &operator=(multi_ptr &&) = default;
  multi_ptr &operator=(pointer_t pointer) {
    m_Pointer = pointer;
    return *this;
  }
#ifdef __SYCL_DEVICE_ONLY__
  multi_ptr &operator=(void *pointer) {
    // TODO An implementation should reject an argument if the deduced
    // address space is not compatible with Space.
    m_Pointer = (pointer_t)pointer;
    return *this;
  }
#endif
  multi_ptr &operator=(std::nullptr_t) {
    m_Pointer = nullptr;
    return *this;
  }

  // Only if Space == global_space || global_device_space
  template <typename ElementType, int dimensions, access::mode Mode,
            typename PropertyListT, access::address_space _Space = Space,
            typename = typename detail::enable_if_t<
                _Space == Space &&
                (Space == access::address_space::global_space ||
                 Space == access::address_space::global_device_space)>>
  multi_ptr(
      accessor<ElementType, dimensions, Mode, access::target::global_buffer,
               access::placeholder::false_t, PropertyListT>
          Accessor)
      : multi_ptr(Accessor.get_pointer()) {}

  // Only if Space == local_space
  template <typename ElementType, int dimensions, access::mode Mode,
            typename PropertyListT, access::address_space _Space = Space,
            typename = typename detail::enable_if_t<
                _Space == Space && Space == access::address_space::local_space>>
  multi_ptr(accessor<ElementType, dimensions, Mode, access::target::local,
                     access::placeholder::false_t, PropertyListT>
                Accessor)
      : multi_ptr(Accessor.get_pointer()) {}

  // Only if Space == constant_space
  template <
      typename ElementType, int dimensions, access::mode Mode,
      typename PropertyListT, access::address_space _Space = Space,
      typename = typename detail::enable_if_t<
          _Space == Space && Space == access::address_space::constant_space>>
  multi_ptr(
      accessor<ElementType, dimensions, Mode, access::target::constant_buffer,
               access::placeholder::false_t, PropertyListT>
          Accessor)
      : multi_ptr(Accessor.get_pointer()) {}

  using ReturnPtr = detail::const_if_const_AS<Space, void> *;
  // Returns the underlying OpenCL C pointer
  pointer_t get() const { return m_Pointer; }

  // Implicit conversion to the underlying pointer type
  operator ReturnPtr() const { return reinterpret_cast<ReturnPtr>(m_Pointer); };

  // Explicit conversion to a multi_ptr<ElementType>
  template <typename ElementType>
  explicit operator multi_ptr<ElementType, Space>() const {
    using elem_pointer_t =
        typename detail::DecoratedType<ElementType, Space>::type *;
    return multi_ptr<ElementType, Space>(
        static_cast<elem_pointer_t>(m_Pointer));
  }

  // Implicit conversion to multi_ptr<const void, Space>
  operator multi_ptr<const void, Space>() const {
    using ptr_t = typename detail::DecoratedType<const void, Space>::type *;
    return multi_ptr<const void, Space>(reinterpret_cast<ptr_t>(m_Pointer));
  }

private:
  pointer_t m_Pointer;
};

// Specialization of multi_ptr for const void
template <access::address_space Space>
class multi_ptr<const void, Space> {
public:
  using element_type = const void;
  using difference_type = std::ptrdiff_t;

  // Implementation defined pointer types that correspond to
  // SYCL/OpenCL interoperability types for OpenCL C functions
  using pointer_t = typename detail::DecoratedType<const void, Space>::type *;
  using const_pointer_t =
      typename detail::DecoratedType<const void, Space>::type const *;

  static constexpr access::address_space address_space = Space;

  // Constructors
  multi_ptr() : m_Pointer(nullptr) {}
  multi_ptr(const multi_ptr &) = default;
  multi_ptr(multi_ptr &&) = default;
  multi_ptr(pointer_t pointer) : m_Pointer(pointer) {}
#ifdef __SYCL_DEVICE_ONLY__
  multi_ptr(const void *pointer) : m_Pointer((pointer_t)pointer) {
    // TODO An implementation should reject an argument if the deduced
    // address space is not compatible with Space.
  }
#if defined(RESTRICT_WRITE_ACCESS_TO_CONSTANT_PTR)
  template <typename = typename detail::const_if_const_AS<Space, void>>
  multi_ptr(const void *pointer) : m_Pointer((pointer_t)(pointer)) {}
#endif
#endif
  multi_ptr(std::nullptr_t) : m_Pointer(nullptr) {}
  ~multi_ptr() = default;

  // TODO: This constructor is the temporary solution for the existing problem
  // with conversions from multi_ptr<ElementType, Space> to
  // multi_ptr<const void, Space>. Without it the compiler
  // fails due to having 3 different same rank paths available.
  template <typename ElementType>
  multi_ptr(const multi_ptr<ElementType, Space> &ETP) : m_Pointer(ETP.get()) {}

  // Assignment operators
  multi_ptr &operator=(const multi_ptr &) = default;
  multi_ptr &operator=(multi_ptr &&) = default;
  multi_ptr &operator=(pointer_t pointer) {
    m_Pointer = pointer;
    return *this;
  }
#ifdef __SYCL_DEVICE_ONLY__
  multi_ptr &operator=(const void *pointer) {
    // TODO An implementation should reject an argument if the deduced
    // address space is not compatible with Space.
    m_Pointer = (pointer_t)pointer;
    return *this;
  }
#endif
  multi_ptr &operator=(std::nullptr_t) {
    m_Pointer = nullptr;
    return *this;
  }

  // Only if Space == global_space || global_device_space
  template <typename ElementType, int dimensions, access::mode Mode,
            typename PropertyListT, access::address_space _Space = Space,
            typename = typename detail::enable_if_t<
                _Space == Space &&
                (Space == access::address_space::global_space ||
                 Space == access::address_space::global_device_space)>>
  multi_ptr(
      accessor<ElementType, dimensions, Mode, access::target::global_buffer,
               access::placeholder::false_t, PropertyListT>
          Accessor)
      : multi_ptr(Accessor.get_pointer()) {}

  // Only if Space == local_space
  template <typename ElementType, int dimensions, access::mode Mode,
            typename PropertyListT, access::address_space _Space = Space,
            typename = typename detail::enable_if_t<
                _Space == Space && Space == access::address_space::local_space>>
  multi_ptr(accessor<ElementType, dimensions, Mode, access::target::local,
                     access::placeholder::false_t, PropertyListT>
                Accessor)
      : multi_ptr(Accessor.get_pointer()) {}

  // Only if Space == constant_space
  template <
      typename ElementType, int dimensions, access::mode Mode,
      typename PropertyListT, access::address_space _Space = Space,
      typename = typename detail::enable_if_t<
          _Space == Space && Space == access::address_space::constant_space>>
  multi_ptr(
      accessor<ElementType, dimensions, Mode, access::target::constant_buffer,
               access::placeholder::false_t, PropertyListT>
          Accessor)
      : multi_ptr(Accessor.get_pointer()) {}

  // Returns the underlying OpenCL C pointer
  pointer_t get() const { return m_Pointer; }

  // Implicit conversion to the underlying pointer type
  operator const void*() const {
    return reinterpret_cast<const void *>(m_Pointer);
  };

  // Explicit conversion to a multi_ptr<const ElementType>
  // multi_ptr<const void, Space> -> multi_ptr<const void, Space>
  // The result type must have const specifier.
  template <typename ElementType>
  explicit operator multi_ptr<const ElementType, Space>() const {
    using elem_pointer_t =
        typename detail::DecoratedType<const ElementType, Space>::type *;
    return multi_ptr<const ElementType, Space>(
        static_cast<elem_pointer_t>(m_Pointer));
  }

private:
  pointer_t m_Pointer;
};

#ifdef __cpp_deduction_guides
template <int dimensions, access::mode Mode, access::placeholder isPlaceholder,
          typename PropertyListT, class T>
multi_ptr(accessor<T, dimensions, Mode, access::target::global_buffer,
                   isPlaceholder, PropertyListT>)
    ->multi_ptr<T, access::address_space::global_space>;
template <int dimensions, access::mode Mode, access::placeholder isPlaceholder,
          typename PropertyListT, class T>
multi_ptr(accessor<T, dimensions, Mode, access::target::constant_buffer,
                   isPlaceholder, PropertyListT>)
    ->multi_ptr<T, access::address_space::constant_space>;
template <int dimensions, access::mode Mode, access::placeholder isPlaceholder,
          typename PropertyListT, class T>
multi_ptr(accessor<T, dimensions, Mode, access::target::local, isPlaceholder,
                   PropertyListT>)
    ->multi_ptr<T, access::address_space::local_space>;
#endif

template <typename ElementType, access::address_space Space>
multi_ptr<ElementType, Space>
make_ptr(typename multi_ptr<ElementType, Space>::pointer_t pointer) {
  return multi_ptr<ElementType, Space>(pointer);
}

#ifdef __SYCL_DEVICE_ONLY__
// An implementation should reject an argument if the deduced address space
// is not compatible with Space.
// This is guaranteed by the c'tor.
template <typename ElementType, access::address_space Space>
multi_ptr<ElementType, Space> make_ptr(ElementType *pointer) {
  return multi_ptr<ElementType, Space>(pointer);
}
#if defined(RESTRICT_WRITE_ACCESS_TO_CONSTANT_PTR)
template <typename ElementType, access::address_space Space,
          typename = typename detail::const_if_const_AS<Space, ElementType>>
multi_ptr<ElementType, Space> make_ptr(const ElementType *pointer) {
  return multi_ptr<ElementType, Space>(pointer);
}
#endif // RESTRICT_WRITE_ACCESS_TO_CONSTANT_PTR
#endif // // __SYCL_DEVICE_ONLY__

template <typename ElementType, access::address_space Space>
bool operator==(const multi_ptr<ElementType, Space> &lhs,
                const multi_ptr<ElementType, Space> &rhs) {
  return lhs.get() == rhs.get();
}

template <typename ElementType, access::address_space Space>
bool operator!=(const multi_ptr<ElementType, Space> &lhs,
                const multi_ptr<ElementType, Space> &rhs) {
  return lhs.get() != rhs.get();
}

template <typename ElementType, access::address_space Space>
bool operator<(const multi_ptr<ElementType, Space> &lhs,
               const multi_ptr<ElementType, Space> &rhs) {
  return lhs.get() < rhs.get();
}

template <typename ElementType, access::address_space Space>
bool operator>(const multi_ptr<ElementType, Space> &lhs,
               const multi_ptr<ElementType, Space> &rhs) {
  return lhs.get() > rhs.get();
}

template <typename ElementType, access::address_space Space>
bool operator<=(const multi_ptr<ElementType, Space> &lhs,
                const multi_ptr<ElementType, Space> &rhs) {
  return lhs.get() <= rhs.get();
}

template <typename ElementType, access::address_space Space>
bool operator>=(const multi_ptr<ElementType, Space> &lhs,
                const multi_ptr<ElementType, Space> &rhs) {
  return lhs.get() >= rhs.get();
}

template <typename ElementType, access::address_space Space>
bool operator!=(const multi_ptr<ElementType, Space> &lhs, std::nullptr_t) {
  return lhs.get() != nullptr;
}

template <typename ElementType, access::address_space Space>
bool operator!=(std::nullptr_t, const multi_ptr<ElementType, Space> &rhs) {
  return rhs.get() != nullptr;
}

template <typename ElementType, access::address_space Space>
bool operator==(const multi_ptr<ElementType, Space> &lhs, std::nullptr_t) {
  return lhs.get() == nullptr;
}

template <typename ElementType, access::address_space Space>
bool operator==(std::nullptr_t, const multi_ptr<ElementType, Space> &rhs) {
  return rhs.get() == nullptr;
}

template <typename ElementType, access::address_space Space>
bool operator>(const multi_ptr<ElementType, Space> &lhs, std::nullptr_t) {
  return lhs.get() != nullptr;
}

template <typename ElementType, access::address_space Space>
bool operator>(std::nullptr_t, const multi_ptr<ElementType, Space> &) {
  return false;
}

template <typename ElementType, access::address_space Space>
bool operator<(const multi_ptr<ElementType, Space> &, std::nullptr_t) {
  return false;
}

template <typename ElementType, access::address_space Space>
bool operator<(std::nullptr_t, const multi_ptr<ElementType, Space> &rhs) {
  return rhs.get() != nullptr;
}

template <typename ElementType, access::address_space Space>
bool operator>=(const multi_ptr<ElementType, Space> &, std::nullptr_t) {
  return true;
}

template <typename ElementType, access::address_space Space>
bool operator>=(std::nullptr_t, const multi_ptr<ElementType, Space> &rhs) {
  return rhs.get() == nullptr;
}

template <typename ElementType, access::address_space Space>
bool operator<=(const multi_ptr<ElementType, Space> &lhs, std::nullptr_t) {
  return lhs.get() == nullptr;
}

template <typename ElementType, access::address_space Space>
bool operator<=(std::nullptr_t, const multi_ptr<ElementType, Space> &rhs) {
  return rhs.get() == nullptr;
}

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
