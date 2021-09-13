// -*- C++ -*-
//===------------------------------ span ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#if __cplusplus >= 201703L

#ifndef _SYCL_SPAN
#define _SYCL_SPAN

/*
    Derived from libcxx span.
    Original _LIBCPP macros replaced with _SYCL_SPAN to avoid collisions.


    span synopsis

namespace std {

// constants
inline constexpr size_t dynamic_extent = numeric_limits<size_t>::max();

// [views.span], class template span
template <class ElementType, size_t Extent = dynamic_extent>
    class span;

// [span.objectrep], views of object representation
template <class ElementType, size_t Extent>
    span<const byte, ((Extent == dynamic_extent) ? dynamic_extent :
        (sizeof(ElementType) * Extent))> as_bytes(span<ElementType, Extent> s)
noexcept;

template <class ElementType, size_t Extent>
    span<      byte, ((Extent == dynamic_extent) ? dynamic_extent :
        (sizeof(ElementType) * Extent))> as_writable_bytes(span<ElementType,
Extent> s) noexcept;


namespace std {
template <class ElementType, size_t Extent = dynamic_extent>
class span {
public:
    // constants and types
    using element_type = ElementType;
    using value_type = std::remove_cv_t<ElementType>;
    using size_type = size_t;
    using difference_type = ptrdiff_t;
    using pointer = element_type*;
    using const_pointer = const element_type*;
    using reference = element_type&;
    using const_reference = const element_type&;
    using iterator = implementation-defined;
    using reverse_iterator = std::reverse_iterator<iterator>;
    static constexpr size_type extent = Extent;

    // [span.cons], span constructors, copy, assignment, and destructor
    constexpr span() noexcept;
    constexpr explicit(Extent != dynamic_extent) span(pointer ptr, size_type
count); constexpr explicit(Extent != dynamic_extent) span(pointer firstElem,
pointer lastElem); template <size_t N> constexpr span(element_type (&arr)[N])
noexcept; template <size_t N> constexpr span(array<value_type, N>& arr)
noexcept; template <size_t N> constexpr span(const array<value_type, N>& arr)
noexcept; template <class Container> constexpr explicit(Extent !=
dynamic_extent) span(Container& cont); template <class Container> constexpr
explicit(Extent != dynamic_extent) span(const Container& cont); constexpr
span(const span& other) noexcept = default; template <class OtherElementType,
size_t OtherExtent> constexpr explicit(Extent != dynamic_extent) span(const
span<OtherElementType, OtherExtent>& s) noexcept; ~span() noexcept = default;
    constexpr span& operator=(const span& other) noexcept = default;

    // [span.sub], span subviews
    template <size_t Count>
        constexpr span<element_type, Count> first() const;
    template <size_t Count>
        constexpr span<element_type, Count> last() const;
    template <size_t Offset, size_t Count = dynamic_extent>
        constexpr span<element_type, see below> subspan() const;

    constexpr span<element_type, dynamic_extent> first(size_type count) const;
    constexpr span<element_type, dynamic_extent> last(size_type count) const;
    constexpr span<element_type, dynamic_extent> subspan(size_type offset,
size_type count = dynamic_extent) const;

    // [span.obs], span observers
    constexpr size_type size() const noexcept;
    constexpr size_type size_bytes() const noexcept;
    constexpr bool empty() const noexcept;

    // [span.elem], span element access
    constexpr reference operator[](size_type idx) const;
    constexpr reference front() const;
    constexpr reference back() const;
    constexpr pointer data() const noexcept;

    // [span.iterators], span iterator support
    constexpr iterator begin() const noexcept;
    constexpr iterator end() const noexcept;
    constexpr reverse_iterator rbegin() const noexcept;
    constexpr reverse_iterator rend() const noexcept;

private:
    pointer data_;    // exposition only
    size_type size_;  // exposition only
};

template<class T, size_t N>
    span(T (&)[N]) -> span<T, N>;

template<class T, size_t N>
    span(array<T, N>&) -> span<T, N>;

template<class T, size_t N>
    span(const array<T, N>&) -> span<const T, N>;

template<class Container>
    span(Container&) -> span<typename Container::value_type>;

template<class Container>
    span(const Container&) -> span<const typename Container::value_type>;

} // namespace std

*/

#include <array>       // for array
#include <cassert>     // for assert
#include <cstddef>     // for byte
#include <iterator>    // for iterators
#include <type_traits> // for remove_cv, etc

#define _SYCL_SPAN_TEMPLATE_VIS
#define _SYCL_SPAN_INLINE_VISIBILITY inline

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

// byte is unsigned char at sycl/image.hpp:58
using byte = unsigned char;

// asserts suppressed for device compatibility.
// TODO: enable
#if defined(__SYCL_DEVICE_ONLY__)
#define _SYCL_SPAN_ASSERT(x, m) ((void)0)
#else
#define _SYCL_SPAN_ASSERT(x, m) assert(((x) && m))
#endif

inline constexpr size_t dynamic_extent = SIZE_MAX;
template <typename _Tp, size_t _Extent = dynamic_extent> class span;

template <class _Tp> struct __is_span_impl : public std::false_type {};

template <class _Tp, size_t _Extent>
struct __is_span_impl<span<_Tp, _Extent>> : public std::true_type {};

template <class _Tp>
struct __is_span : public __is_span_impl<std::remove_cv_t<_Tp>> {};

template <class _Tp> struct __is_std_array_impl : public std::false_type {};

template <class _Tp, size_t _Sz>
struct __is_std_array_impl<std::array<_Tp, _Sz>> : public std::true_type {};

template <class _Tp>
struct __is_std_array : public __is_std_array_impl<std::remove_cv_t<_Tp>> {};

template <class _Tp, class _ElementType, class = void>
struct __is_span_compatible_container : public std::false_type {};

template <class _Tp, class _ElementType>
struct __is_span_compatible_container<
    _Tp, _ElementType,
    std::void_t<
        // is not a specialization of span
        typename std::enable_if<!__is_span<_Tp>::value, std::nullptr_t>::type,
        // is not a specialization of array
        typename std::enable_if<!__is_std_array<_Tp>::value,
                                std::nullptr_t>::type,
        // is_array_v<Container> is false,
        typename std::enable_if<!std::is_array_v<_Tp>, std::nullptr_t>::type,
        // data(cont) and size(cont) are well formed
        decltype(data(std::declval<_Tp>())),
        decltype(size(std::declval<_Tp>())),
        // remove_pointer_t<decltype(data(cont))>(*)[] is convertible to
        // ElementType(*)[]
        typename std::enable_if<
            std::is_convertible_v<std::remove_pointer_t<decltype(
                                      data(std::declval<_Tp &>()))> (*)[],
                                  _ElementType (*)[]>,
            std::nullptr_t>::type>> : public std::true_type {};

template <typename _Tp, size_t _Extent> class _SYCL_SPAN_TEMPLATE_VIS span {
public:
  //  constants and types
  using element_type = _Tp;
  using value_type = std::remove_cv_t<_Tp>;
  using size_type = size_t;
  using difference_type = ptrdiff_t;
  using pointer = _Tp *;
  using const_pointer = const _Tp *;
  using reference = _Tp &;
  using const_reference = const _Tp &;
  using iterator = pointer;
  using rev_iterator = std::reverse_iterator<pointer>;

  static constexpr size_type extent = _Extent;

  // [span.cons], span constructors, copy, assignment, and destructor
  template <size_t _Sz = _Extent,
            std::enable_if_t<_Sz == 0, std::nullptr_t> = nullptr>
  _SYCL_SPAN_INLINE_VISIBILITY constexpr span() noexcept : __data{nullptr} {}

  constexpr span(const span &) noexcept = default;
  constexpr span &operator=(const span &) noexcept = default;

  _SYCL_SPAN_INLINE_VISIBILITY constexpr explicit span(pointer __ptr,
                                                       size_type __count)
      : __data{__ptr} {
    (void)__count;
    _SYCL_SPAN_ASSERT(_Extent == __count,
                      "size mismatch in span's constructor (ptr, len)");
  }
  _SYCL_SPAN_INLINE_VISIBILITY constexpr explicit span(pointer __f, pointer __l)
      : __data{__f} {
    (void)__l;
    _SYCL_SPAN_ASSERT(_Extent == distance(__f, __l),
                      "size mismatch in span's constructor (ptr, ptr)");
  }

  template <class _OtherElementType,
            std::enable_if_t<std::is_convertible_v<_OtherElementType (*)[],
                                                   element_type (*)[]>,
                             std::nullptr_t> = nullptr>
  _SYCL_SPAN_INLINE_VISIBILITY constexpr span(
      std::array<_OtherElementType, _Extent> &__arr) noexcept
      : __data{__arr.data()} {}

  template <
      class _OtherElementType,
      std::enable_if_t<std::is_convertible_v<const _OtherElementType (*)[],
                                             element_type (*)[]>,
                       std::nullptr_t> = nullptr>
  _SYCL_SPAN_INLINE_VISIBILITY constexpr span(
      const std::array<_OtherElementType, _Extent> &__arr) noexcept
      : __data{__arr.data()} {}

  template <class _Container>
  _SYCL_SPAN_INLINE_VISIBILITY constexpr explicit span(
      _Container &__c,
      std::enable_if_t<__is_span_compatible_container<_Container, _Tp>::value,
                       std::nullptr_t> = nullptr)
      : __data{std::data(__c)} {
    _SYCL_SPAN_ASSERT(_Extent == std::size(__c),
                      "size mismatch in span's constructor (range)");
  }

  template <class _Container>
  _SYCL_SPAN_INLINE_VISIBILITY constexpr explicit span(
      const _Container &__c,
      std::enable_if_t<
          __is_span_compatible_container<const _Container, _Tp>::value,
          std::nullptr_t> = nullptr)
      : __data{std::data(__c)} {
    _SYCL_SPAN_ASSERT(_Extent == std::size(__c),
                      "size mismatch in span's constructor (range)");
  }

  template <class _OtherElementType>
  _SYCL_SPAN_INLINE_VISIBILITY constexpr span(
      const span<_OtherElementType, _Extent> &__other,
      std::enable_if_t<
          std::is_convertible_v<_OtherElementType (*)[], element_type (*)[]>,
          std::nullptr_t> = nullptr)
      : __data{__other.data()} {}

  //  ~span() noexcept = default;

  template <size_t _Count>
  _SYCL_SPAN_INLINE_VISIBILITY constexpr span<element_type, _Count>
  first() const noexcept {
    static_assert(_Count <= _Extent, "Count out of range in span::first()");
    return span<element_type, _Count>{data(), _Count};
  }

  template <size_t _Count>
  _SYCL_SPAN_INLINE_VISIBILITY constexpr span<element_type, _Count> last() const
      noexcept {
    static_assert(_Count <= _Extent, "Count out of range in span::last()");
    return span<element_type, _Count>{data() + size() - _Count, _Count};
  }

  _SYCL_SPAN_INLINE_VISIBILITY
  constexpr span<element_type, dynamic_extent> first(size_type __count) const
      noexcept {
    _SYCL_SPAN_ASSERT(__count <= size(),
                      "Count out of range in span::first(count)");
    return {data(), __count};
  }

  _SYCL_SPAN_INLINE_VISIBILITY
  constexpr span<element_type, dynamic_extent> last(size_type __count) const
      noexcept {
    _SYCL_SPAN_ASSERT(__count <= size(),
                      "Count out of range in span::last(count)");
    return {data() + size() - __count, __count};
  }

  template <size_t _Offset, size_t _Count = dynamic_extent>
  _SYCL_SPAN_INLINE_VISIBILITY constexpr auto subspan() const noexcept
      -> span<element_type,
              (_Count != dynamic_extent ? _Count : _Extent - _Offset)> {
    static_assert(_Offset <= _Extent, "Offset out of range in span::subspan()");
    static_assert(_Count == dynamic_extent || _Count <= _Extent - _Offset,
                  "Offset + count out of range in span::subspan()");

    using _ReturnType =
        span<element_type,
             _Count != dynamic_extent ? _Count : _Extent - _Offset>;
    return _ReturnType{data() + _Offset,
                       _Count == dynamic_extent ? size() - _Offset : _Count};
  }

  _SYCL_SPAN_INLINE_VISIBILITY
  constexpr span<element_type, dynamic_extent>
  subspan(size_type __offset, size_type __count = dynamic_extent) const
      noexcept {
    _SYCL_SPAN_ASSERT(__offset <= size(),
                      "Offset out of range in span::subspan(offset, count)");
    _SYCL_SPAN_ASSERT(__count <= size() || __count == dynamic_extent,
                      "Count out of range in span::subspan(offset, count)");
    if (__count == dynamic_extent)
      return {data() + __offset, size() - __offset};
    _SYCL_SPAN_ASSERT(
        __count <= size() - __offset,
        "Offset + count out of range in span::subspan(offset, count)");
    return {data() + __offset, __count};
  }

  _SYCL_SPAN_INLINE_VISIBILITY constexpr size_type size() const noexcept {
    return _Extent;
  }
  _SYCL_SPAN_INLINE_VISIBILITY constexpr size_type size_bytes() const noexcept {
    return _Extent * sizeof(element_type);
  }
  _SYCL_SPAN_INLINE_VISIBILITY constexpr bool empty() const noexcept {
    return _Extent == 0;
  }

  _SYCL_SPAN_INLINE_VISIBILITY constexpr reference
  operator[](size_type __idx) const noexcept {
    _SYCL_SPAN_ASSERT(__idx < size(), "span<T,N>[] index out of bounds");
    return __data[__idx];
  }

  _SYCL_SPAN_INLINE_VISIBILITY constexpr reference front() const noexcept {
    _SYCL_SPAN_ASSERT(!empty(), "span<T, N>::front() on empty span");
    return __data[0];
  }

  _SYCL_SPAN_INLINE_VISIBILITY constexpr reference back() const noexcept {
    _SYCL_SPAN_ASSERT(!empty(), "span<T, N>::back() on empty span");
    return __data[size() - 1];
  }

  _SYCL_SPAN_INLINE_VISIBILITY constexpr pointer data() const noexcept {
    return __data;
  }

  // [span.iter], span iterator support
  _SYCL_SPAN_INLINE_VISIBILITY constexpr iterator begin() const noexcept {
    return iterator(data());
  }
  _SYCL_SPAN_INLINE_VISIBILITY constexpr iterator end() const noexcept {
    return iterator(data() + size());
  }
  _SYCL_SPAN_INLINE_VISIBILITY constexpr rev_iterator rbegin() const noexcept {
    return rev_iterator(end());
  }
  _SYCL_SPAN_INLINE_VISIBILITY constexpr rev_iterator rend() const noexcept {
    return rev_iterator(begin());
  }

  _SYCL_SPAN_INLINE_VISIBILITY span<const byte, _Extent * sizeof(element_type)>
  __as_bytes() const noexcept {
    return span<const byte, _Extent * sizeof(element_type)>{
        reinterpret_cast<const byte *>(data()), size_bytes()};
  }

  _SYCL_SPAN_INLINE_VISIBILITY span<byte, _Extent * sizeof(element_type)>
  __as_writable_bytes() const noexcept {
    return span<byte, _Extent * sizeof(element_type)>{
        reinterpret_cast<byte *>(data()), size_bytes()};
  }

private:
  pointer __data;
};

template <typename _Tp>
class _SYCL_SPAN_TEMPLATE_VIS span<_Tp, dynamic_extent> {
private:
public:
  //  constants and types
  using element_type = _Tp;
  using value_type = std::remove_cv_t<_Tp>;
  using size_type = size_t;
  using difference_type = ptrdiff_t;
  using pointer = _Tp *;
  using const_pointer = const _Tp *;
  using reference = _Tp &;
  using const_reference = const _Tp &;
  using iterator = pointer;
  using rev_iterator = std::reverse_iterator<pointer>;

  static constexpr size_type extent = dynamic_extent;

  // [span.cons], span constructors, copy, assignment, and destructor
  _SYCL_SPAN_INLINE_VISIBILITY constexpr span() noexcept
      : __data{nullptr}, __size{0} {}

  constexpr span(const span &) noexcept = default;
  constexpr span &operator=(const span &) noexcept = default;

  _SYCL_SPAN_INLINE_VISIBILITY constexpr span(pointer __ptr, size_type __count)
      : __data{__ptr}, __size{__count} {}
  _SYCL_SPAN_INLINE_VISIBILITY constexpr span(pointer __f, pointer __l)
      : __data{__f}, __size{static_cast<size_t>(distance(__f, __l))} {}

  template <size_t _Sz>
  _SYCL_SPAN_INLINE_VISIBILITY constexpr span(
      element_type (&__arr)[_Sz]) noexcept
      : __data{__arr}, __size{_Sz} {}

  template <class _OtherElementType, size_t _Sz,
            std::enable_if_t<std::is_convertible_v<_OtherElementType (*)[],
                                                   element_type (*)[]>,
                             std::nullptr_t> = nullptr>
  _SYCL_SPAN_INLINE_VISIBILITY constexpr span(
      std::array<_OtherElementType, _Sz> &__arr) noexcept
      : __data{__arr.data()}, __size{_Sz} {}

  template <
      class _OtherElementType, size_t _Sz,
      std::enable_if_t<std::is_convertible_v<const _OtherElementType (*)[],
                                             element_type (*)[]>,
                       std::nullptr_t> = nullptr>
  _SYCL_SPAN_INLINE_VISIBILITY constexpr span(
      const std::array<_OtherElementType, _Sz> &__arr) noexcept
      : __data{__arr.data()}, __size{_Sz} {}

  template <class _Container>
  _SYCL_SPAN_INLINE_VISIBILITY constexpr span(
      _Container &__c,
      std::enable_if_t<__is_span_compatible_container<_Container, _Tp>::value,
                       std::nullptr_t> = nullptr)
      : __data{std::data(__c)}, __size{(size_type)std::size(__c)} {}

  template <class _Container>
  _SYCL_SPAN_INLINE_VISIBILITY constexpr span(
      const _Container &__c,
      std::enable_if_t<
          __is_span_compatible_container<const _Container, _Tp>::value,
          std::nullptr_t> = nullptr)
      : __data{std::data(__c)}, __size{(size_type)std::size(__c)} {}

  template <class _OtherElementType, size_t _OtherExtent>
  _SYCL_SPAN_INLINE_VISIBILITY constexpr span(
      const span<_OtherElementType, _OtherExtent> &__other,
      std::enable_if_t<
          std::is_convertible_v<_OtherElementType (*)[], element_type (*)[]>,
          std::nullptr_t> = nullptr) noexcept
      : __data{__other.data()}, __size{__other.size()} {}

  //    ~span() noexcept = default;

  template <size_t _Count>
  _SYCL_SPAN_INLINE_VISIBILITY constexpr span<element_type, _Count>
  first() const noexcept {
    _SYCL_SPAN_ASSERT(_Count <= size(), "Count out of range in span::first()");
    return span<element_type, _Count>{data(), _Count};
  }

  template <size_t _Count>
  _SYCL_SPAN_INLINE_VISIBILITY constexpr span<element_type, _Count> last() const
      noexcept {
    _SYCL_SPAN_ASSERT(_Count <= size(), "Count out of range in span::last()");
    return span<element_type, _Count>{data() + size() - _Count, _Count};
  }

  _SYCL_SPAN_INLINE_VISIBILITY
  constexpr span<element_type, dynamic_extent> first(size_type __count) const
      noexcept {
    _SYCL_SPAN_ASSERT(__count <= size(),
                      "Count out of range in span::first(count)");
    return {data(), __count};
  }

  _SYCL_SPAN_INLINE_VISIBILITY
  constexpr span<element_type, dynamic_extent> last(size_type __count) const
      noexcept {
    _SYCL_SPAN_ASSERT(__count <= size(),
                      "Count out of range in span::last(count)");
    return {data() + size() - __count, __count};
  }

  template <size_t _Offset, size_t _Count = dynamic_extent>
  _SYCL_SPAN_INLINE_VISIBILITY constexpr span<element_type, _Count>
  subspan() const noexcept {
    _SYCL_SPAN_ASSERT(_Offset <= size(),
                      "Offset out of range in span::subspan()");
    _SYCL_SPAN_ASSERT(_Count == dynamic_extent || _Count <= size() - _Offset,
                      "Offset + count out of range in span::subspan()");
    return span<element_type, _Count>{
        data() + _Offset, _Count == dynamic_extent ? size() - _Offset : _Count};
  }

  constexpr span<element_type, dynamic_extent> _SYCL_SPAN_INLINE_VISIBILITY
  subspan(size_type __offset, size_type __count = dynamic_extent) const
      noexcept {
    _SYCL_SPAN_ASSERT(__offset <= size(),
                      "Offset out of range in span::subspan(offset, count)");
    _SYCL_SPAN_ASSERT(__count <= size() || __count == dynamic_extent,
                      "count out of range in span::subspan(offset, count)");
    if (__count == dynamic_extent)
      return {data() + __offset, size() - __offset};
    _SYCL_SPAN_ASSERT(
        __count <= size() - __offset,
        "Offset + count out of range in span::subspan(offset, count)");
    return {data() + __offset, __count};
  }

  _SYCL_SPAN_INLINE_VISIBILITY constexpr size_type size() const noexcept {
    return __size;
  }
  _SYCL_SPAN_INLINE_VISIBILITY constexpr size_type size_bytes() const noexcept {
    return __size * sizeof(element_type);
  }
  _SYCL_SPAN_INLINE_VISIBILITY constexpr bool empty() const noexcept {
    return __size == 0;
  }

  _SYCL_SPAN_INLINE_VISIBILITY constexpr reference
  operator[](size_type __idx) const noexcept {
    _SYCL_SPAN_ASSERT(__idx < size(), "span<T>[] index out of bounds");
    return __data[__idx];
  }

  _SYCL_SPAN_INLINE_VISIBILITY constexpr reference front() const noexcept {
    _SYCL_SPAN_ASSERT(!empty(), "span<T>[].front() on empty span");
    return __data[0];
  }

  _SYCL_SPAN_INLINE_VISIBILITY constexpr reference back() const noexcept {
    _SYCL_SPAN_ASSERT(!empty(), "span<T>[].back() on empty span");
    return __data[size() - 1];
  }

  _SYCL_SPAN_INLINE_VISIBILITY constexpr pointer data() const noexcept {
    return __data;
  }

  // [span.iter], span iterator support
  _SYCL_SPAN_INLINE_VISIBILITY constexpr iterator begin() const noexcept {
    return iterator(data());
  }
  _SYCL_SPAN_INLINE_VISIBILITY constexpr iterator end() const noexcept {
    return iterator(data() + size());
  }
  _SYCL_SPAN_INLINE_VISIBILITY constexpr rev_iterator rbegin() const noexcept {
    return rev_iterator(end());
  }
  _SYCL_SPAN_INLINE_VISIBILITY constexpr rev_iterator rend() const noexcept {
    return rev_iterator(begin());
  }

  _SYCL_SPAN_INLINE_VISIBILITY span<const byte, dynamic_extent>
  __as_bytes() const noexcept {
    return {reinterpret_cast<const byte *>(data()), size_bytes()};
  }

  _SYCL_SPAN_INLINE_VISIBILITY span<byte, dynamic_extent>
  __as_writable_bytes() const noexcept {
    return {reinterpret_cast<byte *>(data()), size_bytes()};
  }

private:
  pointer __data;
  size_type __size;
};

//  as_bytes & as_writable_bytes
template <class _Tp, size_t _Extent>
_SYCL_SPAN_INLINE_VISIBILITY auto as_bytes(span<_Tp, _Extent> __s) noexcept
    -> decltype(__s.__as_bytes()) {
  return __s.__as_bytes();
}

template <class _Tp, size_t _Extent>
_SYCL_SPAN_INLINE_VISIBILITY auto
as_writable_bytes(span<_Tp, _Extent> __s) noexcept
    -> std::enable_if_t<!std::is_const_v<_Tp>,
                        decltype(__s.__as_writable_bytes())> {
  return __s.__as_writable_bytes();
}

//  Deduction guides
template <class _Tp, size_t _Sz> span(_Tp (&)[_Sz])->span<_Tp, _Sz>;

template <class _Tp, size_t _Sz> span(std::array<_Tp, _Sz> &)->span<_Tp, _Sz>;

template <class _Tp, size_t _Sz>
span(const std::array<_Tp, _Sz> &)->span<const _Tp, _Sz>;

template <class _Container>
span(_Container &)->span<typename _Container::value_type>;

template <class _Container>
span(const _Container &)->span<const typename _Container::value_type>;

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

#endif // _SYCL_SPAN

#endif // __cplusplus >= 201703L
