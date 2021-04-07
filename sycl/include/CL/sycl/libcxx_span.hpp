// -*- C++ -*-
//===------------------------------ span ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#ifndef _LIBCPP_SPAN
#define _LIBCPP_SPAN

/*
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
        (sizeof(ElementType) * Extent))> as_bytes(span<ElementType, Extent> s) noexcept;

template <class ElementType, size_t Extent>
    span<      byte, ((Extent == dynamic_extent) ? dynamic_extent :
        (sizeof(ElementType) * Extent))> as_writable_bytes(span<ElementType, Extent> s) noexcept;


namespace std {
template <class ElementType, size_t Extent = dynamic_extent>
class span {
public:
    // constants and types
    using element_type = ElementType;
    using value_type = remove_cv_t<ElementType>;
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
    constexpr explicit(Extent != dynamic_extent) span(pointer ptr, size_type count);
    constexpr explicit(Extent != dynamic_extent) span(pointer firstElem, pointer lastElem);
    template <size_t N>
        constexpr span(element_type (&arr)[N]) noexcept;
    template <size_t N>
        constexpr span(array<value_type, N>& arr) noexcept;
    template <size_t N>
        constexpr span(const array<value_type, N>& arr) noexcept;
    template <class Container>
        constexpr explicit(Extent != dynamic_extent) span(Container& cont);
    template <class Container>
        constexpr explicit(Extent != dynamic_extent) span(const Container& cont);
    constexpr span(const span& other) noexcept = default;
    template <class OtherElementType, size_t OtherExtent>
        constexpr explicit(Extent != dynamic_extent) span(const span<OtherElementType, OtherExtent>& s) noexcept;
    ~span() noexcept = default;
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
    constexpr span<element_type, dynamic_extent> subspan(size_type offset, size_type count = dynamic_extent) const;

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

//CP
//#include <__config>
#include "libcxx_config.hpp"

//CP - violence
#undef _VSTD
#define _VSTD std

#include <array>        // for array
#include <cstddef>      // for byte
#include <iterator>     // for iterators
#include <type_traits>  // for remove_cv, etc



#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
//CP
//#include <__undef_macros>
#include "libcxx_undefmacros.hpp"

//_LIBCPP_BEGIN_NAMESPACE_STD
namespace std {

//CP (when included from stl.hpp ??)
//using std::void_t;
template <class...> using void_t = void;

template< class T >
/*inline*/ constexpr bool is_array_v = is_array<T>::value;

//CP
/*
//-------------------------
// inject __wrap_iter
// __wrap_iter

template <class _Iter> class __wrap_iter;

template <class _Iter1, class _Iter2>
_LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_IF_NODEBUG
bool
operator==(const __wrap_iter<_Iter1>&, const __wrap_iter<_Iter2>&) _NOEXCEPT;

template <class _Iter1, class _Iter2>
_LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_IF_NODEBUG
bool
operator<(const __wrap_iter<_Iter1>&, const __wrap_iter<_Iter2>&) _NOEXCEPT;

template <class _Iter1, class _Iter2>
_LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_IF_NODEBUG
bool
operator!=(const __wrap_iter<_Iter1>&, const __wrap_iter<_Iter2>&) _NOEXCEPT;

template <class _Iter1, class _Iter2>
_LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_IF_NODEBUG
bool
operator>(const __wrap_iter<_Iter1>&, const __wrap_iter<_Iter2>&) _NOEXCEPT;

template <class _Iter1, class _Iter2>
_LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_IF_NODEBUG
bool
operator>=(const __wrap_iter<_Iter1>&, const __wrap_iter<_Iter2>&) _NOEXCEPT;

template <class _Iter1, class _Iter2>
_LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_IF_NODEBUG
bool
operator<=(const __wrap_iter<_Iter1>&, const __wrap_iter<_Iter2>&) _NOEXCEPT;

#ifndef _LIBCPP_CXX03_LANG
template <class _Iter1, class _Iter2>
_LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_IF_NODEBUG
auto
operator-(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter2>& __y) _NOEXCEPT
-> decltype(__x.base() - __y.base());
#else
template <class _Iter1, class _Iter2>
_LIBCPP_INLINE_VISIBILITY
typename __wrap_iter<_Iter1>::difference_type
operator-(const __wrap_iter<_Iter1>&, const __wrap_iter<_Iter2>&) _NOEXCEPT;
#endif

//--------- iterator:1430 - 1631
template <class _Iter>
_LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_IF_NODEBUG
__wrap_iter<_Iter>
operator+(typename __wrap_iter<_Iter>::difference_type, __wrap_iter<_Iter>) _NOEXCEPT;

template <class _Ip, class _Op> _Op _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX17 copy(_Ip, _Ip, _Op);
template <class _B1, class _B2> _B2 _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX17 copy_backward(_B1, _B1, _B2);
template <class _Ip, class _Op> _Op _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX17 move(_Ip, _Ip, _Op);
template <class _B1, class _B2> _B2 _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX17 move_backward(_B1, _B1, _B2);

template <class _Iter>
class __wrap_iter
{
public:
    typedef _Iter                                                      iterator_type;
    typedef typename iterator_traits<iterator_type>::value_type        value_type;
    typedef typename iterator_traits<iterator_type>::difference_type   difference_type;
    typedef typename iterator_traits<iterator_type>::pointer           pointer;
    typedef typename iterator_traits<iterator_type>::reference         reference;
    typedef typename iterator_traits<iterator_type>::iterator_category iterator_category;
#if _LIBCPP_STD_VER > 17
    typedef _If<__is_cpp17_contiguous_iterator<_Iter>::value,
                contiguous_iterator_tag, iterator_category>            iterator_concept;
#endif

private:
    iterator_type __i;
public:
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_IF_NODEBUG __wrap_iter() _NOEXCEPT
#if _LIBCPP_STD_VER > 11
                : __i{}
#endif
    {
#if _LIBCPP_DEBUG_LEVEL == 2
        __get_db()->__insert_i(this);
#endif
    }
    template <class _Up> _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_IF_NODEBUG
        __wrap_iter(const __wrap_iter<_Up>& __u,
            typename enable_if<is_convertible<_Up, iterator_type>::value>::type* = nullptr) _NOEXCEPT
            : __i(__u.base())
    {
#if _LIBCPP_DEBUG_LEVEL == 2
        __get_db()->__iterator_copy(this, &__u);
#endif
    }
#if _LIBCPP_DEBUG_LEVEL == 2
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_IF_NODEBUG
    __wrap_iter(const __wrap_iter& __x)
        : __i(__x.base())
    {
        __get_db()->__iterator_copy(this, &__x);
    }
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_IF_NODEBUG
    __wrap_iter& operator=(const __wrap_iter& __x)
    {
        if (this != &__x)
        {
            __get_db()->__iterator_copy(this, &__x);
            __i = __x.__i;
        }
        return *this;
    }
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_IF_NODEBUG
    ~__wrap_iter()
    {
        __get_db()->__erase_i(this);
    }
#endif
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_IF_NODEBUG reference operator*() const _NOEXCEPT
    {
#if _LIBCPP_DEBUG_LEVEL == 2
        _LIBCPP_ASSERT(__get_const_db()->__dereferenceable(this),
                       "Attempted to dereference a non-dereferenceable iterator");
#endif
        return *__i;
    }
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_IF_NODEBUG pointer  operator->() const _NOEXCEPT
    {
#if _LIBCPP_DEBUG_LEVEL == 2
        _LIBCPP_ASSERT(__get_const_db()->__dereferenceable(this),
                       "Attempted to dereference a non-dereferenceable iterator");
#endif
        return (pointer)_VSTD::addressof(*__i);
    }
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_IF_NODEBUG __wrap_iter& operator++() _NOEXCEPT
    {
#if _LIBCPP_DEBUG_LEVEL == 2
        _LIBCPP_ASSERT(__get_const_db()->__dereferenceable(this),
                       "Attempted to increment non-incrementable iterator");
#endif
        ++__i;
        return *this;
    }
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_IF_NODEBUG __wrap_iter  operator++(int) _NOEXCEPT
        {__wrap_iter __tmp(*this); ++(*this); return __tmp;}

    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_IF_NODEBUG __wrap_iter& operator--() _NOEXCEPT
    {
#if _LIBCPP_DEBUG_LEVEL == 2
        _LIBCPP_ASSERT(__get_const_db()->__decrementable(this),
                       "Attempted to decrement non-decrementable iterator");
#endif
        --__i;
        return *this;
    }
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_IF_NODEBUG __wrap_iter  operator--(int) _NOEXCEPT
        {__wrap_iter __tmp(*this); --(*this); return __tmp;}
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_IF_NODEBUG __wrap_iter  operator+ (difference_type __n) const _NOEXCEPT
        {__wrap_iter __w(*this); __w += __n; return __w;}
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_IF_NODEBUG __wrap_iter& operator+=(difference_type __n) _NOEXCEPT
    {
#if _LIBCPP_DEBUG_LEVEL == 2
        _LIBCPP_ASSERT(__get_const_db()->__addable(this, __n),
                   "Attempted to add/subtract iterator outside of valid range");
#endif
        __i += __n;
        return *this;
    }
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_IF_NODEBUG __wrap_iter  operator- (difference_type __n) const _NOEXCEPT
        {return *this + (-__n);}
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_IF_NODEBUG __wrap_iter& operator-=(difference_type __n) _NOEXCEPT
        {*this += -__n; return *this;}
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_IF_NODEBUG reference    operator[](difference_type __n) const _NOEXCEPT
    {
#if _LIBCPP_DEBUG_LEVEL == 2
        _LIBCPP_ASSERT(__get_const_db()->__subscriptable(this, __n),
                   "Attempted to subscript iterator outside of valid range");
#endif
        return __i[__n];
    }

    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_IF_NODEBUG iterator_type base() const _NOEXCEPT {return __i;}

private:
#if _LIBCPP_DEBUG_LEVEL == 2
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_IF_NODEBUG __wrap_iter(const void* __p, iterator_type __x) : __i(__x)
    {
        __get_db()->__insert_ic(this, __p);
    }
#else
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_IF_NODEBUG __wrap_iter(iterator_type __x) _NOEXCEPT : __i(__x) {}
#endif

    template <class _Up> friend class __wrap_iter;
    template <class _CharT, class _Traits, class _Alloc> friend class basic_string;
    template <class _Tp, class _Alloc> friend class _LIBCPP_TEMPLATE_VIS vector;
    template <class _Tp, size_t> friend class _LIBCPP_TEMPLATE_VIS span;

    template <class _Iter1, class _Iter2>
    _LIBCPP_CONSTEXPR_IF_NODEBUG friend
    bool
    operator==(const __wrap_iter<_Iter1>&, const __wrap_iter<_Iter2>&) _NOEXCEPT;

    template <class _Iter1, class _Iter2>
    _LIBCPP_CONSTEXPR_IF_NODEBUG friend
    bool
    operator<(const __wrap_iter<_Iter1>&, const __wrap_iter<_Iter2>&) _NOEXCEPT;

    template <class _Iter1, class _Iter2>
    _LIBCPP_CONSTEXPR_IF_NODEBUG friend
    bool
    operator!=(const __wrap_iter<_Iter1>&, const __wrap_iter<_Iter2>&) _NOEXCEPT;

    template <class _Iter1, class _Iter2>
    _LIBCPP_CONSTEXPR_IF_NODEBUG friend
    bool
    operator>(const __wrap_iter<_Iter1>&, const __wrap_iter<_Iter2>&) _NOEXCEPT;

    template <class _Iter1, class _Iter2>
    _LIBCPP_CONSTEXPR_IF_NODEBUG friend
    bool
    operator>=(const __wrap_iter<_Iter1>&, const __wrap_iter<_Iter2>&) _NOEXCEPT;

    template <class _Iter1, class _Iter2>
    _LIBCPP_CONSTEXPR_IF_NODEBUG friend
    bool
    operator<=(const __wrap_iter<_Iter1>&, const __wrap_iter<_Iter2>&) _NOEXCEPT;

#ifndef _LIBCPP_CXX03_LANG
    template <class _Iter1, class _Iter2>
    _LIBCPP_CONSTEXPR_IF_NODEBUG friend
    auto
    operator-(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter2>& __y) _NOEXCEPT
    -> decltype(__x.base() - __y.base());
#else
    template <class _Iter1, class _Iter2>
    _LIBCPP_CONSTEXPR_IF_NODEBUG friend
    typename __wrap_iter<_Iter1>::difference_type
    operator-(const __wrap_iter<_Iter1>&, const __wrap_iter<_Iter2>&) _NOEXCEPT;
#endif

    template <class _Iter1>
    _LIBCPP_CONSTEXPR_IF_NODEBUG friend
    __wrap_iter<_Iter1>
    operator+(typename __wrap_iter<_Iter1>::difference_type, __wrap_iter<_Iter1>) _NOEXCEPT;

    template <class _Ip, class _Op> friend _LIBCPP_CONSTEXPR_AFTER_CXX17 _Op copy(_Ip, _Ip, _Op);
    template <class _B1, class _B2> friend _LIBCPP_CONSTEXPR_AFTER_CXX17 _B2 copy_backward(_B1, _B1, _B2);
    template <class _Ip, class _Op> friend _LIBCPP_CONSTEXPR_AFTER_CXX17 _Op move(_Ip, _Ip, _Op);
    template <class _B1, class _B2> friend _LIBCPP_CONSTEXPR_AFTER_CXX17 _B2 move_backward(_B1, _B1, _B2);
};

//------------------------
// end inject __wrap_iter
*/

//CP living dangerously
#define _LIBCPP_ASSERT(x, m) ((void)0)

//CP
//#if _LIBCPP_STD_VER > 17
#if _LIBCPP_STD_VER < 22

inline constexpr size_t dynamic_extent = numeric_limits<size_t>::max();
template <typename _Tp, size_t _Extent = dynamic_extent> class span;


template <class _Tp>
struct __is_span_impl : public false_type {};

template <class _Tp, size_t _Extent>
struct __is_span_impl<span<_Tp, _Extent>> : public true_type {};

template <class _Tp>
struct __is_span : public __is_span_impl<remove_cv_t<_Tp>> {};

template <class _Tp>
struct __is_std_array_impl : public false_type {};

template <class _Tp, size_t _Sz>
struct __is_std_array_impl<array<_Tp, _Sz>> : public true_type {};

template <class _Tp>
struct __is_std_array : public __is_std_array_impl<remove_cv_t<_Tp>> {};

template <class _Tp, class _ElementType, class = void>
struct __is_span_compatible_container : public false_type {};

template <class _Tp, class _ElementType>
struct __is_span_compatible_container<_Tp, _ElementType,
        void_t<
        // is not a specialization of span
            typename enable_if<!__is_span<_Tp>::value, nullptr_t>::type,
        // is not a specialization of array
            typename enable_if<!__is_std_array<_Tp>::value, nullptr_t>::type,
        // is_array_v<Container> is false,
            typename enable_if<!is_array_v<_Tp>, nullptr_t>::type,
        // data(cont) and size(cont) are well formed
            decltype(data(declval<_Tp>())),
            decltype(size(declval<_Tp>())),
        // remove_pointer_t<decltype(data(cont))>(*)[] is convertible to ElementType(*)[]
            typename enable_if<
                is_convertible_v<remove_pointer_t<decltype(data(declval<_Tp &>()))>(*)[],
                                 _ElementType(*)[]>,
                nullptr_t>::type
        >>
    : public true_type {};


template <typename _Tp, size_t _Extent>
class _LIBCPP_TEMPLATE_VIS span {
public:
//  constants and types
    using element_type           = _Tp;
    using value_type             = remove_cv_t<_Tp>;
    using size_type              = size_t;
    using difference_type        = ptrdiff_t;
    using pointer                = _Tp *;
    using const_pointer          = const _Tp *;
    using reference              = _Tp &;
    using const_reference        = const _Tp &;
    //CP
    //using iterator               =  __wrap_iter<pointer>;
    using iterator               = pointer;
    //using iterator               = iterator<pointer>;

    //using reverse_iterator       = _VSTD::reverse_iterator<iterator>;
    using reverse_iterator       = reverse_iterator<pointer>;

    static constexpr size_type extent = _Extent;

// [span.cons], span constructors, copy, assignment, and destructor
    template <size_t _Sz = _Extent, enable_if_t<_Sz == 0, nullptr_t> = nullptr>
    _LIBCPP_INLINE_VISIBILITY constexpr span() noexcept : __data{nullptr} {}

    constexpr span           (const span&) noexcept = default;
    constexpr span& operator=(const span&) noexcept = default;

    _LIBCPP_INLINE_VISIBILITY constexpr explicit span(pointer __ptr, size_type __count) : __data{__ptr}
        { (void)__count; _LIBCPP_ASSERT(_Extent == __count, "size mismatch in span's constructor (ptr, len)"); }
    _LIBCPP_INLINE_VISIBILITY constexpr explicit span(pointer __f, pointer __l) : __data{__f}
        { (void)__l;     _LIBCPP_ASSERT(_Extent == distance(__f, __l), "size mismatch in span's constructor (ptr, ptr)"); }

    _LIBCPP_INLINE_VISIBILITY constexpr span(element_type (&__arr)[_Extent])          noexcept : __data{__arr} {}

    template <class _OtherElementType,
              enable_if_t<is_convertible_v<_OtherElementType(*)[], element_type (*)[]>, nullptr_t> = nullptr>
    _LIBCPP_INLINE_VISIBILITY
    constexpr span(array<_OtherElementType, _Extent>& __arr) noexcept : __data{__arr.data()} {}

    template <class _OtherElementType,
              enable_if_t<is_convertible_v<const _OtherElementType(*)[], element_type (*)[]>, nullptr_t> = nullptr>
    _LIBCPP_INLINE_VISIBILITY
    constexpr span(const array<_OtherElementType, _Extent>& __arr) noexcept : __data{__arr.data()} {}

    template <class _Container>
    _LIBCPP_INLINE_VISIBILITY
        constexpr explicit span(      _Container& __c,
            enable_if_t<__is_span_compatible_container<_Container, _Tp>::value, nullptr_t> = nullptr)
        : __data{_VSTD::data(__c)} {  
            _LIBCPP_ASSERT(_Extent == _VSTD::size(__c), "size mismatch in span's constructor (range)");
        }

    template <class _Container>
    _LIBCPP_INLINE_VISIBILITY
        constexpr explicit span(const _Container& __c,
            enable_if_t<__is_span_compatible_container<const _Container, _Tp>::value, nullptr_t> = nullptr)
        : __data{_VSTD::data(__c)} {
            _LIBCPP_ASSERT(_Extent == _VSTD::size(__c), "size mismatch in span's constructor (range)");
        }

    template <class _OtherElementType>
    _LIBCPP_INLINE_VISIBILITY
        constexpr span(const span<_OtherElementType, _Extent>& __other,
                       enable_if_t<
                          is_convertible_v<_OtherElementType(*)[], element_type (*)[]>,
                          nullptr_t> = nullptr)
        : __data{__other.data()} {}

    template <class _OtherElementType>
    _LIBCPP_INLINE_VISIBILITY
        constexpr explicit span(const span<_OtherElementType, dynamic_extent>& __other,
                       enable_if_t<
                          is_convertible_v<_OtherElementType(*)[], element_type (*)[]>,
                          nullptr_t> = nullptr) noexcept
        : __data{__other.data()} { _LIBCPP_ASSERT(_Extent == __other.size(), "size mismatch in span's constructor (other span)"); }


//  ~span() noexcept = default;

    template <size_t _Count>
    _LIBCPP_INLINE_VISIBILITY
    constexpr span<element_type, _Count> first() const noexcept
    {
        static_assert(_Count <= _Extent, "Count out of range in span::first()");
        return span<element_type, _Count>{data(), _Count};
    }

    template <size_t _Count>
    _LIBCPP_INLINE_VISIBILITY
    constexpr span<element_type, _Count> last() const noexcept
    {
        static_assert(_Count <= _Extent, "Count out of range in span::last()");
        return span<element_type, _Count>{data() + size() - _Count, _Count};
    }

    _LIBCPP_INLINE_VISIBILITY
    constexpr span<element_type, dynamic_extent> first(size_type __count) const noexcept
    {
        _LIBCPP_ASSERT(__count <= size(), "Count out of range in span::first(count)");
        return {data(), __count};
    }

    _LIBCPP_INLINE_VISIBILITY
    constexpr span<element_type, dynamic_extent> last(size_type __count) const noexcept
    {
        _LIBCPP_ASSERT(__count <= size(), "Count out of range in span::last(count)");
        return {data() + size() - __count, __count};
    }

    template <size_t _Offset, size_t _Count = dynamic_extent>
    _LIBCPP_INLINE_VISIBILITY
    constexpr auto subspan() const noexcept
        -> span<element_type, _Count != dynamic_extent ? _Count : _Extent - _Offset>
    {
        static_assert(_Offset <= _Extent, "Offset out of range in span::subspan()");
        static_assert(_Count == dynamic_extent || _Count <= _Extent - _Offset, "Offset + count out of range in span::subspan()");

        using _ReturnType = span<element_type, _Count != dynamic_extent ? _Count : _Extent - _Offset>;
        return _ReturnType{data() + _Offset, _Count == dynamic_extent ? size() - _Offset : _Count};
    }


    _LIBCPP_INLINE_VISIBILITY
    constexpr span<element_type, dynamic_extent>
       subspan(size_type __offset, size_type __count = dynamic_extent) const noexcept
    {
        _LIBCPP_ASSERT(__offset <= size(), "Offset out of range in span::subspan(offset, count)");
        _LIBCPP_ASSERT(__count  <= size() || __count == dynamic_extent, "Count out of range in span::subspan(offset, count)");
        if (__count == dynamic_extent)
            return {data() + __offset, size() - __offset};
        _LIBCPP_ASSERT(__count <= size() - __offset, "Offset + count out of range in span::subspan(offset, count)");
        return {data() + __offset, __count};
    }

    _LIBCPP_INLINE_VISIBILITY constexpr size_type size()       const noexcept { return _Extent; }
    _LIBCPP_INLINE_VISIBILITY constexpr size_type size_bytes() const noexcept { return _Extent * sizeof(element_type); }
    _LIBCPP_INLINE_VISIBILITY constexpr bool empty()           const noexcept { return _Extent == 0; }

    _LIBCPP_INLINE_VISIBILITY constexpr reference operator[](size_type __idx) const noexcept
    {
        _LIBCPP_ASSERT(__idx < size(), "span<T,N>[] index out of bounds");
        return __data[__idx];
    }

    _LIBCPP_INLINE_VISIBILITY constexpr reference front() const noexcept
    {
        _LIBCPP_ASSERT(!empty(), "span<T, N>::front() on empty span");
        return __data[0];
    }

    _LIBCPP_INLINE_VISIBILITY constexpr reference back() const noexcept
    {
        _LIBCPP_ASSERT(!empty(), "span<T, N>::back() on empty span");
        return __data[size()-1];
    }

    _LIBCPP_INLINE_VISIBILITY constexpr pointer data()                         const noexcept { return __data; }

// [span.iter], span iterator support
    _LIBCPP_INLINE_VISIBILITY constexpr iterator                 begin() const noexcept { return iterator(data()); }
    _LIBCPP_INLINE_VISIBILITY constexpr iterator                   end() const noexcept { return iterator(data() + size()); }
    _LIBCPP_INLINE_VISIBILITY constexpr reverse_iterator        rbegin() const noexcept { return reverse_iterator(end()); }
    _LIBCPP_INLINE_VISIBILITY constexpr reverse_iterator          rend() const noexcept { return reverse_iterator(begin()); }

    _LIBCPP_INLINE_VISIBILITY span<const byte, _Extent * sizeof(element_type)> __as_bytes() const noexcept
    { return span<const byte, _Extent * sizeof(element_type)>{reinterpret_cast<const byte *>(data()), size_bytes()}; }

    _LIBCPP_INLINE_VISIBILITY span<byte, _Extent * sizeof(element_type)> __as_writable_bytes() const noexcept
    { return span<byte, _Extent * sizeof(element_type)>{reinterpret_cast<byte *>(data()), size_bytes()}; }

private:
    pointer    __data;

};


template <typename _Tp>
class _LIBCPP_TEMPLATE_VIS span<_Tp, dynamic_extent> {
private:

public:
//  constants and types
    using element_type           = _Tp;
    using value_type             = remove_cv_t<_Tp>;
    using size_type              = size_t;
    using difference_type        = ptrdiff_t;
    using pointer                = _Tp *;
    using const_pointer          = const _Tp *;
    using reference              = _Tp &;
    using const_reference        = const _Tp &;

    //CP
    //using iterator               =  __wrap_iter<pointer>;
    using iterator               = pointer;
    //using iterator               = iterator<pointer>;

    //using reverse_iterator       = _VSTD::reverse_iterator<iterator>;
    using reverse_iterator       = reverse_iterator<pointer>;

    static constexpr size_type extent = dynamic_extent;

// [span.cons], span constructors, copy, assignment, and destructor
    _LIBCPP_INLINE_VISIBILITY constexpr span() noexcept : __data{nullptr}, __size{0} {}

    constexpr span           (const span&) noexcept = default;
    constexpr span& operator=(const span&) noexcept = default;

    _LIBCPP_INLINE_VISIBILITY constexpr span(pointer __ptr, size_type __count) : __data{__ptr}, __size{__count} {}
    _LIBCPP_INLINE_VISIBILITY constexpr span(pointer __f, pointer __l) : __data{__f}, __size{static_cast<size_t>(distance(__f, __l))} {}

    template <size_t _Sz>
    _LIBCPP_INLINE_VISIBILITY
    constexpr span(element_type (&__arr)[_Sz])          noexcept : __data{__arr}, __size{_Sz} {}

    template <class _OtherElementType, size_t _Sz,
              enable_if_t<is_convertible_v<_OtherElementType(*)[], element_type (*)[]>, nullptr_t> = nullptr>
    _LIBCPP_INLINE_VISIBILITY
    constexpr span(array<_OtherElementType, _Sz>& __arr) noexcept : __data{__arr.data()}, __size{_Sz} {}

    template <class _OtherElementType, size_t _Sz,
              enable_if_t<is_convertible_v<const _OtherElementType(*)[], element_type (*)[]>, nullptr_t> = nullptr>
    _LIBCPP_INLINE_VISIBILITY
    constexpr span(const array<_OtherElementType, _Sz>& __arr) noexcept : __data{__arr.data()}, __size{_Sz} {}

    template <class _Container>
    _LIBCPP_INLINE_VISIBILITY
        constexpr span(      _Container& __c,
            enable_if_t<__is_span_compatible_container<_Container, _Tp>::value, nullptr_t> = nullptr)
        : __data{_VSTD::data(__c)}, __size{(size_type) _VSTD::size(__c)} {}

    template <class _Container>
    _LIBCPP_INLINE_VISIBILITY
        constexpr span(const _Container& __c,
            enable_if_t<__is_span_compatible_container<const _Container, _Tp>::value, nullptr_t> = nullptr)
        : __data{_VSTD::data(__c)}, __size{(size_type) _VSTD::size(__c)} {}


    template <class _OtherElementType, size_t _OtherExtent>
    _LIBCPP_INLINE_VISIBILITY
        constexpr span(const span<_OtherElementType, _OtherExtent>& __other,
                       enable_if_t<
                          is_convertible_v<_OtherElementType(*)[], element_type (*)[]>,
                          nullptr_t> = nullptr) noexcept
        : __data{__other.data()}, __size{__other.size()} {}

//    ~span() noexcept = default;

    template <size_t _Count>
    _LIBCPP_INLINE_VISIBILITY
    constexpr span<element_type, _Count> first() const noexcept
    {
        _LIBCPP_ASSERT(_Count <= size(), "Count out of range in span::first()");
        return span<element_type, _Count>{data(), _Count};
    }

    template <size_t _Count>
    _LIBCPP_INLINE_VISIBILITY
    constexpr span<element_type, _Count> last() const noexcept
    {
        _LIBCPP_ASSERT(_Count <= size(), "Count out of range in span::last()");
        return span<element_type, _Count>{data() + size() - _Count, _Count};
    }

    _LIBCPP_INLINE_VISIBILITY
    constexpr span<element_type, dynamic_extent> first(size_type __count) const noexcept
    {
        _LIBCPP_ASSERT(__count <= size(), "Count out of range in span::first(count)");
        return {data(), __count};
    }

    _LIBCPP_INLINE_VISIBILITY
    constexpr span<element_type, dynamic_extent> last (size_type __count) const noexcept
    {
        _LIBCPP_ASSERT(__count <= size(), "Count out of range in span::last(count)");
        return {data() + size() - __count, __count};
    }

    template <size_t _Offset, size_t _Count = dynamic_extent>
    _LIBCPP_INLINE_VISIBILITY
    constexpr span<element_type, _Count> subspan() const noexcept
    {
        _LIBCPP_ASSERT(_Offset <= size(), "Offset out of range in span::subspan()");
        _LIBCPP_ASSERT(_Count == dynamic_extent || _Count <= size() - _Offset, "Offset + count out of range in span::subspan()");
        return span<element_type, _Count>{data() + _Offset, _Count == dynamic_extent ? size() - _Offset : _Count};
    }

    constexpr span<element_type, dynamic_extent>
    _LIBCPP_INLINE_VISIBILITY
    subspan(size_type __offset, size_type __count = dynamic_extent) const noexcept
    {
        _LIBCPP_ASSERT(__offset <= size(), "Offset out of range in span::subspan(offset, count)");
        _LIBCPP_ASSERT(__count  <= size() || __count == dynamic_extent, "count out of range in span::subspan(offset, count)");
        if (__count == dynamic_extent)
            return {data() + __offset, size() - __offset};
        _LIBCPP_ASSERT(__count <= size() - __offset, "Offset + count out of range in span::subspan(offset, count)");
        return {data() + __offset, __count};
    }

    _LIBCPP_INLINE_VISIBILITY constexpr size_type size()       const noexcept { return __size; }
    _LIBCPP_INLINE_VISIBILITY constexpr size_type size_bytes() const noexcept { return __size * sizeof(element_type); }
    _LIBCPP_INLINE_VISIBILITY constexpr bool empty()           const noexcept { return __size == 0; }

    _LIBCPP_INLINE_VISIBILITY constexpr reference operator[](size_type __idx) const noexcept
    {
        _LIBCPP_ASSERT(__idx < size(), "span<T>[] index out of bounds");
        return __data[__idx];
    }

    _LIBCPP_INLINE_VISIBILITY constexpr reference front() const noexcept
    {
        _LIBCPP_ASSERT(!empty(), "span<T>[].front() on empty span");
        return __data[0];
    }

    _LIBCPP_INLINE_VISIBILITY constexpr reference back() const noexcept
    {
        _LIBCPP_ASSERT(!empty(), "span<T>[].back() on empty span");
        return __data[size()-1];
    }


    _LIBCPP_INLINE_VISIBILITY constexpr pointer data()                         const noexcept { return __data; }

// [span.iter], span iterator support
    _LIBCPP_INLINE_VISIBILITY constexpr iterator                 begin() const noexcept { return iterator(data()); }
    _LIBCPP_INLINE_VISIBILITY constexpr iterator                   end() const noexcept { return iterator(data() + size()); }
    _LIBCPP_INLINE_VISIBILITY constexpr reverse_iterator        rbegin() const noexcept { return reverse_iterator(end()); }
    _LIBCPP_INLINE_VISIBILITY constexpr reverse_iterator          rend() const noexcept { return reverse_iterator(begin()); }

    _LIBCPP_INLINE_VISIBILITY span<const byte, dynamic_extent> __as_bytes() const noexcept
    { return {reinterpret_cast<const byte *>(data()), size_bytes()}; }

    _LIBCPP_INLINE_VISIBILITY span<byte, dynamic_extent> __as_writable_bytes() const noexcept
    { return {reinterpret_cast<byte *>(data()), size_bytes()}; }

private:
    pointer   __data;
    size_type __size;
};

//  as_bytes & as_writable_bytes
template <class _Tp, size_t _Extent>
_LIBCPP_INLINE_VISIBILITY
auto as_bytes(span<_Tp, _Extent> __s) noexcept
-> decltype(__s.__as_bytes())
{ return    __s.__as_bytes(); }

template <class _Tp, size_t _Extent>
_LIBCPP_INLINE_VISIBILITY
auto as_writable_bytes(span<_Tp, _Extent> __s) noexcept
-> enable_if_t<!is_const_v<_Tp>, decltype(__s.__as_writable_bytes())>
{ return __s.__as_writable_bytes(); }

//  Deduction guides
template<class _Tp, size_t _Sz>
    span(_Tp (&)[_Sz]) -> span<_Tp, _Sz>;

template<class _Tp, size_t _Sz>
    span(array<_Tp, _Sz>&) -> span<_Tp, _Sz>;

template<class _Tp, size_t _Sz>
    span(const array<_Tp, _Sz>&) -> span<const _Tp, _Sz>;

template<class _Container>
    span(_Container&) -> span<typename _Container::value_type>;

template<class _Container>
    span(const _Container&) -> span<const typename _Container::value_type>;

#endif // _LIBCPP_STD_VER > 17

//_LIBCPP_END_NAMESPACE_STD
} // namespace std

_LIBCPP_POP_MACROS

#endif // _LIBCPP_SPAN
