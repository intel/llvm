//==------------ accessor_iterator.hpp - SYCL standard header file ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/id.hpp>

#include <iterator>
#include <type_traits>

/// \file accessor_iterator.hpp
/// The file contains implementation of accessor iterator class.
///
/// The reason why we can't use a plain pointer as an interator and have to
/// implement a custom class here is explained in section 4.7.6.8. Ranged
/// accessors of SYCL 2020 specification. A couple of quotes from there:
///
/// > Accessors of type accessor and host_accessor can be constructed from a
/// > sub-range of a buffer by providing a range and offset to the constructor.
/// >
/// > If the ranged accessor is multi-dimensional, the sub-range is allowed to
/// > describe a region of memory in the underlying buffer that is not
/// > contiguous in the linear address space.
/// >
/// > Most of the accessor member functions which provide a reference to the
/// > underlying buffer elements are affected by a ranged accessor’s offset and
/// > range. ... In addition, the accessor’s iterator functions iterate only
/// > over the elements that are within the sub-range.
///
/// Classes below implement the logic of iterating through N-dimensional
/// (1 <= N <= 3) space, which covers a potentially non-contiguous memory
/// region in the underlying accessor bufffer.
///
/// Most of the logic is implemented in __accessor_iterator_base class, which
/// provides routines for all the indexing logic such as
/// incrementing/decrementing iterators, addition/substraction and comparison
/// operators of iterators, etc.
///
/// Pointer to accessor is held by __accessor_iterator class, which provides
/// user-visible interface of iterator.

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
template <typename _DataT, int _Dimensions, access::mode _AccessMode,
          access::target _AccessTarget, access::placeholder _IsPlaceholder,
          typename _PropertyListT>
class accessor;

namespace detail {

/// Base class for accessor iterators, which implements common logic between
/// all iterators (const, reverse, const reverse, etc.)
///
/// In order to iterate through a possibly non-contiguous N-dimensional space,
/// the class holds an N-dimensional `id`, which is carefuly incremented each
/// time iterator is incremented/decrementing, taking into account the
/// shape/size of a space iterator goes through.
///
/// Whilst increment/decrement operation can be implemented through a couple of
/// 'if's and assignments, additon/substraction operators which can move an
/// iterator up to N elements, are harder to implement on a N-dimensional id.
/// In order to implement them, the class also holds and maintains a linearized
/// id, which can be quickly updated to perform an addition/substraction of an
/// iterator. However, that id has to be deleniarized in order to be used to
/// dereference particular element of an accessor and that operation includes
/// division and taking reminder of the division. Those operations are more
/// expensive than simple additional and conditionals and therefore the class
/// maintains both N-dimensional and linear id to balance between implementation
/// simplicity and performance of (presumably) most oftenly used operations with
/// an accessor.
template <int _Dimensions, bool _IsReverse = false>
class __accessor_iterator_base {
protected:
  using difference_type = size_t;
  using iterator_category = std::random_access_iterator_tag;

private:
  id<_Dimensions> _MBegin;
  // Holds an id which is relative to _MBegin.
  id<_Dimensions> _MCurrent;
  id<_Dimensions> _MEnd;

  static constexpr int _Index0 = _Dimensions - 1;
  static constexpr int _Index1 = _Dimensions - 2;
  static constexpr int _Index2 = _Dimensions - 3;

  static constexpr difference_type _LinearBegin = 0;
  // Holds an id which is relative to _LinearBegin
  difference_type _MLinearCurrent = 0;
  difference_type _MLinearEnd = 0;

  difference_type _MRowSize = 0;
  difference_type _MSliceSize = 0;

protected:
  __accessor_iterator_base() {}

  __accessor_iterator_base(const id<_Dimensions> &_Begin,
                           const id<_Dimensions> &_End,
                           const id<_Dimensions> &_Current)
      : _MBegin(_Begin), _MCurrent(_Current - _MBegin), _MEnd(_End) {
    _MLinearEnd = _MRowSize = _MEnd[_Index0] - _MBegin[_Index0];
    if constexpr (_Dimensions > 1) {
      _MSliceSize = (_MEnd[_Index1] - _MBegin[_Index1]) * _MRowSize;
      // Multiply by number of rows
      _MLinearEnd *= _MEnd[_Index1] - _MBegin[_Index1];
    }
    if constexpr (_Dimensions > 2) {
      // Multiply by number of slices
      _MLinearEnd *= _MEnd[_Index2] - _MBegin[_Index2];
    }
    _MLinearCurrent = __linearizeIndex(_MCurrent);
  }

  id<_Dimensions> __get_current_id() const { return _MBegin + _MCurrent; }

  __accessor_iterator_base &operator++() {
    if constexpr (_IsReverse)
      __decrement();
    else
      __increment();
    return *this;
  }

  __accessor_iterator_base operator++(int) {
    auto _Old = *this;
    ++(*this);
    return _Old;
  }

  __accessor_iterator_base &operator--() {
    if constexpr (_IsReverse)
      __increment();
    else
      __decrement();
    return *this;
  }

  __accessor_iterator_base operator--(int) {
    auto _Old = *this;
    --(*this);
    return _Old;
  }

  __accessor_iterator_base &operator+=(difference_type _N) {
    // iterator && N > 0 -> forward
    // iterator && N < 0 -> backwards
    // reverse iterator && N > 0 -> backwards
    // reverse iterator && N < 0 -> forward
    bool _BackwardsDirection = !_IsReverse ^ (_N > 0);
    if (_BackwardsDirection)
      __adjustBackwards(_N);
    else
      __adjustForward(_N);
    return *this;
  }

  __accessor_iterator_base &operator-=(difference_type _N) {
    // iterator && N > 0 -> backwards
    // iterator && N < 0 -> forward
    // reverse iterator && N > 0 -> forward
    // reverse iterator && N < 0 -> backwards
    bool _ForwardDirection = !_IsReverse ^ (_N > 0);
    if (_ForwardDirection)
      __adjustForward(_N);
    else
      __adjustBackwards(_N);
    return *this;
  }

  difference_type operator-(const __accessor_iterator_base &_Rhs) {
    if (_Rhs._MLinearCurrent > _MLinearCurrent)
      return _Rhs._MLinearCurrent - _MLinearCurrent;
    else
      return _MLinearCurrent - _Rhs._MLinearCurrent;
  }

  bool operator<(const __accessor_iterator_base<_Dimensions> &_Other) const {
    return _MLinearCurrent < _Other._MLinearCurrent;
  }

  bool operator>(const __accessor_iterator_base<_Dimensions> &_Other) const {
    return _Other < *this;
  }

  bool operator<=(const __accessor_iterator_base<_Dimensions> &_Other) const {
    return !(*this > _Other);
  }

  bool operator>=(const __accessor_iterator_base<_Dimensions> &_Other) const {
    return !(*this < _Other);
  }

  bool operator==(const __accessor_iterator_base<_Dimensions> &_Other) const {
    return _MLinearCurrent == _Other._MLinearCurrent;
  }

  bool operator!=(const __accessor_iterator_base<_Dimensions> &_Other) const {
    return !(*this == _Other);
  }

private:
  void __increment() {
    if (_MLinearCurrent >= _MLinearEnd)
      return;

    ++_MLinearCurrent;
    if (_MCurrent[_Index0] < _MEnd[_Index0])
      _MCurrent[_Index0]++;
    if constexpr (_Dimensions > 1) {
      if (_MCurrent[_Index0] == _MEnd[_Index0]) {
        if (_MCurrent[_Index1] < _MEnd[_Index1]) {
          _MCurrent[_Index1]++;
          _MCurrent[_Index0] = _MBegin[_Index0];
        }
      }
    }
    if constexpr (_Dimensions > 2) {
      if (_MCurrent[_Index1] == _MEnd[_Index1]) {
        if (_MCurrent[_Index2] < _MEnd[_Index2]) {
          _MCurrent[_Index2]++;
          _MCurrent[_Index0] = _MBegin[_Index0];
          _MCurrent[_Index1] = _MBegin[_Index1];
        }
      }
    }
  }

  void __decrement() {
    if (_MLinearCurrent == _LinearBegin)
      return;

    --_MLinearCurrent;
    if (_MCurrent[_Index0] > 0)
      _MCurrent[_Index0]--;
    if constexpr (_Dimensions > 1) {
      if (_MCurrent[_Index0] == 0) {
        if (_MCurrent[_Index1] > 0) {
          _MCurrent[_Index1]--;
          _MCurrent[_Index0] = _MEnd[_Index0] - 1;
        }
      }
    }
    if constexpr (_Dimensions > 2) {
      if (_MCurrent[_Index1] == 0) {
        if (_MCurrent[_Index2] > 0) {
          _MCurrent[_Index2]--;
          _MCurrent[_Index0] = _MEnd[_Index0] - 1;
          _MCurrent[_Index1] = _MEnd[_Index1] - 1;
        }
      }
    }
  }

  void __adjustForward(difference_type _N) {
    if (_MLinearCurrent + _N > _MLinearEnd)
      _MLinearCurrent = _MLinearEnd;
    else
      _MLinearCurrent += _N;
    _MCurrent = __delinearizeIndex(_MLinearCurrent);
  }

  void __adjustBackwards(difference_type _N) {
    if (_N > _MLinearCurrent)
      _MLinearCurrent = _LinearBegin;
    else
      _MLinearCurrent -= _N;
    _MCurrent = __delinearizeIndex(_MLinearCurrent);
  }

  size_t __linearizeIndex(const id<_Dimensions> &_Id) const {
    size_t _Result = _Id[_Index0];
    if constexpr (_Dimensions > 1)
      _Result += _Id[_Index1] * _MRowSize;
    if constexpr (_Dimensions > 2)
      _Result += _Id[_Index2] * _MSliceSize;
    return _Result;
  }

  id<_Dimensions> __delinearizeIndex(size_t _LinearId) const {
    id<_Dimensions> _Result;
    if constexpr (_Dimensions > 2) {
      _Result[_Index2] = _LinearId / _MSliceSize;
      _LinearId %= _MSliceSize;
    }
    if constexpr (_Dimensions > 1) {
      _Result[_Index1] = _LinearId / _MRowSize;
      _LinearId %= _MRowSize;
    }
    _Result[_Index0] = _LinearId;
    return _Result;
  }
};

template <typename _DataT, int _Dimensions, access::mode _AccessMode,
          access::target _AccessTarget, access::placeholder _IsPlaceholder,
          typename _PropertyListT>
class __accessor_iterator : public __accessor_iterator_base<_Dimensions> {
  using _AccessorT = accessor<_DataT, _Dimensions, _AccessMode, _AccessTarget,
                              _IsPlaceholder, _PropertyListT>;
  const _AccessorT *_MAccessorPtr;

  using _BaseT = __accessor_iterator_base<_Dimensions>;

  friend class accessor<_DataT, _Dimensions, _AccessMode, _AccessTarget,
                        _IsPlaceholder, _PropertyListT>;

  __accessor_iterator(const _AccessorT *_AccessorPtr,
                      const id<_Dimensions> &_Begin,
                      const id<_Dimensions> &_End,
                      const id<_Dimensions> &_Current)
      : __accessor_iterator_base<_Dimensions>(_Begin, _End, _Current),
        _MAccessorPtr(_AccessorPtr) {}

  static __accessor_iterator __get_begin(const _AccessorT *_AccessorPtr,
                                         const id<_Dimensions> &_Begin,
                                         const id<_Dimensions> &_End) {
    return __accessor_iterator(_AccessorPtr, _Begin, _End, _Begin);
  }

  static __accessor_iterator __get_end(const _AccessorT *_AccessorPtr,
                                       const id<_Dimensions> &_Begin,
                                       const id<_Dimensions> &_End) {
    // As `.end()` iterator we use an iterator which points to the first element
    // past the end of an accessible range. That is done to simplify the process
    // of transforming an iterator to an `.end()` state by incrementing it.
    //
    // However, `_End` id passed here highlights an accessible range and do not
    // point to the first element past the end of the accessible range in all
    // cases. For example, let's take a look at a case where we access a
    // 2-dimensional buffer of size 2x2. Inputs to this method will be:
    // _Begin: (0, 0; _End(2, 2):
    //   Begin Elem .
    //   Elem  Elem .
    //   .     .    End
    //
    // As showed above, _End simply defines the shape/size, but it doesn't point
    // to the element we would like it to point to. That happens because _End
    // passed here comes from an accessor range, which is 1-indexed. However,
    // accessor::operator[] accepts a 0-indexed id. In order to create a
    // past-the-end iterator, we convert _End id to a 0-indexed one,
    // create an interator out of it and then simply increment it.
    auto _EndCopy = _End;
    for (auto _I = 0; _I < _Dimensions; ++_I)
      _EndCopy[_I]--;

    auto _Ret = __accessor_iterator(_AccessorPtr, _Begin, _End, _EndCopy);
    return ++_Ret;
  }

public:
  using difference_type = typename _BaseT::difference_type;
  using value_type = _DataT;
  // FIXME: this should likely include address space
  using pointer = _DataT *;
  using reference = _DataT &;
  using iterator_category = typename _BaseT::iterator_category;

  __accessor_iterator() : _MAccessorPtr(nullptr) {}

  _DataT &operator*() {
    return _MAccessorPtr->operator[](this->__get_current_id());
  }

  __accessor_iterator &operator++() {
    _BaseT::operator++();
    return *this;
  }

  __accessor_iterator operator++(int) {
    auto _Old = *this;
    _BaseT::operator++();
    return _Old;
  }

  __accessor_iterator &operator--() {
    _BaseT::operator--();
    return *this;
  }

  __accessor_iterator operator--(int) {
    auto _Old = *this;
    _BaseT::operator--();
    return _Old;
  }

  __accessor_iterator &operator+=(difference_type _N) {
    _BaseT::operator+=(_N);
    return *this;
  }

  friend __accessor_iterator operator+(const __accessor_iterator &_Lhs,
                                       difference_type _N) {
    auto _Ret = _Lhs;
    _Ret += _N;
    return _Ret;
  }

  friend __accessor_iterator operator+(difference_type _N,
                                       const __accessor_iterator &_Rhs) {
    auto _Ret = _Rhs;
    _Ret += _N;
    return _Ret;
  }

  __accessor_iterator &operator-=(difference_type _N) {
    _BaseT::operator-=(_N);
    return *this;
  }

  friend __accessor_iterator operator-(__accessor_iterator &_Lhs,
                                       difference_type _N) {
    _Lhs -= _N;
    return _Lhs;
  }

  reference &operator[](difference_type _N) {
    auto _Copy = *this;
    _Copy += _N;
    return *_Copy;
  }

  using __accessor_iterator_base<_Dimensions>::operator-;
  using __accessor_iterator_base<_Dimensions>::operator==;
  using __accessor_iterator_base<_Dimensions>::operator!=;
  using __accessor_iterator_base<_Dimensions>::operator<;
  using __accessor_iterator_base<_Dimensions>::operator<=;
  using __accessor_iterator_base<_Dimensions>::operator>;
  using __accessor_iterator_base<_Dimensions>::operator>=;
};
} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
