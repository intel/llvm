//==------------ accessor_iterator.hpp - SYCL standard header file ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/id.hpp>

#include <cstddef>
#include <iterator>
#include <ostream>
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

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {

template <typename _AccessorDataT, int _AccessorDimensions,
          access::mode _AccessMode, access::target _AccessTarget,
          access::placeholder _IsPlaceholder, typename _PropertyListT>
class accessor;

namespace detail {

template <typename _DataT, int _Dimensions> class __accessor_iterator {
public:
  using difference_type = std::ptrdiff_t;
  using value_type = _DataT;
  // FIXME: this should likely include address space
  using pointer = _DataT *;
  using reference = _DataT &;
  using iterator_category = std::random_access_iterator_tag;

  __accessor_iterator() = default;

  reference operator*() {
    return *(_MDataPtr + __get_absolute_offset_to_buffer());
  }

  __accessor_iterator &operator++() {
    if (_MLinearId < _MEnd)
      ++_MLinearId;
    return *this;
  }

  __accessor_iterator operator++(int) {
    auto _Old = *this;
    ++(*this);
    return _Old;
  }

  __accessor_iterator &operator--() {
    if (_MLinearId > _MBegin)
      --_MLinearId;
    return *this;
  }

  __accessor_iterator operator--(int) {
    auto _Old = *this;
    --(*this);
    return _Old;
  }

  __accessor_iterator &operator+=(difference_type _N) {
    if (_N < 0) {
      *this -= -_N;
      return *this;
    }

    if (static_cast<size_t>(_N) > _MEnd || _MEnd - _N < _MLinearId)
      _MLinearId = _MEnd;
    else
      _MLinearId += _N;

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
    if (_N < 0) {
      *this += -_N;
      return *this;
    }

    if (_MBegin + _N > _MLinearId)
      _MLinearId = _MBegin;
    else
      _MLinearId -= _N;

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

  bool operator<(const __accessor_iterator &_Other) const {
    return _MLinearId < _Other._MLinearId;
  }

  bool operator>(const __accessor_iterator &_Other) const {
    return _Other < *this;
  }

  bool operator<=(const __accessor_iterator &_Other) const {
    return !(*this > _Other);
  }

  bool operator>=(const __accessor_iterator &_Other) const {
    return !(*this < _Other);
  }

  bool operator==(const __accessor_iterator &_Other) const {
    return _MLinearId == _Other._MLinearId;
  }

  bool operator!=(const __accessor_iterator &_Other) const {
    return !(*this == _Other);
  }

  difference_type operator-(const __accessor_iterator &_Rhs) {
    return _MLinearId - _Rhs._MLinearId;
  }

private:
  template <typename _AccessorDataT, int _AccessorDimensions,
            access::mode _AccessMode, access::target _AccessTarget,
            access::placeholder _IsPlaceholder, typename _PropertyListT>
  friend class sycl::accessor;

  _DataT *_MDataPtr = nullptr;

  // Stores a linear id of an accessor's buffer element the iterator points to.
  // This id is relative to a range accessible through an accessor, i.e. it is
  // limited by a space with top left corner defiend as accessor::get_offset()
  // and bottom right corner defined as accesor::get_range().
  size_t _MLinearId = 0;

  // Describes range of linear IDs accessible by the iterator. _MEnd corresponds
  // to ID of en element past the last accessible element of accessors's
  // buffer.
  size_t _MBegin = 0;
  size_t _MEnd = 0;

  // If set to true, then it indicates that accessor has its offset and/or range
  // set to non-zero, i.e. it is a ranged accessor.
  bool _MAccessorIsRanged = false;

  // Fields below are used (and changed to be non-zero) only if we deal with
  // a ranged accessor.
  //
  // TODO: consider making their existance dependable on _Dimensions template
  // parameter, because not all of them are needed for all possible dimensions.

  // Three field below allow us to calculate an absolute offset to an accessor's
  // buffer to correctly identify a memory region which this iterator should
  // point to. Comments below describe them using an iterator to the following
  // accessor as an example:
  //
  //   buffer<int, 2> buf(input.data(), range<2>{5, 5});
  //   auto acc = buf.get_access(range<2>{3, 3}, id<2>{1, 1});
  //
  // Such combination of buffer size, access range and offset is visualized
  // below. Dot (.) symbols represent buffer elements NOT reacheable by the
  // accessor; X symbols represent buffer elements which ARE reachable by the
  // the accessor.
  //
  //   . . . . .
  //   . X X X .
  //   . X X X .
  //   . X X X .
  //   . . . . .
  //
  // _MStaticOffset stores a number of elements which precede the first
  // accessible element, calculated as if the buffer was linearized.
  // For the example above, _MStaticOffset would be equal to 6, because
  // there is one full row before the first accessible element and a one more on
  // the second line. "Static" in the name highlights that this is a constant
  // element in an equation which calculates an absoulte offset to an accessor's
  // buffer, it doesn't depend on the current state of the iterator.
  //
  // NOTE: _MStaticOffset is set to 0 in 1D case even if the accessor was
  // created with offset: it is done to further optimize 1D case by
  // incorporating that offset into _MLinearId right away.
  //
  // _MPerRowOffset stores a number of _inaccessible_ elements in each
  // _accessible_ row. For the example above it would be equal to 2 (leftmost
  // and the rightmost elements of a row).
  //
  // _MPerSliceOffset stores a number of _inaccessible_ elements in each
  // _accessible_ slice. Slice here means a single 2D layer in a 3D buffer. For
  // the example above it would be equal to 0, because we are not looking at a
  // 3D buffer. However, if we had two slices like visualized above,
  // _MPerSliceOffset would be equal to 16 (elements on the "perimeter" of the
  // slice, i.e. ones represented as dots (.)).

  size_t _MStaticOffset = 0;
  size_t _MPerRowOffset = 0;
  size_t _MPerSliceOffset = 0;

  // Contains a number of _accessible_ elements in a row
  size_t _MRowSize = 0;
  // Contains a number of _accessible_ elements in a slice
  size_t _MSliceSize = 0;

  // _MLinearId stores an offset which is relative to the accessible range of
  // the accessor, which means that it could be the case that _MlinearId equal
  // to 0 should not correspond to the beginning of the underlying buffer, but
  // instead should be re-adjusted to account for an offset passed to the
  // accessor constructor.
  //
  // This function performs necessary calculations to make sure that all
  // access ranges and offsets are taken into account.
  size_t __get_absolute_offset_to_buffer() {
    // For 1D case, any possible offsets are already incorporated into
    // _MLinearId, so 1D is always treated as a non-ranged accessor
    if (!_MAccessorIsRanged || _Dimensions == 1)
      return _MLinearId;

    // Here we need to deal with 2D or 3D ranged accessor.
    // _MLinearId points to an element relative to the accessible range. It
    // should be adjusted to account for elements which are outside of the
    // accessible range of the accessor.

    // We start with static offset: that is a number of elements in full rows
    // and full slices before the first accessible element.
    size_t _AbsoluteId = _MLinearId + _MStaticOffset;

    // Then we account for inaccessible elements in each full slice
    size_t _Remaining = _MLinearId;
    if constexpr (_Dimensions == 3) {
      _AbsoluteId += _MPerSliceOffset * (_Remaining / _MSliceSize);
      _Remaining %= _MSliceSize;
    }

    // Then we account for inaccessible elements in each full row
    _AbsoluteId += _MPerRowOffset * (_Remaining / _MRowSize);
    _Remaining %= _MRowSize;

    return _AbsoluteId;
  }

  __accessor_iterator(_DataT *_DataPtr, const range<_Dimensions> &_MemoryRange,
                      const range<_Dimensions> &_AccessRange,
                      const id<_Dimensions> &_Offset)
      : _MDataPtr(_DataPtr) {
    constexpr int _XIndex = _Dimensions - 1;
    constexpr int _YIndex = _Dimensions - 2;
    (void)_YIndex;
    constexpr int _ZIndex = _Dimensions - 3;
    (void)_ZIndex;

    if constexpr (_Dimensions > 1)
      _MRowSize = _AccessRange[_XIndex];
    if constexpr (_Dimensions > 2)
      _MSliceSize = _AccessRange[_YIndex] * _MRowSize;

    if (id<_Dimensions>{} != _Offset)
      _MAccessorIsRanged = true;
    else {
      for (size_t _I = 0; _I < _Dimensions; ++_I)
        if (_AccessRange[_I] != _MemoryRange[_I])
          _MAccessorIsRanged = true;
    }

    if (_MAccessorIsRanged) {
      if constexpr (_Dimensions > 2) {
        _MStaticOffset +=
            _MemoryRange[_XIndex] * _MemoryRange[_YIndex] * _Offset[_ZIndex];
        _MPerSliceOffset =
            _MemoryRange[_XIndex] * _MemoryRange[_YIndex] - _MSliceSize;
      }
      if constexpr (_Dimensions > 1) {
        // Elements in fully inaccessible rows
        _MStaticOffset += _MemoryRange[_XIndex] * _Offset[_YIndex];
        _MPerRowOffset = _MemoryRange[_XIndex] - _MRowSize;
      }

      // Elements from the first accessible row
      if constexpr (_Dimensions == 1)
        // To further optimize 1D case, offset is already included into _Begin
        _MBegin = _Offset[_XIndex];
      else
        _MStaticOffset += _Offset[_XIndex];
    }

    _MEnd = _MBegin + _AccessRange.size();
  }

  static __accessor_iterator __get_begin(_DataT *_DataPtr,
                                         const range<_Dimensions> &_MemoryRange,
                                         const range<_Dimensions> &_AccessRange,
                                         const id<_Dimensions> &_Offset) {
    auto _It =
        __accessor_iterator(_DataPtr, _MemoryRange, _AccessRange, _Offset);
    _It._MLinearId = _It._MBegin;
    return _It;
  }

  static __accessor_iterator __get_end(_DataT *_DataPtr,
                                       const range<_Dimensions> &_MemoryRange,
                                       const range<_Dimensions> &_AccessRange,
                                       const id<_Dimensions> &_Offset) {
    auto _It =
        __accessor_iterator(_DataPtr, _MemoryRange, _AccessRange, _Offset);
    _It._MLinearId = _It._MEnd;
    return _It;
  }

public:
#ifndef NDEBUG
  // Could be useful for debugging, but not a part of the official API,
  // therefore only available in builds with assertions enabled.
  friend std::ostream &operator<<(std::ostream &os,
                                  const __accessor_iterator &it) {
    os << "__accessor_iterator {\n";
    os << "\t_MLinearId: " << it._MLinearId << "\n";
    os << "\t_MEnd: " << it._MEnd << "\n";
    os << "\t_MStaticOffset: " << it._MStaticOffset << "\n";
    os << "\t_MPerRowOffset: " << it._MPerRowOffset << "\n";
    os << "\t_MPerSliceOffset: " << it._MPerSliceOffset << "\n";
    os << "\t_MRowSize: " << it._MRowSize << "\n";
    os << "\t_MSliceSize: " << it._MSliceSize << "\n";
    os << "\t_MAccessorIsRanged: " << it._MAccessorIsRanged << "\n";
    os << "}";
    return os;
  }
#endif // NDEBUG
};
} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
