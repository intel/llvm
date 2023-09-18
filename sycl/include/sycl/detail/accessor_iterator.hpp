//==------------ accessor_iterator.hpp - SYCL standard header file ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/access/access.hpp> // for mode, placeholder, target
#include <sycl/buffer.hpp>        // for range
#include <sycl/id.hpp>            // for id

#include <cstddef>  // for size_t
#include <iterator> // for random_access_iterator_tag
#include <ostream>  // for operator<<, ostream, ptrdiff_t

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
inline namespace _V1 {

template <typename AccessorDataT, int AccessorDimensions,
          access::mode AccessMode, access::target AccessTarget,
          access::placeholder IsPlaceholder, typename PropertyListT>
class accessor;

namespace detail {

template <typename DataT, int Dimensions> class accessor_iterator {
public:
  using difference_type = std::ptrdiff_t;
  using value_type = DataT;
  // FIXME: this should likely include address space
  using pointer = DataT *;
  using reference = DataT &;
  using iterator_category = std::random_access_iterator_tag;

  accessor_iterator() = default;

  reference operator*() const {
    return *(MDataPtr + getAbsoluteOffsetToBuffer());
  }

  accessor_iterator &operator++() {
    ++MLinearId;
    return *this;
  }

  accessor_iterator operator++(int) {
    auto Old = *this;
    ++(*this);
    return Old;
  }

  accessor_iterator &operator--() {
    --MLinearId;
    return *this;
  }

  accessor_iterator operator--(int) {
    auto Old = *this;
    --(*this);
    return Old;
  }

  accessor_iterator &operator+=(difference_type N) {
    MLinearId += N;

    return *this;
  }

  accessor_iterator operator+(difference_type N) const {
    auto Ret = *this;
    Ret += N;
    return Ret;
  }

  friend accessor_iterator operator+(difference_type N,
                                     const accessor_iterator &Rhs) {
    auto Ret = Rhs;
    Ret += N;
    return Ret;
  }

  accessor_iterator &operator-=(difference_type N) {
    MLinearId -= N;

    return *this;
  }

  accessor_iterator operator-(difference_type N) const {
    auto Temp = *this;
    return Temp -= N;
  }

  reference &operator[](difference_type N) const {
    auto Copy = *this;
    Copy += N;
    return *Copy;
  }

  bool operator<(const accessor_iterator &Other) const {
    return MLinearId < Other.MLinearId;
  }

  bool operator>(const accessor_iterator &Other) const { return Other < *this; }

  bool operator<=(const accessor_iterator &Other) const {
    return !(*this > Other);
  }

  bool operator>=(const accessor_iterator &Other) const {
    return !(*this < Other);
  }

  bool operator==(const accessor_iterator &Other) const {
    return MLinearId == Other.MLinearId;
  }

  bool operator!=(const accessor_iterator &Other) const {
    return !(*this == Other);
  }

  difference_type operator-(const accessor_iterator &Rhs) const {
    return MLinearId - Rhs.MLinearId;
  }

private:
  template <typename AccessorDataT, int AccessorDimensions,
            access::mode AccessMode, access::target AccessTarget,
            access::placeholder IsPlaceholder, typename PropertyListT>
  friend class sycl::accessor;

  DataT *MDataPtr = nullptr;

  // Stores a linear id of an accessor's buffer element the iterator points to.
  // This id is relative to a range accessible through an accessor, i.e. it is
  // limited by a space with top left corner defiend as accessor::get_offset()
  // and bottom right corner defined as accesor::get_range().
  size_t MLinearId = 0;

  // Describes range of linear IDs accessible by the iterator. MEnd corresponds
  // to ID of en element past the last accessible element of accessors's
  // buffer.
  size_t MBegin = 0;
  size_t MEnd = 0;

  // If set to true, then it indicates that accessor has its offset and/or range
  // set to non-zero, i.e. it is a ranged accessor.
  bool MAccessorIsRanged = false;

  // Fields below are used (and changed to be non-zero) only if we deal with
  // a ranged accessor.
  //
  // TODO: consider making their existance dependable on Dimensions template
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
  // MStaticOffset stores a number of elements which precede the first
  // accessible element, calculated as if the buffer was linearized.
  // For the example above, MStaticOffset would be equal to 6, because
  // there is one full row before the first accessible element and a one more on
  // the second line. "Static" in the name highlights that this is a constant
  // element in an equation which calculates an absoulte offset to an accessor's
  // buffer, it doesn't depend on the current state of the iterator.
  //
  // NOTE: MStaticOffset is set to 0 in 1D case even if the accessor was
  // created with offset: it is done to further optimize 1D case by
  // incorporating that offset into MLinearId right away.
  //
  // MPerRowOffset stores a number of _inaccessible_ elements in each
  // _accessible_ row. For the example above it would be equal to 2 (leftmost
  // and the rightmost elements of a row).
  //
  // MPerSliceOffset stores a number of _inaccessible_ elements in each
  // _accessible_ slice. Slice here means a single 2D layer in a 3D buffer. For
  // the example above it would be equal to 0, because we are not looking at a
  // 3D buffer. However, if we had two slices like visualized above,
  // MPerSliceOffset would be equal to 16 (elements on the "perimeter" of the
  // slice, i.e. ones represented as dots (.)).

  size_t MStaticOffset = 0;
  size_t MPerRowOffset = 0;
  size_t MPerSliceOffset = 0;

  // Contains a number of _accessible_ elements in a row
  size_t MRowSize = 0;
  // Contains a number of _accessible_ elements in a slice
  size_t MSliceSize = 0;

  // MLinearId stores an offset which is relative to the accessible range of
  // the accessor, which means that it could be the case that MlinearId equal
  // to 0 should not correspond to the beginning of the underlying buffer, but
  // instead should be re-adjusted to account for an offset passed to the
  // accessor constructor.
  //
  // This function performs necessary calculations to make sure that all
  // access ranges and offsets are taken into account.
  size_t getAbsoluteOffsetToBuffer() const {
    // For 1D case, any possible offsets are already incorporated into
    // MLinearId, so 1D is always treated as a non-ranged accessor
    if (!MAccessorIsRanged || Dimensions == 1)
      return MLinearId;

    // Here we need to deal with 2D or 3D ranged accessor.
    // MLinearId points to an element relative to the accessible range. It
    // should be adjusted to account for elements which are outside of the
    // accessible range of the accessor.

    // We start with static offset: that is a number of elements in full rows
    // and full slices before the first accessible element.
    size_t AbsoluteId = MLinearId + MStaticOffset;

    // Then we account for inaccessible elements in each full slice
    size_t Remaining = MLinearId;
    if constexpr (Dimensions == 3) {
      AbsoluteId += MPerSliceOffset * (Remaining / MSliceSize);
      Remaining %= MSliceSize;
    }

    // Then we account for inaccessible elements in each full row
    AbsoluteId += MPerRowOffset * (Remaining / MRowSize);
    Remaining %= MRowSize;

    return AbsoluteId;
  }

  accessor_iterator(DataT *DataPtr, const range<Dimensions> &MemoryRange,
                    const range<Dimensions> &AccessRange,
                    const id<Dimensions> &Offset)
      : MDataPtr(DataPtr) {
    constexpr int XIndex = Dimensions - 1;
    constexpr int YIndex = Dimensions - 2;
    (void)YIndex;
    constexpr int ZIndex = Dimensions - 3;
    (void)ZIndex;

    if constexpr (Dimensions > 1)
      MRowSize = AccessRange[XIndex];
    if constexpr (Dimensions > 2)
      MSliceSize = AccessRange[YIndex] * MRowSize;

    if (id<Dimensions>{} != Offset)
      MAccessorIsRanged = true;
    else {
      for (size_t I = 0; I < Dimensions; ++I)
        if (AccessRange[I] != MemoryRange[I])
          MAccessorIsRanged = true;
    }

    if (MAccessorIsRanged) {
      if constexpr (Dimensions > 2) {
        MStaticOffset +=
            MemoryRange[XIndex] * MemoryRange[YIndex] * Offset[ZIndex];
        MPerSliceOffset =
            MemoryRange[XIndex] * MemoryRange[YIndex] - MSliceSize;
      }
      if constexpr (Dimensions > 1) {
        // Elements in fully inaccessible rows
        MStaticOffset += MemoryRange[XIndex] * Offset[YIndex];
        MPerRowOffset = MemoryRange[XIndex] - MRowSize;
      }

      // Elements from the first accessible row
      if constexpr (Dimensions == 1)
        // To further optimize 1D case, offset is already included into Begin
        MBegin = Offset[XIndex];
      else
        MStaticOffset += Offset[XIndex];
    }

    MEnd = MBegin + AccessRange.size();
  }

  static accessor_iterator getBegin(DataT *DataPtr,
                                    const range<Dimensions> &MemoryRange,
                                    const range<Dimensions> &AccessRange,
                                    const id<Dimensions> &Offset) {
    auto It = accessor_iterator(DataPtr, MemoryRange, AccessRange, Offset);
    It.MLinearId = It.MBegin;
    return It;
  }

  static accessor_iterator getEnd(DataT *DataPtr,
                                  const range<Dimensions> &MemoryRange,
                                  const range<Dimensions> &AccessRange,
                                  const id<Dimensions> &Offset) {
    auto It = accessor_iterator(DataPtr, MemoryRange, AccessRange, Offset);
    It.MLinearId = It.MEnd;
    return It;
  }

public:
#ifndef NDEBUG
  // Could be useful for debugging, but not a part of the official API,
  // therefore only available in builds with assertions enabled.
  friend std::ostream &operator<<(std::ostream &os,
                                  const accessor_iterator &it) {
    os << "accessor_iterator {\n";
    os << "\tMLinearId: " << it.MLinearId << "\n";
    os << "\tMEnd: " << it.MEnd << "\n";
    os << "\tMStaticOffset: " << it.MStaticOffset << "\n";
    os << "\tMPerRowOffset: " << it.MPerRowOffset << "\n";
    os << "\tMPerSliceOffset: " << it.MPerSliceOffset << "\n";
    os << "\tMRowSize: " << it.MRowSize << "\n";
    os << "\tMSliceSize: " << it.MSliceSize << "\n";
    os << "\tMAccessorIsRanged: " << it.MAccessorIsRanged << "\n";
    os << "}";
    return os;
  }
#endif // NDEBUG
};
} // namespace detail
} // namespace _V1
} // namespace sycl
