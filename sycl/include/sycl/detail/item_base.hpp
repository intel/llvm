//==----------- item_base.hpp --- SYCL iteration ItemBase ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/id.hpp>    // for id
#include <sycl/range.hpp> // for range

#include <stddef.h> // for size_t

namespace sycl {
inline namespace _V1 {
template <int dimensions> class id;
template <int dimensions> class range;

namespace detail {
template <int Dims, bool WithOffset> struct ItemBase;

template <int Dims> struct ItemBase<Dims, true> {

  bool operator==(const ItemBase &Rhs) const {
    return (Rhs.MIndex == MIndex) && (Rhs.MExtent == MExtent) &&
           (Rhs.MOffset == MOffset);
  }

  bool operator!=(const ItemBase &Rhs) const { return !((*this) == Rhs); }

  size_t get_linear_id() const {
    if (1 == Dims) {
      return MIndex[0] - MOffset[0];
    }
    if (2 == Dims) {
      return (MIndex[0] - MOffset[0]) * MExtent[1] + (MIndex[1] - MOffset[1]);
    }
    return ((MIndex[0] - MOffset[0]) * MExtent[1] * MExtent[2]) +
           ((MIndex[1] - MOffset[1]) * MExtent[2]) + (MIndex[2] - MOffset[2]);
  }

  range<Dims> MExtent;
  id<Dims> MIndex;
  id<Dims> MOffset;
};

template <int Dims> struct ItemBase<Dims, false> {

  bool operator==(const ItemBase &Rhs) const {
    return (Rhs.MIndex == MIndex) && (Rhs.MExtent == MExtent);
  }

  bool operator!=(const ItemBase &Rhs) const { return !((*this) == Rhs); }

  operator ItemBase<Dims, true>() const {
    return ItemBase<Dims, true>(MExtent, MIndex, id<Dims>{});
  }

  size_t get_linear_id() const {
    if (1 == Dims) {
      return MIndex[0];
    }
    if (2 == Dims) {
      return MIndex[0] * MExtent[1] + MIndex[1];
    }
    return (MIndex[0] * MExtent[1] * MExtent[2]) + (MIndex[1] * MExtent[2]) +
           MIndex[2];
  }

  range<Dims> MExtent;
  id<Dims> MIndex;
};

} // namespace detail
} // namespace _V1
} // namespace sycl
