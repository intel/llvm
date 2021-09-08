//==----------------- group_mask.hpp --- SYCL group mask -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <CL/__spirv/spirv_ops.hpp>
#include <CL/__spirv/spirv_vars.hpp>
#include <CL/sycl/exception.hpp>
#include <CL/sycl/id.hpp>
#include <CL/sycl/marray.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace oneapi {

struct group_mask {
  static constexpr size_t max_bits = 128 /* implementation-defined */;
  static constexpr size_t word_size = sizeof(uint32_t) * CHAR_BIT;
  /* Bitmask is packed in marray of uint32_t elements. This value represents
   * legth of marray. Round up in case when it is not evenly divisible. */
  static constexpr size_t marray_size = (max_bits + word_size - 1) / word_size;
  /* The bits are stored in the memory in the following way:
  marray id |     0     |     1     |     2     |     3     |
  bit id    |127 ..   96|95  ..   64|63  ..   32|31  ..    0|
  */

  // enable reference to individual bit
  struct reference {
    reference &operator=(bool x) {
      if (x) {
        Ref |= RefBit;
      } else {
        Ref &= ~RefBit;
      }
      return *this;
    }
    reference &operator=(const reference &x) {
      operator=((bool)x);
      return *this;
    }
    bool operator~() const { return !(Ref & RefBit); }
    operator bool() const { return Ref & RefBit; }
    reference &flip() {
      operator=(!(bool)*this);
      return *this;
    }

    reference(group_mask &gmask, size_t pos)
        : Ref(gmask.Bits[marray_size - (pos / word_size) - 1]) {
      RefBit = 1 << pos % word_size;
    }

  private:
    // Reference to the word containing the bit
    uint32_t &Ref;
    // Bit mask where only referenced bit is set
    uint32_t RefBit;
  };

  bool operator[](id<1> id) const {
    return Bits[marray_size - id.get(0) / word_size - 1] &
           (1 << (id.get(0) % word_size));
  }
  reference operator[](id<1> id) { return {*this, id.get(0)}; }
  bool test(id<1> id) const { return operator[](id); }
  bool all() const { return !(~(Bits[0] & Bits[1] & Bits[2] & Bits[3])); }
  bool any() const { return Bits[0] | Bits[1] | Bits[2] | Bits[3]; }
  bool none() const { return !any(); }
  uint32_t count() const {
    unsigned int count = 0;
    for (auto word : Bits) {
      while (word) {
        word &= (word - 1);
        count++;
      }
    }
    return count;
  }
  uint32_t size() const { return max_bits; }
  id<1> find_low() const {
    size_t i = 0;
    while (i < size() && !operator[](i))
      i++;
    return {i};
  }
  id<1> find_high() const {
    size_t i = size() - 1;
    while (i > 0 && !operator[](i))
      i--;
    return {operator[](i) ? i : size()};
  }

  template <typename T = marray<uint32_t, marray_size>>
  void insert_bits(const T &bits, id<1> pos = 0) {
    group_mask tmp(bits);
    if (pos.get(0) > 0) {
      operator<<=(max_bits - pos.get(0));
      operator>>=(max_bits - pos.get(0));
      tmp <<= pos.get(0);
    } else {
      reset();
    }
    Bits |= tmp.Bits;
  }

  template <typename T = marray<uint32_t, marray_size>>
  T extract_bits(id<1> pos = 0) {
    group_mask Tmp = *this;
    Tmp <<= pos.get(0);
    return Tmp.Bits;
  }

  void set() { Bits = ~(uint32_t{0}); }
  void set(id<1> id, bool value = true) { operator[](id) = value; }
  void reset() { Bits = uint32_t{0}; }
  void reset(id<1> id) { operator[](id) = 0; }
  void reset_low() { reset(find_low()); }
  void reset_high() { reset(find_high()); }
  void flip() { Bits = ~Bits; }
  void flip(id<1> id) { operator[](id).flip(); }

  bool operator==(const group_mask &rhs) const {
    bool Res = true;
    for (size_t i = 0; i < marray_size; i++)
      Res &= Bits[i] == rhs.Bits[i];
    return Res;
  }
  bool operator!=(const group_mask &rhs) const { return !(*this == rhs); }

  group_mask &operator&=(const group_mask &rhs) {
    Bits &= rhs.Bits;
    return *this;
  }
  group_mask &operator|=(const group_mask &rhs) {
    Bits |= rhs.Bits;
    return *this;
  }

  group_mask &operator^=(const group_mask &rhs) {
    Bits ^= rhs.Bits;
    return *this;
  }

  group_mask &operator<<=(size_t pos) {
    if (pos > 0) {
      marray<uint32_t, marray_size> Res{0};
      size_t word_shift = pos / word_size;
      size_t bit_shift = pos % word_size;
      uint32_t extra_bits = 0;
      for (int i = marray_size - 1; i >= 0; i--) {
        Res[i - word_shift] = (Bits[i] << bit_shift) + extra_bits;
        extra_bits = Bits[i] >> (word_size - bit_shift);
      }
      Bits = Res;
    }
    return *this;
  }

  group_mask &operator>>=(size_t pos) {
    if (pos > 0) {
      marray<uint32_t, marray_size> Res{0};
      size_t word_shift = pos / word_size;
      size_t bit_shift = pos % word_size;
      uint32_t extra_bits = 0;
      for (size_t i = 0; i < marray_size; i++) {
        Res[i + word_shift] = (Bits[i] >> bit_shift) + extra_bits;
        extra_bits = Bits[i] << (word_size - bit_shift);
      }
      Bits = Res;
    }
    return *this;
  }

  group_mask operator~() const {
    auto Tmp = *this;
    Tmp.flip();
    return Tmp;
  }
  group_mask operator<<(size_t pos) const {
    auto Tmp = *this;
    Tmp <<= pos;
    return Tmp;
  }
  group_mask operator>>(size_t pos) const {
    auto Tmp = *this;
    Tmp >>= pos;
    return Tmp;
  }

  group_mask(const group_mask &rhs) : Bits(rhs.Bits) {}
  template <typename Group>
  friend group_mask group_ballot(Group g, bool predicate);

  group_mask(const marray<uint32_t, marray_size> &rhs) : Bits(rhs) {}

  group_mask operator&(const group_mask &rhs) const {
    auto Res = *this;
    Res &= rhs;
    return Res;
  }
  group_mask operator|(const group_mask &rhs) const {
    auto Res = *this;
    Res |= rhs;
    return Res;
  }
  group_mask operator^(const group_mask &rhs) const {
    auto Res = *this;
    Res ^= rhs;
    return Res;
  }

private:
  marray<uint32_t, marray_size> Bits;
};
template <typename Group> group_mask group_ballot(Group g, bool predicate) {
  (void)g;
#ifdef __SYCL_DEVICE_ONLY__
  auto res = __spirv_GroupNonUniformBallot(
      detail::spirv::group_scope<Group>::value, predicate);
  return marray<uint32_t, group_mask::marray_size>{res[3], res[2],
                                                               res[1], res[0]};
#else
  (void)predicate;
  throw exception{errc::feature_not_supported,
                  "Group mask is not supported on host device"};
#endif
}
} // namespace oneapi
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
