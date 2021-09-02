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
  using WordType = uint32_t;
  static constexpr size_t max_bits = 128 /* implementation-defined */;
  static constexpr size_t word_size = sizeof(WordType) * CHAR_BIT;
  /* Bitmask is packed in marray of uint32_t elements. This value represents
   * legth of marray. */
  static constexpr size_t marray_size = max_bits / word_size;

  // enable reference to individual bit
  struct reference {
    reference &operator=(bool x) {
      Ref &= x;
      return *this;
    }
    reference &operator=(const reference &x) {
      Ref &= (bool)x;
      return *this;
    }
    bool operator~() const { return !(Ref & RefBit); }
    operator bool() const { return Ref & RefBit; }
    reference &flip() {
      if ((bool)*this) {
        Ref &= ~RefBit;
      } else {
        Ref |= RefBit;
      }
      return *this;
    }

    reference(group_mask &gmask, size_t pos)
        : Ref(gmask.Bits[pos / word_size]) {
      size_t WordPos = pos;
      while (WordPos -= word_size && WordPos)
        WordPos = pos;
      RefBit = 1 << WordPos;
    }
  private:
    WordType &Ref;
    WordType RefBit;
  };

  bool operator[](id<1> id) const;
  reference operator[](id<1> id) {
    return {*this, id.get(0)};
  }
  bool test(id<1> id) const;
  bool all() const;
  bool any() const;
  bool none() const;
  uint32_t count() const;
  uint32_t size() const;
  id<1> find_low() const;
  id<1> find_high() const;

  template <typename T = marray<WordType, marray_size>>
  void insert_bits(const T &bits, id<1> pos = 0);

  template <typename T = marray<WordType, marray_size>>
  T extract_bits(id<1> pos = 0) {
    T Res = Bits;
    Res <<= pos;
    return Res;
  }

  void set();
  void set(id<1> id, bool value = true);
  void reset();
  void reset(id<1> id);
  void reset_low();
  void reset_high();
  void flip();
  void flip(id<1> id);

  bool operator==(const group_mask &rhs) const { return Bits == rhs.Bits; }
  bool operator!=(const group_mask &rhs) const { return Bits != rhs.Bits; }

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

  group_mask &operator<<=(size_t);
  group_mask &operator>>=(size_t rhs);

  group_mask operator~() const {
    auto Tmp = *this;
    Tmp.flip();
    return Tmp;
  }
  group_mask &operator<<(size_t) const;
  group_mask &operator>>(size_t) const;
  group_mask(const group_mask &rhs) : Bits(rhs.Bits) {}
  template <typename Group>
  friend group_mask group_ballot(Group g, bool predicate);

protected:
  group_mask(const marray<WordType, marray_size> &rhs) : Bits(rhs) {}
  marray<WordType, marray_size> Bits;
};

group_mask operator&(const group_mask &lhs, const group_mask &rhs);
group_mask operator|(const group_mask &lhs, const group_mask &rhs);
group_mask operator^(const group_mask &lhs, const group_mask &rhs);

template <typename Group> group_mask group_ballot(Group g, bool predicate) {
  (void)g;
#ifdef __SYCL_DEVICE_ONLY__
  auto res = __spirv_GroupNonUniformBallot(
      detail::spirv::group_scope<Group>::value, predicate);
  return marray<group_mask::WordType, group_mask::marray_size>{res[3], res[2],
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
