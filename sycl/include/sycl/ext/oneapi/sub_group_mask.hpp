//==------------ sub_group_mask.hpp --- SYCL sub-group mask ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <CL/__spirv/spirv_ops.hpp>
#include <CL/__spirv/spirv_vars.hpp>
#include <sycl/detail/helpers.hpp>
#include <sycl/exception.hpp>
#include <sycl/id.hpp>
#include <sycl/marray.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {
class Builder;
} // namespace detail

namespace ext::oneapi {

#if defined(__SYCL_DEVICE_ONLY__) && defined(__AMDGCN__) &&                    \
    (__AMDGCN_WAVEFRONT_SIZE == 64)
#define BITS_TYPE uint64_t
#else
#define BITS_TYPE uint32_t
#endif

struct sub_group_mask {
  friend class detail::Builder;
  using BitsType = BITS_TYPE;

  static constexpr size_t max_bits =
      sizeof(BitsType) * CHAR_BIT /* implementation-defined */;
  static constexpr size_t word_size = sizeof(uint32_t) * CHAR_BIT;

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

    reference(sub_group_mask &gmask, size_t pos) : Ref(gmask.Bits) {
      RefBit = (pos < gmask.bits_num) ? (1UL << pos) : 0;
    }

  private:
    // Reference to the word containing the bit
    BitsType &Ref;
    // Bit mask where only referenced bit is set
    BitsType RefBit;
  };

  bool operator[](id<1> id) const {
    return (Bits & ((id.get(0) < bits_num) ? (1UL << id.get(0)) : 0));
  }

  reference operator[](id<1> id) { return {*this, id.get(0)}; }
  bool test(id<1> id) const { return operator[](id); }
  bool all() const { return count() == bits_num; }
  bool any() const { return count() != 0; }
  bool none() const { return count() == 0; }
  uint32_t count() const {
    unsigned int count = 0;
    auto word = (Bits & valuable_bits(bits_num));
    while (word) {
      word &= (word - 1);
      count++;
    }
    return count;
  }
  uint32_t size() const { return bits_num; }
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

  template <typename Type,
            typename = std::enable_if_t<std::is_integral<Type>::value>>
  void insert_bits(Type bits, id<1> pos = 0) {
    size_t insert_size = sizeof(Type) * CHAR_BIT;
    BitsType insert_data = (BitsType)bits;
    insert_data <<= pos.get(0);
    BitsType mask = 0;
    if (pos.get(0) + insert_size < size())
      mask |= (valuable_bits(bits_num) << (pos.get(0) + insert_size));
    if (pos.get(0) < size() && pos.get(0))
      mask |= (valuable_bits(max_bits) >> (max_bits - pos.get(0)));
    Bits &= mask;
    Bits += insert_data;
  }

  /* The bits are stored in the memory in the following way:
  marray id |     0     |     1     |     2     |     3     |...
  bit id    |7   ..    0|15   ..   8|23   ..  16|31  ..   24|...
  */
  template <typename Type, size_t Size,
            typename = std::enable_if_t<std::is_integral<Type>::value>>
  void insert_bits(const marray<Type, Size> &bits, id<1> pos = 0) {
    size_t cur_pos = pos.get(0);
    for (auto elem : bits) {
      if (cur_pos < size()) {
        this->insert_bits(elem, cur_pos);
        cur_pos += sizeof(Type) * CHAR_BIT;
      }
    }
  }

  template <typename Type,
            typename = std::enable_if_t<std::is_integral<Type>::value>>
  void extract_bits(Type &bits, id<1> pos = 0) const {
    auto Res = Bits;
    Res &= valuable_bits(bits_num);
    if (pos.get(0) < size()) {
      if (pos.get(0) > 0) {
        Res >>= pos.get(0);
      }

      if (sizeof(Type) * CHAR_BIT < max_bits) {
        Res &= valuable_bits(sizeof(Type) * CHAR_BIT);
      }
      bits = (Type)Res;
    } else {
      bits = 0;
    }
  }

  template <typename Type, size_t Size,
            typename = std::enable_if_t<std::is_integral<Type>::value>>
  void extract_bits(marray<Type, Size> &bits, id<1> pos = 0) const {
    size_t cur_pos = pos.get(0);
    for (auto &elem : bits) {
      if (cur_pos < size()) {
        this->extract_bits(elem, cur_pos);
        cur_pos += sizeof(Type) * CHAR_BIT;
      } else {
        elem = 0;
      }
    }
  }

  void set() { Bits = valuable_bits(bits_num); }
  void set(id<1> id, bool value = true) { operator[](id) = value; }
  void reset() { Bits = BitsType{0}; }
  void reset(id<1> id) { operator[](id) = 0; }
  void reset_low() { reset(find_low()); }
  void reset_high() { reset(find_high()); }
  void flip() { Bits = (~Bits & valuable_bits(bits_num)); }
  void flip(id<1> id) { operator[](id).flip(); }

  bool operator==(const sub_group_mask &rhs) const { return Bits == rhs.Bits; }
  bool operator!=(const sub_group_mask &rhs) const { return !(*this == rhs); }

  sub_group_mask &operator&=(const sub_group_mask &rhs) {
    Bits &= rhs.Bits;
    return *this;
  }
  sub_group_mask &operator|=(const sub_group_mask &rhs) {
    Bits |= rhs.Bits;
    return *this;
  }

  sub_group_mask &operator^=(const sub_group_mask &rhs) {
    Bits ^= rhs.Bits;
    Bits &= valuable_bits(bits_num);
    return *this;
  }

  sub_group_mask &operator<<=(size_t pos) {
    Bits <<= pos;
    Bits &= valuable_bits(bits_num);
    return *this;
  }

  sub_group_mask &operator>>=(size_t pos) {
    Bits >>= pos;
    return *this;
  }

  sub_group_mask operator~() const {
    auto Tmp = *this;
    Tmp.flip();
    return Tmp;
  }
  sub_group_mask operator<<(size_t pos) const {
    auto Tmp = *this;
    Tmp <<= pos;
    return Tmp;
  }
  sub_group_mask operator>>(size_t pos) const {
    auto Tmp = *this;
    Tmp >>= pos;
    return Tmp;
  }

  sub_group_mask(const sub_group_mask &rhs)
      : Bits(rhs.Bits), bits_num(rhs.bits_num) {}

  template <typename Group>
  friend std::enable_if_t<std::is_same<std::decay_t<Group>, sub_group>::value,
                          sub_group_mask>
  group_ballot(Group g, bool predicate);

  friend sub_group_mask operator&(const sub_group_mask &lhs,
                                  const sub_group_mask &rhs) {
    auto Res = lhs;
    Res &= rhs;
    return Res;
  }

  friend sub_group_mask operator|(const sub_group_mask &lhs,
                                  const sub_group_mask &rhs) {
    auto Res = lhs;
    Res |= rhs;
    return Res;
  }

  friend sub_group_mask operator^(const sub_group_mask &lhs,
                                  const sub_group_mask &rhs) {
    auto Res = lhs;
    Res ^= rhs;
    return Res;
  }

private:
  sub_group_mask(BitsType rhs, size_t bn) : Bits(rhs), bits_num(bn) {
    assert(bits_num <= max_bits);
  }
  inline BitsType valuable_bits(size_t bn) const {
    assert(bn <= max_bits);
    BitsType one = 1;
    if (bn == max_bits)
      return -one;
    return (one << bn) - one;
  }
  BitsType Bits;
  // Number of valuable bits
  size_t bits_num;
};

template <typename Group>
std::enable_if_t<std::is_same<std::decay_t<Group>, sub_group>::value,
                 sub_group_mask>
group_ballot(Group g, bool predicate) {
  (void)g;
#ifdef __SYCL_DEVICE_ONLY__
  auto res = __spirv_GroupNonUniformBallot(
      detail::spirv::group_scope<Group>::value, predicate);
  BITS_TYPE val = res[0];
  if constexpr (sizeof(BITS_TYPE) == 8)
    val |= ((BITS_TYPE)res[1]) << 32;
  return detail::Builder::createSubGroupMask<sub_group_mask>(
      val, g.get_max_local_range()[0]);
#else
  (void)predicate;
  throw exception{errc::feature_not_supported,
                  "Sub-group mask is not supported on host device"};
#endif
}

#undef BITS_TYPE

} // namespace ext::oneapi
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
