//==-------------- half_type.hpp --- SYCL half type ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <functional>

namespace cl {
namespace sycl {
namespace detail {
namespace half_impl {

class half {
public:
  half() = default;
  half(const half &) = default;
  half(half &&) = default;

  half(const float &rhs);

  half &operator=(const half &rhs) = default;

  // Operator +=, -=, *=, /=
  half &operator+=(const half &rhs);

  half &operator-=(const half &rhs);

  half &operator*=(const half &rhs);

  half &operator/=(const half &rhs);

  // Operator ++, --
  half &operator++() {
    *this += 1;
    return *this;
  }

  half operator++(int) {
    half ret(*this);
    operator++();
    return ret;
  }

  half &operator--() {
    *this -= 1;
    return *this;
  }

  half operator--(int) {
    half ret(*this);
    operator--();
    return ret;
  }

  // Operator float
  operator float() const;

  template <typename Key> friend struct std::hash;

private:
  uint16_t Buf;
};
} // namespace half_impl

// Accroding to C++ standard math functions from cmath/math.h should work only
// on arithmetic types. We can't specify half type as arithmetic/floating
// point(via std::is_floating_point) since only float, double and long double
// types are "floating point" according to the standard. In order to use half
// type with these math functions we cast half to float using template function
// helper.
template <typename T> inline T cast_if_host_half(T val) { return val; }

inline float cast_if_host_half(half_impl::half val) {
  return static_cast<float>(val);
}

} // namespace detail

} // namespace sycl
} // namespace cl

namespace std {

template <> struct hash<cl::sycl::detail::half_impl::half> {
  size_t operator()(cl::sycl::detail::half_impl::half const &key) const
      noexcept {
    return hash<uint16_t>()(key.Buf);
  }
};

} // namespace std
