//==----------- functional.hpp --- SYCL functional -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

namespace cl {
namespace sycl {
namespace intel {

template <typename T = void> struct minimum {
  T operator()(const T &lhs, const T &rhs) const {
    return (lhs <= rhs) ? lhs : rhs;
  }
};

template <> struct minimum<void> {
  template <typename T> T operator()(const T &lhs, const T &rhs) const {
    return (lhs <= rhs) ? lhs : rhs;
  }
};

template <typename T = void> struct maximum {
  T operator()(const T &lhs, const T &rhs) const {
    return (lhs >= rhs) ? lhs : rhs;
  }
};

template <> struct maximum<void> {
  template <typename T> T operator()(const T &lhs, const T &rhs) const {
    return (lhs >= rhs) ? lhs : rhs;
  }
};

template <typename T = void> struct plus {
  T operator()(const T &lhs, const T &rhs) const { return lhs + rhs; }
};

template <> struct plus<void> {
  template <typename T> T operator()(const T &lhs, const T &rhs) const {
    return lhs + rhs;
  }
};

} // namespace intel
} // namespace sycl
} // namespace cl
