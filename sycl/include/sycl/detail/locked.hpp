//==---------------- locked.hpp - Reference with lock -----------*- C++-*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines.hpp>

#include <functional>
#include <mutex>

namespace sycl {
inline namespace _V1 {
namespace detail {
/// Represents a reference to value with appropriate lock acquired.
/// Employed for acquire/release logic. Acquire action is construction
/// of instance of locked<>. Release action is destruction of instance of
/// locked<>.
template <typename T> class Locked {
  std::reference_wrapper<T> m_Value;
  std::unique_lock<std::mutex> m_Lock;

public:
  Locked(T &v, std::mutex &mutex) : m_Value{v}, m_Lock{mutex} {}

  T &get() const { return m_Value.get(); }
};
} // namespace detail
} // namespace _V1
} // namespace sycl
