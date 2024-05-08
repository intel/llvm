//==-- spec_constant_impl.hpp - SYCL RT model for specialization constants -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines.hpp>
#include <sycl/detail/iostream_proxy.hpp>
#include <sycl/detail/util.hpp>

#include <map>
#include <vector>

namespace sycl {
inline namespace _V1 {
namespace detail {

// Represents a specialization constant value in SYCL runtime.
class spec_constant_impl {
public:
  spec_constant_impl() = default;

  spec_constant_impl(size_t Size, const void *Val) { set(Size, Val); }

  void set(size_t Size, const void *Val);

  size_t getSize() const { return Bytes.size(); }
  const char *getValuePtr() const { return Bytes.data(); }
  bool isSet() const { return !Bytes.empty(); }

private:
  std::vector<char> Bytes;
};

std::ostream &operator<<(std::ostream &Out, const spec_constant_impl &V);

// Used to define specialization constant registry. Must be ordered map, since
// the order of entries matters in stableSerializeSpecConstRegistry.
using SpecConstRegistryT = std::map<std::string, spec_constant_impl>;

void stableSerializeSpecConstRegistry(const SpecConstRegistryT &Reg,
                                      SerializedObj &Dst);

} // namespace detail
} // namespace _V1
} // namespace sycl
