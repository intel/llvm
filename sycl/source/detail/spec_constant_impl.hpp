//==-- spec_constant_impl.hpp - SYCL RT model for specialization constants -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/__impl/detail/defines.hpp>
#include <sycl/__impl/detail/util.hpp>
#include <sycl/__impl/stl.hpp>

#include <iostream>
#include <map>
#include <vector>

#ifdef __SYCL_ENABLE_SYCL121_NAMESPACE
__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
#else
namespace __sycl_internal {
inline namespace __v1 {
#endif
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
using SpecConstRegistryT = std::map<string_class, spec_constant_impl>;

void stableSerializeSpecConstRegistry(const SpecConstRegistryT &Reg,
                                      SerializedObj &Dst);

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
