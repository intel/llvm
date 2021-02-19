//==----- group_local_memory.hpp --- SYCL group local memory extension -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <CL/sycl/detail/defines_elementary.hpp>
#include <CL/sycl/detail/sycl_fe_intrins.hpp>
#include <CL/sycl/detail/type_traits.hpp>
#include <CL/sycl/multi_ptr.hpp>

#include <cstdint>
#include <type_traits>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

template <typename T, typename Group>
std::enable_if_t<std::is_trivially_destructible<T>::value &&
                     detail::is_group<Group>::value,
                 multi_ptr<T, access::address_space::local_space>>
    __SYCL_ALWAYS_INLINE group_local_memory_for_overwrite(Group g) {
  (void)g;
#ifdef __SYCL_DEVICE_ONLY__
  __attribute__((opencl_local)) std::uint8_t *AllocatedMem =
      __sycl_allocateLocalMemory(sizeof(T), alignof(T));
  return reinterpret_cast<__attribute__((opencl_local)) T *>(AllocatedMem);
#else
  throw feature_not_supported(
      "SYCL_INTEL_local_memory extension is not supported on host device",
      PI_INVALID_OPERATION);
#endif
}

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
