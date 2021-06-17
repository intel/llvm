//==----- group_local_memory.hpp --- SYCL group local memory extension -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <sycl/__spirv/spirv_vars.hpp>
#include <sycl/__impl/detail/defines_elementary.hpp>
#include <sycl/__impl/detail/sycl_fe_intrins.hpp>
#include <sycl/__impl/detail/type_traits.hpp>
#include <sycl/__impl/exception.hpp>
#include <sycl/__impl/group.hpp>
#include <sycl/__impl/multi_ptr.hpp>

#include <cstdint>
#include <type_traits>
#include <utility>

namespace __sycl_internal {
inline namespace __v1 {

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

template <typename T, typename Group, typename... Args>
std::enable_if_t<std::is_trivially_destructible<T>::value &&
                     detail::is_group<Group>::value,
                 multi_ptr<T, access::address_space::local_space>>
    __SYCL_ALWAYS_INLINE group_local_memory(Group g, Args &&... args) {
  (void)g;
#ifdef __SYCL_DEVICE_ONLY__
  __attribute__((opencl_local)) std::uint8_t *AllocatedMem =
      __sycl_allocateLocalMemory(sizeof(T), alignof(T));

  // TODO switch to using group::get_local_linear_id here once it's implemented
  id<3> Id = __spirv::initLocalInvocationId<3, id<3>>();
  if (Id == id<3>(0, 0, 0))
    new (AllocatedMem) T(std::forward<Args>(args)...);
  detail::workGroupBarrier();
  return reinterpret_cast<__attribute__((opencl_local)) T *>(AllocatedMem);
#else
  // Silence unused variable warning
  [&args...] {}();
  throw feature_not_supported(
      "SYCL_INTEL_local_memory extension is not supported on host device",
      PI_INVALID_OPERATION);
#endif
}
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

#ifdef __SYCL_ENABLE_SYCL121_NAMESPACE
__SYCL_INLINE_NAMESPACE(cl) {
#endif
namespace sycl {
  using namespace __sycl_internal::__v1;
}
#ifdef __SYCL_ENABLE_SYCL121_NAMESPACE
}
#endif
