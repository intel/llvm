//==------- ipc_physical_memory.cpp -- SYCL inter-process for physical mem -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/adapter_impl.hpp>
#include <detail/context_impl.hpp>
#include <sycl/context.hpp>
#include <sycl/ext/oneapi/experimental/ipc_physical_memory.hpp>

namespace sycl {
inline namespace _V1 {

namespace detail {

__SYCL_EXPORT sycl::ext::oneapi::experimental::physical_mem
openIPCPhysicalMemHandle(const std::byte *HandleData, size_t HandleDataSize,
                         const sycl::context &Ctx, const sycl::device &Dev) {
  if (!Dev.has(aspect::ext_oneapi_ipc_physical_memory))
    throw sycl::exception(
        sycl::make_error_code(errc::feature_not_supported),
        "Device does not support aspect::ext_oneapi_ipc_physical_memory.");

  return sycl::ext::oneapi::experimental::physical_mem{Dev, Ctx, 0};
}

} // namespace detail

namespace ext::oneapi::experimental::ipc::physical_memory {

__SYCL_EXPORT handle get(physical_mem &physmem) {
  void *HandlePtr = nullptr;
  size_t HandleSize = 0;

  return {HandlePtr, HandleSize};
}

__SYCL_EXPORT void put(handle &ipc_handle, const sycl::context &ctx) {}

} // namespace ext::oneapi::experimental::ipc::physical_memory
} // namespace _V1
} // namespace sycl
