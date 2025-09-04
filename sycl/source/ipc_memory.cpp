//==------- ipc_memory.cpp --- SYCL inter-process communicable memory ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/ipc_memory_impl.hpp>
#include <sycl/context.hpp>
#include <sycl/ext/oneapi/experimental/ipc_memory.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

ipc_memory::ipc_memory(void *Ptr, const sycl::context &Ctx)
    : impl(detail::ipc_memory_impl::create(Ptr, Ctx)) {}

ipc_memory::ipc_memory(
    span<const char, sycl::dynamic_extent> IPCMemoryHandleData,
    const sycl::context &Ctx, const sycl::device &Dev)
    : impl(detail::ipc_memory_impl::create(IPCMemoryHandleData, Ctx, Dev)) {}

span<const char, sycl::dynamic_extent> ipc_memory::get_handle_data() const {
  return impl->get_handle_data();
}

void *ipc_memory::get_ptr() const { return impl->get_ptr(); }

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
