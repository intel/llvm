//==------- ipc_common.hpp ------- SYCL inter-process common ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#if (!defined(_HAS_STD_BYTE) || _HAS_STD_BYTE != 0)

#include <sycl/context.hpp>
#include <sycl/detail/defines_elementary.hpp>
#include <sycl/detail/export.hpp>

#include <cstddef>
#include <vector>

#if __has_include(<span>)
#include <span>
#endif

namespace sycl {
inline namespace _V1 {

namespace ext::oneapi::experimental::ipc {
struct handle;
}

namespace ext::oneapi::experimental::ipc::memory {
__SYCL_EXPORT handle get(void *Ptr, const sycl::context &Ctx);
__SYCL_EXPORT void put(handle &HandleData, const sycl::context &Ctx);
} // namespace ext::oneapi::experimental::ipc::memory

namespace ext::oneapi::experimental {
class physical_mem;
} // namespace ext::oneapi::experimental

namespace ext::oneapi::experimental::ipc::physical_memory {
__SYCL_EXPORT handle get(physical_mem &physmem);
__SYCL_EXPORT void put(handle &ipc_handle, const sycl::context &ctx);
} // namespace ext::oneapi::experimental::ipc::physical_memory

namespace ext::oneapi::experimental::ipc {

using handle_data_t = std::vector<std::byte>;

#if __cpp_lib_span
using handle_data_view_t = std::span<const std::byte, std::dynamic_extent>;
#endif

struct handle {
public:
  handle_data_t data() const { return {MData, MData + MSize}; }

#if __cpp_lib_span
  handle_data_view_t data_view() const { return {MData, MSize}; }
#endif

private:
  handle(void *Data, size_t Size)
      : MData{reinterpret_cast<std::byte *>(Data)}, MSize{Size} {}

  std::byte *MData;
  size_t MSize;

  friend __SYCL_EXPORT handle memory::get(void *Ptr, const sycl::context &Ctx);
  friend __SYCL_EXPORT void memory::put(handle &HandleData,
                                        const sycl::context &Ctx);
  friend __SYCL_EXPORT handle physical_memory::get(physical_mem &physmem);
  friend __SYCL_EXPORT void physical_memory::put(handle &ipc_handle,
                                                 const sycl::context &ctx);
};

} // namespace ext::oneapi::experimental::ipc
} // namespace _V1
} // namespace sycl

#endif
