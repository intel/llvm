//==------- ipc_memory.hpp --- SYCL inter-process communicable memory ------==//
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
#include <sycl/device.hpp>
#include <sycl/platform.hpp>
#include <sycl/sycl_span.hpp>

#include <cstddef>

namespace sycl {
inline namespace _V1 {

namespace ext::oneapi::experimental::ipc_memory {

using handle_data_t = sycl::span<const std::byte, dynamic_extent>;

struct handle {
public:
  handle_data_t data() const {
    return {reinterpret_cast<std::byte *>(MData), MSize};
  }

private:
  handle(void *Data, size_t Size) : MData{Data}, MSize{Size} {}

  void *MData;
  size_t MSize;

  friend handle get(void *Ptr, const sycl::context &Ctx);
  friend void put(handle &HandleData, const sycl::context &Ctx);
};

__SYCL_EXPORT handle get(void *Ptr, const sycl::context &Ctx);

inline handle get(void *Ptr) {
  sycl::device Dev;
  sycl::context Ctx = Dev.get_platform().khr_get_default_context();
  return ipc_memory::get(Ptr, Ctx);
}

__SYCL_EXPORT void put(handle &HandleData, const sycl::context &Ctx);

inline void put(handle &HandleData) {
  sycl::device Dev;
  sycl::context Ctx = Dev.get_platform().khr_get_default_context();
  ipc_memory::put(HandleData, Ctx);
}

__SYCL_EXPORT void *open(handle_data_t HandleData, const sycl::context &Ctx,
                         const sycl::device &Dev);

inline void *open(handle_data_t HandleData, const sycl::device &Dev) {
  sycl::context Ctx = Dev.get_platform().khr_get_default_context();
  return ipc_memory::open(HandleData, Ctx, Dev);
}

inline void *open(handle_data_t HandleData) {
  sycl::device Dev;
  sycl::context Ctx = Dev.get_platform().khr_get_default_context();
  return ipc_memory::open(HandleData, Ctx, Dev);
}

__SYCL_EXPORT void close(void *Ptr, const sycl::context &Ctx);

inline void close(void *Ptr) {
  sycl::device Dev;
  sycl::context Ctx = Dev.get_platform().khr_get_default_context();
  ipc_memory::close(Ptr, Ctx);
}

} // namespace ext::oneapi::experimental::ipc_memory
} // namespace _V1
} // namespace sycl

#endif
