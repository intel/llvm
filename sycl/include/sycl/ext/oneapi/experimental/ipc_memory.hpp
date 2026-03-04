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

#include <cstddef>

#if __has_include(<span>)
#include <span>
#endif

namespace sycl {
inline namespace _V1 {

namespace detail {
__SYCL_EXPORT void *openIPCMemHandle(const std::byte *HandleData,
                                     size_t HandleDataSize,
                                     const sycl::context &Ctx,
                                     const sycl::device &Dev);
}

namespace ext::oneapi::experimental::ipc_memory {

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

  friend __SYCL_EXPORT handle get(void *Ptr, const sycl::context &Ctx);
  friend __SYCL_EXPORT void put(handle &HandleData, const sycl::context &Ctx);
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

inline void *open(const handle_data_t &HandleData, const sycl::context &Ctx,
                  const sycl::device &Dev) {
  return sycl::detail::openIPCMemHandle(HandleData.data(), HandleData.size(),
                                        Ctx, Dev);
}

inline void *open(handle_data_t HandleData, const sycl::device &Dev) {
  sycl::context Ctx = Dev.get_platform().khr_get_default_context();
  return ipc_memory::open(HandleData, Ctx, Dev);
}

inline void *open(handle_data_t HandleData) {
  sycl::device Dev;
  sycl::context Ctx = Dev.get_platform().khr_get_default_context();
  return ipc_memory::open(HandleData, Ctx, Dev);
}

#if __cpp_lib_span
inline void *open(const handle_data_view_t &HandleDataView,
                  const sycl::context &Ctx, const sycl::device &Dev) {
  return sycl::detail::openIPCMemHandle(HandleDataView.data(),
                                        HandleDataView.size(), Ctx, Dev);
}

inline void *open(handle_data_view_t HandleDataView, const sycl::device &Dev) {
  sycl::context Ctx = Dev.get_platform().khr_get_default_context();
  return ipc_memory::open(HandleDataView, Ctx, Dev);
}

inline void *open(handle_data_view_t HandleDataView) {
  sycl::device Dev;
  sycl::context Ctx = Dev.get_platform().khr_get_default_context();
  return ipc_memory::open(HandleDataView, Ctx, Dev);
}
#endif

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
