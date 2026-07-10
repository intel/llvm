//==------- ipc_event.hpp -- SYCL inter-process for events -----------------==//
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
#include <sycl/event.hpp>
#include <sycl/ext/oneapi/experimental/reusable_events.hpp>
#include <sycl/ext/oneapi/properties/properties.hpp>
#include <sycl/platform.hpp>

#include "detail/ipc_common.hpp"

#include <cstddef>

#if __has_include(<span>)
#include <span>
#endif

namespace sycl {
inline namespace _V1 {

namespace detail {
__SYCL_EXPORT sycl::event openIPCEventHandle(const std::byte *HandleData,
                                             size_t HandleDataSize,
                                             const sycl::context &Ctx);
} // namespace detail

namespace ext::oneapi::experimental::ipc::event {

// get / put are exported from libsycl; declarations live in
// detail/ipc_common.hpp because they're friended on ipc::handle.

inline void put(handle &IpcHandle) {
  sycl::device Dev;
  sycl::context Ctx = Dev.get_platform().khr_get_default_context();
  return put(IpcHandle, Ctx);
}

inline sycl::event open(const ipc::handle_data_t &HandleData,
                        const sycl::context &Ctx) {
  return sycl::detail::openIPCEventHandle(HandleData.data(), HandleData.size(),
                                          Ctx);
}

inline sycl::event open(const ipc::handle_data_t &HandleData) {
  sycl::device Dev;
  sycl::context Ctx = Dev.get_platform().khr_get_default_context();
  return open(HandleData, Ctx);
}

#if __cpp_lib_span
inline sycl::event open(const ipc::handle_data_view_t &HandleDataView,
                        const sycl::context &Ctx) {
  return sycl::detail::openIPCEventHandle(HandleDataView.data(),
                                          HandleDataView.size(), Ctx);
}

inline sycl::event open(const ipc::handle_data_view_t &HandleDataView) {
  sycl::device Dev;
  sycl::context Ctx = Dev.get_platform().khr_get_default_context();
  return open(HandleDataView, Ctx);
}
#endif

} // namespace ext::oneapi::experimental::ipc::event
} // namespace _V1
} // namespace sycl

#endif
