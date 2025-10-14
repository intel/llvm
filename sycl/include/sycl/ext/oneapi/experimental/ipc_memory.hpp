//==------- ipc_memory.hpp --- SYCL inter-process communicable memory ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#if (!defined(_HAS_STD_BYTE) || _HAS_STD_BYTE != 0)

#include <sycl/detail/defines_elementary.hpp>
#include <sycl/detail/export.hpp>

#include <cstddef>
#include <vector>

namespace sycl {
inline namespace _V1 {

class context;
class device;

namespace ext::oneapi::experimental::ipc_memory {

using handle_data_t = std::vector<std::byte>;

__SYCL_EXPORT handle_data_t get(void *Ptr, const sycl::context &Ctx);

__SYCL_EXPORT void put(handle_data_t &HandleData, const sycl::context &Ctx);

__SYCL_EXPORT void *open(handle_data_t &HandleData, const sycl::context &Ctx,
                         const sycl::device &Dev);

__SYCL_EXPORT void close(void *Ptr, const sycl::context &Ctx);

} // namespace ext::oneapi::experimental::ipc_memory
} // namespace _V1
} // namespace sycl

#endif
