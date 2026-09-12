//==-------------------- host_pipe_map.hpp -----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/export.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {
namespace host_pipe_map {

__SYCL_EXPORT void add(const void *HostPipePtr, const char *UniqueId);

} // namespace host_pipe_map
} // namespace detail
} // namespace _V1
} // namespace sycl
