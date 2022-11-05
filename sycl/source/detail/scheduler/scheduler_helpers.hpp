//==---------- scheduler_helpers.hpp - SYCL standard header file -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines_elementary.hpp>

#include <memory>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {

class stream_impl;
class queue_impl;

using StreamImplPtr = std::shared_ptr<detail::stream_impl>;
using QueueImplPtr = std::shared_ptr<detail::queue_impl>;

void initStream(StreamImplPtr Stream, QueueImplPtr Queue);

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
