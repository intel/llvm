//==---------- scheduler_helpers.hpp - SYCL standard header file -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/defines_elementary.hpp>

#include <memory>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

class stream_impl;
class queue_impl;

using StreamImplPtr = std::shared_ptr<detail::stream_impl>;
using QueueImplPtr = std::shared_ptr<detail::queue_impl>;

void initStream(StreamImplPtr Stream, QueueImplPtr Queue);

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
