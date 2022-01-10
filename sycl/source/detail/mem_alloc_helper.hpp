//==-------- mem_alloc_helper.hpp - SYCL mem alloc helper ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/pi.h>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
void memBufferCreateHelper(const plugin &Plugin, pi_context Ctx,
                           pi_mem_flags Flags, size_t Size, void *HostPtr,
                           pi_mem *RetMem,
                           const pi_mem_properties *Props = nullptr);
void memReleaseHelper(const plugin &Plugin, pi_mem Mem);
void memBufferMapHelper(const plugin &Plugin, pi_queue command_queue,
                        pi_mem buffer, pi_bool blocking_map,
                        pi_map_flags map_flags, size_t offset, size_t size,
                        pi_uint32 num_events_in_wait_list,
                        const pi_event *event_wait_list, pi_event *event,
                        void **ret_map);
void memUnmapHelper(const plugin &Plugin, pi_queue command_queue, pi_mem memobj,
                    void *mapped_ptr, pi_uint32 num_events_in_wait_list,
                    const pi_event *event_wait_list, pi_event *event);
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
