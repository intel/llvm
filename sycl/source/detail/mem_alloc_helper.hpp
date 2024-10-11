//==-------- mem_alloc_helper.hpp - SYCL mem alloc helper ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <ur_api.h>

namespace sycl {
inline namespace _V1 {
namespace detail {
void memBufferCreateHelper(const AdapterPtr &Adapter, ur_context_handle_t Ctx,
                           ur_mem_flags_t Flags, size_t Size,
                           ur_mem_handle_t *RetMem,
                           const ur_buffer_properties_t *Props = nullptr);
void memReleaseHelper(const AdapterPtr &Adapter, ur_mem_handle_t Mem);
void memBufferMapHelper(const AdapterPtr &Adapter,
                        ur_queue_handle_t command_queue, ur_mem_handle_t buffer,
                        bool blocking_map, ur_map_flags_t map_flags,
                        size_t offset, size_t size,
                        uint32_t num_events_in_wait_list,
                        const ur_event_handle_t *event_wait_list,
                        ur_event_handle_t *event, void **ret_map);
void memUnmapHelper(const AdapterPtr &Adapter, ur_queue_handle_t command_queue,
                    ur_mem_handle_t memobj, void *mapped_ptr,
                    uint32_t num_events_in_wait_list,
                    const ur_event_handle_t *event_wait_list,
                    ur_event_handle_t *event);
} // namespace detail
} // namespace _V1
} // namespace sycl
