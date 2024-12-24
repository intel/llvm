/*
 *
 * Copyright (C) 2022-2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ur_ldrddi.hpp
 *
 */
#ifndef UR_LOADER_LDRDDI_H
#define UR_LOADER_LDRDDI_H 1

#include "ur_object.hpp"
#include "ur_singleton.hpp"

namespace ur_loader {
///////////////////////////////////////////////////////////////////////////////

using ur_adapter_object_t = object_t<ur_adapter_handle_t>;
using ur_adapter_factory_t =
    singleton_factory_t<ur_adapter_object_t, ur_adapter_handle_t>;

using ur_platform_object_t = object_t<ur_platform_handle_t>;
using ur_platform_factory_t =
    singleton_factory_t<ur_platform_object_t, ur_platform_handle_t>;

using ur_device_object_t = object_t<ur_device_handle_t>;
using ur_device_factory_t =
    singleton_factory_t<ur_device_object_t, ur_device_handle_t>;

using ur_context_object_t = object_t<ur_context_handle_t>;
using ur_context_factory_t =
    singleton_factory_t<ur_context_object_t, ur_context_handle_t>;

using ur_event_object_t = object_t<ur_event_handle_t>;
using ur_event_factory_t =
    singleton_factory_t<ur_event_object_t, ur_event_handle_t>;

using ur_program_object_t = object_t<ur_program_handle_t>;
using ur_program_factory_t =
    singleton_factory_t<ur_program_object_t, ur_program_handle_t>;

using ur_kernel_object_t = object_t<ur_kernel_handle_t>;
using ur_kernel_factory_t =
    singleton_factory_t<ur_kernel_object_t, ur_kernel_handle_t>;

using ur_queue_object_t = object_t<ur_queue_handle_t>;
using ur_queue_factory_t =
    singleton_factory_t<ur_queue_object_t, ur_queue_handle_t>;

using ur_sampler_object_t = object_t<ur_sampler_handle_t>;
using ur_sampler_factory_t =
    singleton_factory_t<ur_sampler_object_t, ur_sampler_handle_t>;

using ur_mem_object_t = object_t<ur_mem_handle_t>;
using ur_mem_factory_t = singleton_factory_t<ur_mem_object_t, ur_mem_handle_t>;

using ur_physical_mem_object_t = object_t<ur_physical_mem_handle_t>;
using ur_physical_mem_factory_t =
    singleton_factory_t<ur_physical_mem_object_t, ur_physical_mem_handle_t>;

using ur_usm_pool_object_t = object_t<ur_usm_pool_handle_t>;
using ur_usm_pool_factory_t =
    singleton_factory_t<ur_usm_pool_object_t, ur_usm_pool_handle_t>;

using ur_exp_external_mem_object_t = object_t<ur_exp_external_mem_handle_t>;
using ur_exp_external_mem_factory_t =
    singleton_factory_t<ur_exp_external_mem_object_t,
                        ur_exp_external_mem_handle_t>;

using ur_exp_external_semaphore_object_t =
    object_t<ur_exp_external_semaphore_handle_t>;
using ur_exp_external_semaphore_factory_t =
    singleton_factory_t<ur_exp_external_semaphore_object_t,
                        ur_exp_external_semaphore_handle_t>;

using ur_exp_command_buffer_object_t = object_t<ur_exp_command_buffer_handle_t>;
using ur_exp_command_buffer_factory_t =
    singleton_factory_t<ur_exp_command_buffer_object_t,
                        ur_exp_command_buffer_handle_t>;

using ur_exp_command_buffer_command_object_t =
    object_t<ur_exp_command_buffer_command_handle_t>;
using ur_exp_command_buffer_command_factory_t =
    singleton_factory_t<ur_exp_command_buffer_command_object_t,
                        ur_exp_command_buffer_command_handle_t>;

using ur_exp_tensor_map_object_t = object_t<ur_exp_tensor_map_handle_t>;
using ur_exp_tensor_map_factory_t =
    singleton_factory_t<ur_exp_tensor_map_object_t, ur_exp_tensor_map_handle_t>;

struct handle_factories {
    ur_adapter_factory_t ur_adapter_factory;
    ur_platform_factory_t ur_platform_factory;
    ur_device_factory_t ur_device_factory;
    ur_context_factory_t ur_context_factory;
    ur_event_factory_t ur_event_factory;
    ur_program_factory_t ur_program_factory;
    ur_kernel_factory_t ur_kernel_factory;
    ur_queue_factory_t ur_queue_factory;
    ur_sampler_factory_t ur_sampler_factory;
    ur_mem_factory_t ur_mem_factory;
    ur_physical_mem_factory_t ur_physical_mem_factory;
    ur_usm_pool_factory_t ur_usm_pool_factory;
    ur_exp_external_mem_factory_t ur_exp_external_mem_factory;
    ur_exp_external_semaphore_factory_t ur_exp_external_semaphore_factory;
    ur_exp_command_buffer_factory_t ur_exp_command_buffer_factory;
    ur_exp_command_buffer_command_factory_t
        ur_exp_command_buffer_command_factory;
    ur_exp_tensor_map_factory_t ur_exp_tensor_map_factory;
};

} // namespace ur_loader

#endif /* UR_LOADER_LDRDDI_H */
