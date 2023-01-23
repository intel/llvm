/*
 *
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 * @file ur_ldrddi.hpp
 *
 */
#ifndef UR_LOADER_LDRDDI_H
#define UR_LOADER_LDRDDI_H 1

namespace loader
{
    ///////////////////////////////////////////////////////////////////////////////
    using ur_platform_object_t                = object_t < ur_platform_handle_t >;
    using ur_platform_factory_t               = singleton_factory_t < ur_platform_object_t, ur_platform_handle_t >;

    using ur_device_object_t                  = object_t < ur_device_handle_t >;
    using ur_device_factory_t                 = singleton_factory_t < ur_device_object_t, ur_device_handle_t >;

    using ur_context_object_t                 = object_t < ur_context_handle_t >;
    using ur_context_factory_t                = singleton_factory_t < ur_context_object_t, ur_context_handle_t >;

    using ur_event_object_t                   = object_t < ur_event_handle_t >;
    using ur_event_factory_t                  = singleton_factory_t < ur_event_object_t, ur_event_handle_t >;

    using ur_program_object_t                 = object_t < ur_program_handle_t >;
    using ur_program_factory_t                = singleton_factory_t < ur_program_object_t, ur_program_handle_t >;

    using ur_module_object_t                  = object_t < ur_module_handle_t >;
    using ur_module_factory_t                 = singleton_factory_t < ur_module_object_t, ur_module_handle_t >;

    using ur_kernel_object_t                  = object_t < ur_kernel_handle_t >;
    using ur_kernel_factory_t                 = singleton_factory_t < ur_kernel_object_t, ur_kernel_handle_t >;

    using ur_queue_object_t                   = object_t < ur_queue_handle_t >;
    using ur_queue_factory_t                  = singleton_factory_t < ur_queue_object_t, ur_queue_handle_t >;

    using ur_native_object_t                  = object_t < ur_native_handle_t >;
    using ur_native_factory_t                 = singleton_factory_t < ur_native_object_t, ur_native_handle_t >;

    using ur_sampler_object_t                 = object_t < ur_sampler_handle_t >;
    using ur_sampler_factory_t                = singleton_factory_t < ur_sampler_object_t, ur_sampler_handle_t >;

    using ur_mem_object_t                     = object_t < ur_mem_handle_t >;
    using ur_mem_factory_t                    = singleton_factory_t < ur_mem_object_t, ur_mem_handle_t >;

}

#endif /* UR_LOADER_LDRDDI_H */
