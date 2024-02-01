// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef UUR_RAII_H_INCLUDED
#define UUR_RAII_H_INCLUDED

#include "ur_api.h"
#include <cstddef>
#include <utility>

namespace uur {
namespace raii {
template <class URHandleT, ur_result_t (*retain)(URHandleT),
          ur_result_t (*release)(URHandleT)>
struct Wrapper {
    using handle_t = URHandleT;

    handle_t handle;

    Wrapper() : handle(nullptr) {}
    explicit Wrapper(handle_t handle) : handle(handle) {}
    ~Wrapper() {
        if (handle) {
            release(handle);
        }
    }

    Wrapper(const Wrapper &other) : handle(other.handle) { retain(handle); }
    Wrapper(Wrapper &&other) : handle(other.handle) { other.handle = nullptr; }
    Wrapper(std::nullptr_t) : handle(nullptr) {}

    Wrapper &operator=(const Wrapper &other) {
        if (handle) {
            release(handle);
        }
        handle = other.handle;
        retain(handle);
        return *this;
    }
    Wrapper &operator=(Wrapper &&other) {
        if (handle) {
            release(handle);
        }
        handle = other.handle;
        other.handle = nullptr;
        return *this;
    }
    Wrapper &operator=(std::nullptr_t) {
        if (handle) {
            release(handle);
        }
        new (this) Wrapper(nullptr);
        return *this;
    }

    handle_t *ptr() { return &handle; }
    handle_t get() { return handle; }
    handle_t operator->() { return handle; }
    operator handle_t() { return handle; }

    friend bool operator==(const Wrapper &lhs, const Wrapper &rhs) {
        return lhs.handle == rhs.handle;
    }
    friend bool operator==(const Wrapper &lhs, const handle_t &rhs) {
        return lhs.handle == rhs;
    }
    friend bool operator==(const handle_t &lhs, const Wrapper &rhs) {
        return lhs == rhs.handle;
    }
    friend bool operator==(const Wrapper &lhs, const std::nullptr_t &rhs) {
        return lhs.handle == rhs;
    }
    friend bool operator==(const std::nullptr_t &lhs, const Wrapper &rhs) {
        return lhs == rhs.handle;
    }

    friend bool operator!=(const Wrapper &lhs, const Wrapper &rhs) {
        return lhs.handle != rhs.handle;
    }
    friend bool operator!=(const Wrapper &lhs, const handle_t &rhs) {
        return lhs.handle != rhs;
    }
    friend bool operator!=(const handle_t &lhs, const Wrapper &rhs) {
        return lhs != rhs.handle;
    }
    friend bool operator!=(const Wrapper &lhs, const std::nullptr_t &rhs) {
        return lhs.handle != rhs;
    }
    friend bool operator!=(const std::nullptr_t &lhs, const Wrapper &rhs) {
        return lhs != rhs.handle;
    }
};

using LoaderConfig = Wrapper<ur_loader_config_handle_t, urLoaderConfigRetain,
                             urLoaderConfigRelease>;
using Adapter = Wrapper<ur_adapter_handle_t, urAdapterRetain, urAdapterRelease>;
using Device = Wrapper<ur_device_handle_t, urDeviceRetain, urDeviceRelease>;
using Context = Wrapper<ur_context_handle_t, urContextRetain, urContextRelease>;
using Mem = Wrapper<ur_mem_handle_t, urMemRetain, urMemRelease>;
using Sampler = Wrapper<ur_sampler_handle_t, urSamplerRetain, urSamplerRelease>;
using USMPool =
    Wrapper<ur_usm_pool_handle_t, urUSMPoolRetain, urUSMPoolRelease>;
using PhysicalMem = Wrapper<ur_physical_mem_handle_t, urPhysicalMemRetain,
                            urPhysicalMemRelease>;
using Program = Wrapper<ur_program_handle_t, urProgramRetain, urProgramRelease>;
using Kernel = Wrapper<ur_kernel_handle_t, urKernelRetain, urKernelRelease>;
using Queue = Wrapper<ur_queue_handle_t, urQueueRetain, urQueueRelease>;
using Event = Wrapper<ur_event_handle_t, urEventRetain, urEventRelease>;
}; // namespace raii
}; // namespace uur

#endif // UUR_RAII_H_INCLUDED
