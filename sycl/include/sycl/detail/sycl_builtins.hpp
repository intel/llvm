//==---------------- sycl_builtins.hpp --- SYCL access ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines_elementary.hpp> // for __DPCPP_SYCL_EXTERNAL

#ifdef __SYCL_DEVICE_ONLY__
// Request a fixed-size allocation in local address space at kernel scope.
// Required for group_local_memory and work_group_static.
extern "C" __DPCPP_SYCL_EXTERNAL __attribute__((opencl_local)) std::uint8_t *
__sycl_allocateLocalMemory(std::size_t Size, std::size_t Alignment);
// Request a placeholder for a dynamically-sized buffer in local address space
// at kernel scope. Required for work_group_static.
extern "C" __DPCPP_SYCL_EXTERNAL __attribute__((opencl_local)) std::uint8_t *
__sycl_dynamicLocalMemoryPlaceholder();
#endif
