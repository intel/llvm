//==---------------- pipes.hpp - SYCL pipes ------------*- C++ -*-----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#pragma once

#include <CL/sycl/INTEL/pipes.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace intel {
template <class name, class dataT, int32_t min_capacity = 0>
using pipe = INTEL::pipe<name, dataT, min_capacity>;

template <class name, class dataT, int32_t min_capacity = 0>
using kernel_readable_io_pipe =
    INTEL::kernel_readable_io_pipe<name, dataT, min_capacity>;

template <class name, class dataT, int32_t min_capacity = 0>
using kernel_writeable_io_pipe =
    INTEL::kernel_writeable_io_pipe<name, dataT, min_capacity>;
} // namespace intel
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
