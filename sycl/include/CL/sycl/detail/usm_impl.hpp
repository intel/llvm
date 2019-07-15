//==-------------- usm_impl.hpp - SYCL USM Utils ---------------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //
#pragma once

#include <CL/cl.h>
#include <CL/cl_usm_ext.h>
#include <CL/sycl/usm/usm_enums.hpp>

namespace cl {
namespace sycl {
namespace detail {
namespace usm {

void *alignedAlloc(size_t alignment, size_t bytes, const context *ctxt,
                   const device *dev, cl::sycl::usm::alloc kind);

void free(void *ptr, const context *ctxt);

} // namespace usm
} // namespace detail
} // namespace sycl
} // namespace cl
