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
#include <CL/sycl/export.hpp>
#include <CL/sycl/usm/usm_enums.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
namespace usm {

SYCL_API void *alignedAlloc(size_t Alignment, size_t Bytes, const context &Ctxt,
                            const device &Dev, cl::sycl::usm::alloc Kind);

SYCL_API void *alignedAllocHost(size_t Alignment, size_t Bytes,
                                const context &Ctxt, cl::sycl::usm::alloc Kind);

SYCL_API void free(void *Ptr, const context &Ctxt);

} // namespace usm
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
