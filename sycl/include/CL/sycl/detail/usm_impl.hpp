//==-------------- usm_impl.hpp - SYCL USM Utils ---------------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //
#pragma once

#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/export.hpp>
#include <CL/sycl/usm/usm_enums.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
namespace usm {

__SYCL_EXPORT void *alignedAlloc(size_t Alignment, size_t Bytes,
                                 const context &Ctxt, const device &Dev,
                                 cl::sycl::usm::alloc Kind,
                                 const code_location &CL);

__SYCL_EXPORT void *alignedAllocHost(size_t Alignment, size_t Bytes,
                                     const context &Ctxt,
                                     cl::sycl::usm::alloc Kind,
                                     const code_location &CL);

__SYCL_EXPORT void free(void *Ptr, const context &Ctxt,
                        const code_location &CL);

} // namespace usm
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
