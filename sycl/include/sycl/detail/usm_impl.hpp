//==-------------- usm_impl.hpp - SYCL USM Utils ---------------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //
#pragma once

#include <sycl/detail/common.hpp>
#include <sycl/detail/export.hpp>
#include <sycl/usm/usm_enums.hpp>

namespace sycl {
inline namespace _V1 {
class device;

namespace detail::usm {

__SYCL_EXPORT void *alignedAlloc(size_t Alignment, size_t Bytes,
                                 const context &Ctxt, const device &Dev,
                                 sycl::usm::alloc Kind,
                                 const code_location &CL);

__SYCL_EXPORT void *alignedAllocHost(size_t Alignment, size_t Bytes,
                                     const context &Ctxt, sycl::usm::alloc Kind,
                                     const code_location &CL);

__SYCL_EXPORT void free(void *Ptr, const context &Ctxt,
                        const code_location &CL);

} // namespace detail::usm
} // namespace _V1
} // namespace sycl
