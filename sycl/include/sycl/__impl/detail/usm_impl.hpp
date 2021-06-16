//==-------------- usm_impl.hpp - SYCL USM Utils ---------------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //
#pragma once

#include <sycl/__impl/detail/export.hpp>
#include <sycl/__impl/usm/usm_enums.hpp>

namespace __sycl_internal {
inline namespace __v1 {
namespace detail {
namespace usm {

__SYCL_EXPORT void *alignedAlloc(size_t Alignment, size_t Bytes,
                                 const context &Ctxt, const device &Dev,
                                 sycl::usm::alloc Kind);

__SYCL_EXPORT void *alignedAllocHost(size_t Alignment, size_t Bytes,
                                     const context &Ctxt,
                                     sycl::usm::alloc Kind);

__SYCL_EXPORT void free(void *Ptr, const context &Ctxt);

} // namespace usm
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
