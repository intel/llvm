//==---- usm_pointer_info.hpp - SYCL USM pointer info queries --*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //
#pragma once

#include <sycl/detail/export.hpp> // for __SYCL_EXPORT
#include <sycl/usm/usm_enums.hpp> // for alloc

namespace sycl {
inline namespace _V1 {

class device;
class context;

// Pointer queries
/// Query the allocation type from a USM pointer
///
/// \param ptr is the USM pointer to query
/// \param ctxt is the sycl context the ptr was allocated in
__SYCL_EXPORT usm::alloc get_pointer_type(const void *ptr, const context &ctxt);

/// Queries the device against which the pointer was allocated
/// Throws an exception with errc::invalid error code if ptr is a host
/// allocation.
///
/// \param ptr is the USM pointer to query
/// \param ctxt is the sycl context the ptr was allocated in
__SYCL_EXPORT device get_pointer_device(const void *ptr, const context &ctxt);

} // namespace _V1
} // namespace sycl
