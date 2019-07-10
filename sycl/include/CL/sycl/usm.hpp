//==---------------- usm.hpp - SYCL USM ------------------------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //
#pragma once

#include <CL/sycl/detail/usm_impl.hpp>
#include <CL/sycl/usm/usm_allocator.hpp>
#include <CL/sycl/usm/usm_enums.hpp>

#include <cstddef>

namespace cl {
namespace sycl {
///
// Explicit USM
///
void *malloc_device(size_t size, const device &dev, const context &ctxt);

void *aligned_alloc_device(size_t alignment, size_t size, const device &dev,
                           const context &ctxt);

void free(void *ptr, const context &ctxt);

///
// Restricted USM
///
void *malloc_host(size_t size, const context &ctxt);

void *malloc_shared(size_t size, const device &dev, const context &ctxt);

void *aligned_alloc_host(size_t alignment, size_t size, const context &ctxt);

void *aligned_alloc_shared(size_t alignment, size_t size, const device &dev,
                           const context &ctxt);

// single form

void *malloc(size_t size, const device &dev, const context &ctxt,
             usm::alloc kind);

void *aligned_alloc(size_t alignment, size_t size, const device &dev,
                    const context &ctxt, usm::alloc kind);

} // namespace sycl
} // namespace cl
