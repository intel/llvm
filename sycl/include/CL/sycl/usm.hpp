//==---------------- usm.hpp - SYCL USM ------------------------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#include <cstddef>

#include <CL/sycl/usm_allocator.hpp>
#include <CL/sycl/detail/usm_impl.hpp>

#pragma once

namespace cl {
namespace sycl {

///
// Explicit USM
///
void *malloc_device(size_t size,
                    const device& dev,
                    const context& ctxt) {
  return detail::usm::alignedAlloc<alloc::device>(0,
                                                  size,
                                                  &ctxt,
                                                  &dev);
}

void *aligned_alloc_device(size_t alignment, size_t size,
                                const device& dev,
                                const context& ctx) {
  return detail::usm::alignedAlloc<alloc::device>(alignment,
                                                  size,
                                                  &ctxt,
                                                  &dev);
}

void free(void *ptr, const context& ctxt) {
  return detail::usm::free(ptr, &ctxt);
}

///
// Restricted USM
///
void *malloc_host(size_t size, const context& ctxt) {
  return detail::usm::alignedAlloc<alloc::host>(0,
                                                size,
                                                &ctxt,
                                                nullptr);
}

void *malloc_shared(size_t size,
                    const device& dev,
                    const context& ctxt) {
  return detail::usm::alignedAlloc<alloc::shared>(0,
                                                  size,
                                                  &ctxt,
                                                  &dev);
}
  
void *aligned_alloc_host(size_t alignment, size_t size,
                         const context& ctxt) {
  return detail::usm::alignedAlloc<alloc::host>(alignment,
                                                size,
                                                &ctxt,
                                                nullptr);
}

void *sycl_aligned_alloc(size_t alignment, size_t size,
                         const device& dev,
                         const context& ctxt) {
  return detail::usm::alignedAlloc<alloc::shared>(alignment,
                                                  size,
                                                  &ctxt,
                                                  &dev);
}

// single form
  
void *malloc(size_t size,
                  const device& dev,
                  const context& ctxt,
                  cl::sycl::alloc Kind) {
  switch (Kind) {
  case cl::sycl::alloc::host: {
    return sycl_malloc_host(size, ctxt);                             
  }
  case cl::sycl::alloc::device: {
    return sycl_malloc_device(size, dev, ctxt);
  }
  case cl::sycl::alloc::shared: {
    return sycl_malloc(size, dev, ctxt);
  }
  default: {
    return nullptr;
  }
  }
}

void *aligned_alloc(size_t alignment, size_t size,
                  const device& dev,
                  const context& ctxt,
                  cl::sycl::alloc Kind) {
  switch (Kind) {
  case cl::sycl::alloc::host: {
    return aligned_alloc_host(alignment, size, ctxt);                             
  }
  case cl::sycl::alloc::device: {
    return aligned_alloc_device(alignment, size, dev, ctxt);
  }
  case cl::sycl::alloc::shared: {
    return aligned_alloc_shared(alignment, size, dev, ctxt);
  }
  default: {
    return nullptr;
  }
  }
}
  
} // namespace sycl
} // namespace cl
