//==--- kernel_program_cache.cpp - Cache for kernel and program -*- C++-*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/adapter.hpp>
#include <detail/context_impl.hpp>
#include <detail/kernel_program_cache.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {
const AdapterPtr &KernelProgramCache::getAdapter() {
  return MParentContext->getAdapter();
}
} // namespace detail
} // namespace _V1
} // namespace sycl
