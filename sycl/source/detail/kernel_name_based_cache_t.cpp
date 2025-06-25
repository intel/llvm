//==-------------------- kernel_name_based_cache_t.cpp ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <detail/kernel_name_based_cache_t.hpp>
#include <detail/program_manager/program_manager.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {

KernelNameBasedCacheT::KernelNameBasedCacheT(KernelNameStrRefT KernelName) {
    init(KernelName);
}

void KernelNameBasedCacheT::init(KernelNameStrRefT KernelName) {
    auto &PM = detail::ProgramManager::getInstance();
    MUsesAssert = PM.kernelUsesAssert(KernelName);
    MImplicitLocalArgPos = PM.kernelImplicitLocalArgPos(KernelName);
#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
    MInitialized.store(true);
#endif
}

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
void KernelNameBasedCacheT::initIfNeeded(KernelNameStrRefT KernelName) {
  if (!MInitialized.load())
    init(KernelName);
}
#endif

FastKernelSubcacheT &KernelNameBasedCacheT::getKernelSubcache() {
  assertInitialized();
  return MFastKernelSubcache;
}
bool KernelNameBasedCacheT::usesAssert(){
  assertInitialized();
  return MUsesAssert;
}
const std::optional<int> &KernelNameBasedCacheT::getImplicitLocalArgPos() {
  assertInitialized();
  return MImplicitLocalArgPos;
}

void KernelNameBasedCacheT::assertInitialized() {
#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
  assert(MInitialized.load() && "Cache needs to be initialized before use");
#endif
 }

} // namespace detail
} // namespace _V1
} // namespace sycl