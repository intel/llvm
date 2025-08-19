//==---------------------- kernel_name_based_data.cpp ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <detail/kernel_name_based_data.hpp>
#include <detail/program_manager/program_manager.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {

KernelNameBasedData::KernelNameBasedData(KernelNameStrRefT KernelName) {
  init(KernelName);
}

void KernelNameBasedData::init(KernelNameStrRefT KernelName) {
  auto &PM = detail::ProgramManager::getInstance();
  MUsesAssert = PM.kernelUsesAssert(KernelName);
  MImplicitLocalArgPos = PM.kernelImplicitLocalArgPos(KernelName);
#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
  MInitialized.store(true);
#endif
}

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
void KernelNameBasedData::initIfNeeded(KernelNameStrRefT KernelName) {
  if (!MInitialized.load())
    init(KernelName);
}
#endif

FastKernelSubcacheT &KernelNameBasedData::getKernelSubcache() {
  assertInitialized();
  return MFastKernelSubcache;
}
bool KernelNameBasedData::usesAssert() {
  assertInitialized();
  return MUsesAssert;
}
const std::optional<int> &KernelNameBasedData::getImplicitLocalArgPos() {
  assertInitialized();
  return MImplicitLocalArgPos;
}

void KernelNameBasedData::assertInitialized() {
#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
  assert(MInitialized.load() && "Data needs to be initialized before use");
#endif
}

} // namespace detail
} // namespace _V1
} // namespace sycl