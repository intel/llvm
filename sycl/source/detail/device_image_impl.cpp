//==----------------- device_image_impl.cpp - SYCL device_image_impl -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/device_image_impl.hpp>
#include <detail/kernel_bundle_impl.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {

std::shared_ptr<kernel_impl> device_image_impl::tryGetSourceBasedKernel(
    std::string_view Name, const context &Context,
    const kernel_bundle_impl &OwnerBundle,
    const std::shared_ptr<device_image_impl> &Self) const {
  if (!(getOriginMask() & ImageOriginKernelCompiler))
    return nullptr;

  assert(MRTCBinInfo);
  std::string AdjustedName = adjustKernelName(Name);
  if (MRTCBinInfo->MLanguage == syclex::source_language::sycl) {
    auto &PM = ProgramManager::getInstance();
    for (const std::string &Prefix : MRTCBinInfo->MPrefixes) {
      auto KID = PM.tryGetSYCLKernelID(Prefix + AdjustedName);

      if (!KID || !has_kernel(*KID))
        continue;

      auto UrProgram = get_ur_program_ref();
      auto [UrKernel, CacheMutex, ArgMask] =
          PM.getOrCreateKernel(Context, AdjustedName,
                               /*PropList=*/{}, UrProgram);
      return std::make_shared<kernel_impl>(UrKernel, *getSyclObjImpl(Context),
                                           Self, OwnerBundle.shared_from_this(),
                                           ArgMask, UrProgram, CacheMutex);
    }
    return nullptr;
  }

  ur_program_handle_t UrProgram = get_ur_program_ref();
  const AdapterPtr &Adapter = getSyclObjImpl(Context)->getAdapter();
  ur_kernel_handle_t UrKernel = nullptr;
  Adapter->call<UrApiKind::urKernelCreate>(UrProgram, AdjustedName.c_str(),
                                           &UrKernel);
  // Kernel created by urKernelCreate is implicitly retained.

  return std::make_shared<kernel_impl>(
      UrKernel, *detail::getSyclObjImpl(Context), Self,
      OwnerBundle.shared_from_this(), /*ArgMask=*/nullptr, UrProgram,
      /*CacheMutex=*/nullptr);
}

} // namespace detail
} // namespace _V1
} // namespace sycl
