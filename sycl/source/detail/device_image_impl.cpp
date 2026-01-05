//==----------------- device_image_impl.cpp - SYCL device_image_impl -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/device_image_impl.hpp>
#include <detail/kernel_arg_mask.hpp>
#include <detail/kernel_bundle_impl.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {

std::shared_ptr<kernel_impl> device_image_impl::tryGetExtensionKernel(
    std::string_view Name, const context &Context,
    const kernel_bundle_impl &OwnerBundle) {
  if (!(getOriginMask() & ImageOriginKernelCompiler) &&
      !((getOriginMask() & ImageOriginSYCLBIN) && hasKernelName(Name)))
    return nullptr;

  std::string_view AdjustedName = getAdjustedKernelNameStrView(Name);
  if (MRTCBinInfo && MRTCBinInfo->MLanguage == syclex::source_language::sycl) {
    auto &PM = ProgramManager::getInstance();
    for (const std::string &Prefix : MRTCBinInfo->MPrefixes) {
      auto KID = PM.tryGetSYCLKernelID(Prefix + std::string(AdjustedName));

      if (!KID || !has_kernel(*KID))
        continue;

      auto UrProgram = get_ur_program();
      auto [UrKernel, CacheMutex, ArgMask] =
          PM.getOrCreateKernel(Context, AdjustedName,
                               /*PropList=*/{}, UrProgram);
      return std::make_shared<kernel_impl>(
          std::move(UrKernel), *getSyclObjImpl(Context), shared_from_this(),
          OwnerBundle, ArgMask, UrProgram, CacheMutex);
    }
    return nullptr;
  }

  ur_program_handle_t UrProgram = get_ur_program();
  detail::adapter_impl &Adapter = getSyclObjImpl(Context)->getAdapter();
  Managed<ur_kernel_handle_t> UrKernel{Adapter};
  Adapter.call<UrApiKind::urKernelCreate>(UrProgram, AdjustedName.data(),
                                          &UrKernel);

  const KernelArgMask *ArgMask = nullptr;
  if (auto ArgMaskIt = MEliminatedKernelArgMasks.find(AdjustedName);
      ArgMaskIt != MEliminatedKernelArgMasks.end())
    ArgMask = &ArgMaskIt->second;

  return std::make_shared<kernel_impl>(
      std::move(UrKernel), *detail::getSyclObjImpl(Context), shared_from_this(),
      OwnerBundle, ArgMask, UrProgram, /*CacheMutex=*/nullptr);
}

} // namespace detail
} // namespace _V1
} // namespace sycl
