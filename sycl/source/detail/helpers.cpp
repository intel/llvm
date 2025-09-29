//==---------------- helpers.cpp - SYCL helpers ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/helpers.hpp>

#include <detail/scheduler/commands.hpp>
#include <sycl/detail/helpers.hpp>

#include <detail/buffer_impl.hpp>
#include <detail/context_impl.hpp>
#include <detail/device_binary_image.hpp>
#include <detail/event_impl.hpp>
#include <detail/kernel_bundle_impl.hpp>
#include <detail/queue_impl.hpp>
#include <sycl/event.hpp>

#include <memory>
#include <tuple>

namespace sycl {
inline namespace _V1 {
namespace detail {
#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
// Unused, only keeping for ABI compatibility reasons.
__SYCL_EXPORT void waitEvents(std::vector<sycl::event> DepEvents) {
  for (auto SyclEvent : DepEvents) {
    detail::getSyclObjImpl(SyclEvent)->waitInternal();
  }
}
#endif

const RTDeviceBinaryImage *retrieveKernelBinary(queue_impl &Queue,
                                                KernelNameStrRefT KernelName,
                                                CGExecKernel *KernelCG) {
  device_impl &Dev = Queue.getDeviceImpl();
  bool isNvidia = Dev.getBackend() == backend::ext_oneapi_cuda;
  bool isHIP = Dev.getBackend() == backend::ext_oneapi_hip;
  if (isNvidia || isHIP) {
    auto KernelID = ProgramManager::getInstance().getSYCLKernelID(KernelName);
    std::vector<kernel_id> KernelIds{std::move(KernelID)};
    auto DeviceImages =
        ProgramManager::getInstance().getRawDeviceImages(KernelIds);
    auto DeviceImage = std::find_if(
        DeviceImages.begin(), DeviceImages.end(),
        [isNvidia](const RTDeviceBinaryImage *DI) {
          const std::string &TargetSpec = isNvidia ? std::string("llvm_nvptx64")
                                                   : std::string("llvm_amdgcn");
          return DI->getFormat() == SYCL_DEVICE_BINARY_TYPE_LLVMIR_BITCODE &&
                 DI->getRawData().DeviceTargetSpec == TargetSpec;
        });
    assert(DeviceImage != DeviceImages.end() &&
           "Failed to obtain a binary image.");
    return *DeviceImage;
  }

  if (KernelCG->MSyclKernel != nullptr)
    return KernelCG->MSyclKernel->getDeviceImage().get_bin_image_ref();

  if (auto KernelBundleImpl = KernelCG->getKernelBundle())
    if (auto SyclKernelImpl = KernelBundleImpl->tryGetKernel(KernelName))
      // Retrieve the device image from the kernel bundle.
      return SyclKernelImpl->getDeviceImage().get_bin_image_ref();

  context_impl &ContextImpl = Queue.getContextImpl();
  return &detail::ProgramManager::getInstance().getDeviceImage(
      KernelName, ContextImpl, Dev);
}

} // namespace detail
} // namespace _V1
} // namespace sycl
