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
using ContextImplPtr = std::shared_ptr<sycl::detail::context_impl>;
namespace detail {
void waitEvents(std::vector<sycl::event> DepEvents) {
  for (auto SyclEvent : DepEvents) {
    detail::getSyclObjImpl(SyclEvent)->waitInternal();
  }
}

__SYCL_EXPORT void
markBufferAsInternal(const std::shared_ptr<buffer_impl> &BufImpl) {
  BufImpl->markAsInternal();
}

std::tuple<const RTDeviceBinaryImage *, ur_program_handle_t>
retrieveKernelBinary(const QueueImplPtr &Queue, const char *KernelName,
                     CGExecKernel *KernelCG) {
  bool isNvidia =
      Queue->getDeviceImplPtr()->getBackend() == backend::ext_oneapi_cuda;
  bool isHIP =
      Queue->getDeviceImplPtr()->getBackend() == backend::ext_oneapi_hip;
  if (isNvidia || isHIP) {
    auto KernelID = ProgramManager::getInstance().getSYCLKernelID(KernelName);
    std::vector<kernel_id> KernelIds{KernelID};
    auto DeviceImages =
        ProgramManager::getInstance().getRawDeviceImages(KernelIds);
    auto DeviceImage = std::find_if(
        DeviceImages.begin(), DeviceImages.end(),
        [isNvidia](RTDeviceBinaryImage *DI) {
          const std::string &TargetSpec = isNvidia ? std::string("llvm_nvptx64")
                                                   : std::string("llvm_amdgcn");
          return DI->getFormat() == SYCL_DEVICE_BINARY_TYPE_LLVMIR_BITCODE &&
                 DI->getRawData().DeviceTargetSpec == TargetSpec;
        });
    if (DeviceImage == DeviceImages.end()) {
      return {nullptr, nullptr};
    }
    auto ContextImpl = Queue->getContextImplPtr();
    auto Context = detail::createSyclObjFromImpl<context>(ContextImpl);
    auto DeviceImpl = Queue->getDeviceImplPtr();
    auto Device = detail::createSyclObjFromImpl<device>(DeviceImpl);
    ur_program_handle_t Program =
        detail::ProgramManager::getInstance().createURProgram(
            **DeviceImage, Context, {Device});
    return {*DeviceImage, Program};
  }

  const RTDeviceBinaryImage *DeviceImage = nullptr;
  ur_program_handle_t Program = nullptr;
  if (KernelCG->getKernelBundle() != nullptr) {
    // Retrieve the device image from the kernel bundle.
    auto KernelBundle = KernelCG->getKernelBundle();
    kernel_id KernelID =
        detail::ProgramManager::getInstance().getSYCLKernelID(KernelName);

    auto SyclKernel = detail::getSyclObjImpl(
        KernelBundle->get_kernel(KernelID, KernelBundle));

    DeviceImage = SyclKernel->getDeviceImage()->get_bin_image_ref();
    Program = SyclKernel->getDeviceImage()->get_ur_program_ref();
  } else if (KernelCG->MSyclKernel != nullptr) {
    DeviceImage = KernelCG->MSyclKernel->getDeviceImage()->get_bin_image_ref();
    Program = KernelCG->MSyclKernel->getDeviceImage()->get_ur_program_ref();
  } else {
    auto ContextImpl = Queue->getContextImplPtr();
    auto Context = detail::createSyclObjFromImpl<context>(ContextImpl);
    auto DeviceImpl = Queue->getDeviceImplPtr();
    auto Device = detail::createSyclObjFromImpl<device>(DeviceImpl);
    DeviceImage = &detail::ProgramManager::getInstance().getDeviceImage(
        KernelName, Context, Device);
    Program = detail::ProgramManager::getInstance().createURProgram(
        *DeviceImage, Context, {std::move(Device)});
  }
  return {DeviceImage, Program};
}

} // namespace detail
} // namespace _V1
} // namespace sycl
