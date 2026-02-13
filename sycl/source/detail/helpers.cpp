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

const RTDeviceBinaryImage *retrieveKernelBinary(queue_impl &Queue,
                                                std::string_view KernelName,
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

// This function is implemented (duplicating getUrEvents a lot) as short term
// solution for the issue that barrier with wait list could not
// handle empty ur event handles when kernel is enqueued on host task
// completion.
std::vector<ur_event_handle_t>
getUrEventsBlocking(std::vector<EventImplPtr> &Events, bool HasEventMode,
                    queue_impl &queue, bool isHostTask) {
  std::vector<ur_event_handle_t> RetUrEvents;
  for (auto &Event : Events) {
    // Throwaway events created with empty constructor will not have a context
    // (which is set lazily) calling getContextImpl() would set that
    // context, which we wish to avoid as it is expensive.
    // Skip host task and NOP events also.
    if (Event->isDefaultConstructed() || Event->isHost() || Event->isNOP())
      continue;

    // If command has not been enqueued then we have to enqueue it.
    // It may happen if async enqueue in a host task is involved.
    // Interoperability events are special cases and they are not enqueued, as
    // they don't have an associated queue and command.
    if (!Event->isInterop() && !Event->isEnqueued()) {
      if (!Event->getCommand() || !Event->getCommand()->producesPiEvent())
        continue;
      std::vector<Command *> AuxCmds;
      Scheduler::getInstance().enqueueCommandForCG(*Event, AuxCmds, BLOCKING);
    }
    // Do not add redundant event dependencies for in-order queues.
    // At this stage dependency is definitely ur task and need to check if
    // current one is a host task. In this case we should not skip pi event due
    // to different sync mechanisms for different task types on in-order queue.
    // If the resulting event is supposed to have a specific event mode,
    // redundant events may still differ from the resulting event, so they are
    // kept.
    if (!HasEventMode && Event->getWorkerQueue().get() == &queue &&
        queue.isInOrder() && !isHostTask)
      continue;

    RetUrEvents.push_back(Event->getHandle());
  }

  return RetUrEvents;
}

} // namespace detail
} // namespace _V1
} // namespace sycl
