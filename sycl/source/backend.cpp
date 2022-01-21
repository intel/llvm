//==------------------- backend.cpp ----------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "detail/context_impl.hpp"
#include "detail/event_impl.hpp"
#include "detail/kernel_bundle_impl.hpp"
#include "detail/kernel_id_impl.hpp"
#include "detail/platform_impl.hpp"
#include "detail/plugin.hpp"
#include "detail/queue_impl.hpp"
#include <CL/sycl/backend.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/export.hpp>
#include <CL/sycl/detail/pi.h>
#include <CL/sycl/detail/pi.hpp>
#include <CL/sycl/exception.hpp>
#include <CL/sycl/exception_list.hpp>
#include <CL/sycl/kernel_bundle.hpp>

#include <algorithm>
#include <memory>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

static const plugin &getPlugin(backend Backend) {
  switch (Backend) {
  case backend::opencl:
    return pi::getPlugin<backend::opencl>();
  case backend::ext_oneapi_level_zero:
    return pi::getPlugin<backend::ext_oneapi_level_zero>();
  default:
    throw sycl::runtime_error{"Unsupported backend", PI_INVALID_OPERATION};
  }
}

platform make_platform(pi_native_handle NativeHandle, backend Backend) {
  const auto &Plugin = getPlugin(Backend);

  // Create PI platform first.
  pi::PiPlatform PiPlatform = nullptr;
  Plugin.call<PiApiKind::piextPlatformCreateWithNativeHandle>(NativeHandle,
                                                              &PiPlatform);

  return detail::createSyclObjFromImpl<platform>(
      platform_impl::getOrMakePlatformImpl(PiPlatform, Plugin));
}

__SYCL_EXPORT device make_device(pi_native_handle NativeHandle,
                                 backend Backend) {
  const auto &Plugin = getPlugin(Backend);

  pi::PiDevice PiDevice = nullptr;
  Plugin.call<PiApiKind::piextDeviceCreateWithNativeHandle>(NativeHandle,
                                                            nullptr, &PiDevice);
  // Construct the SYCL device from PI device.
  return detail::createSyclObjFromImpl<device>(
      std::make_shared<device_impl>(PiDevice, Plugin));
}

__SYCL_EXPORT context make_context(pi_native_handle NativeHandle,
                                   const async_handler &Handler,
                                   backend Backend) {
  const auto &Plugin = getPlugin(Backend);

  pi::PiContext PiContext = nullptr;
  Plugin.call<PiApiKind::piextContextCreateWithNativeHandle>(
      NativeHandle, 0, nullptr, false, &PiContext);
  // Construct the SYCL context from PI context.
  return detail::createSyclObjFromImpl<context>(
      std::make_shared<context_impl>(PiContext, Handler, Plugin));
}

__SYCL_EXPORT queue make_queue(pi_native_handle NativeHandle,
                               const context &Context,
                               const async_handler &Handler, backend Backend) {
  return make_queue(NativeHandle, Context, false, Handler, Backend);
}

__SYCL_EXPORT queue make_queue(pi_native_handle NativeHandle,
                               const context &Context, bool KeepOwnership,
                               const async_handler &Handler, backend Backend) {
  const auto &Plugin = getPlugin(Backend);
  const auto &ContextImpl = getSyclObjImpl(Context);
  // Create PI queue first.
  pi::PiQueue PiQueue = nullptr;
  Plugin.call<PiApiKind::piextQueueCreateWithNativeHandle>(
      NativeHandle, ContextImpl->getHandleRef(), &PiQueue, !KeepOwnership);
  // Construct the SYCL queue from PI queue.
  return detail::createSyclObjFromImpl<queue>(
      std::make_shared<queue_impl>(PiQueue, ContextImpl, Handler));
}

__SYCL_EXPORT event make_event(pi_native_handle NativeHandle,
                               const context &Context, backend Backend) {
  return make_event(NativeHandle, Context, false, Backend);
}

__SYCL_EXPORT event make_event(pi_native_handle NativeHandle,
                               const context &Context, bool KeepOwnership,
                               backend Backend) {
  const auto &Plugin = getPlugin(Backend);
  const auto &ContextImpl = getSyclObjImpl(Context);

  pi::PiEvent PiEvent = nullptr;
  Plugin.call<PiApiKind::piextEventCreateWithNativeHandle>(
      NativeHandle, ContextImpl->getHandleRef(), !KeepOwnership, &PiEvent);

  return detail::createSyclObjFromImpl<event>(
      std::make_shared<event_impl>(PiEvent, Context));
}

std::shared_ptr<detail::kernel_bundle_impl>
make_kernel_bundle(pi_native_handle NativeHandle, const context &TargetContext,
                   bool KeepOwnership, bundle_state State, backend Backend) {
  const auto &Plugin = getPlugin(Backend);
  const auto &ContextImpl = getSyclObjImpl(TargetContext);

  pi::PiProgram PiProgram = nullptr;
  Plugin.call<PiApiKind::piextProgramCreateWithNativeHandle>(
      NativeHandle, ContextImpl->getHandleRef(), KeepOwnership, &PiProgram);

  std::vector<pi::PiDevice> ProgramDevices;
  size_t NumDevices = 0;

  Plugin.call<PiApiKind::piProgramGetInfo>(
      PiProgram, PI_PROGRAM_INFO_NUM_DEVICES, sizeof(size_t), &NumDevices,
      nullptr);
  ProgramDevices.resize(NumDevices);
  Plugin.call<PiApiKind::piProgramGetInfo>(PiProgram, PI_PROGRAM_INFO_DEVICES,
                                           sizeof(pi::PiDevice) * NumDevices,
                                           ProgramDevices.data(), nullptr);

  for (const auto &Dev : ProgramDevices) {
    size_t BinaryType = 0;
    Plugin.call<PiApiKind::piProgramGetBuildInfo>(
        PiProgram, Dev, PI_PROGRAM_BUILD_INFO_BINARY_TYPE, sizeof(size_t),
        &BinaryType, nullptr);
    switch (BinaryType) {
    case (PI_PROGRAM_BINARY_TYPE_NONE):
      if (State == bundle_state::object)
        Plugin.call<PiApiKind::piProgramCompile>(
            PiProgram, 1, &Dev, nullptr, 0, nullptr, nullptr, nullptr, nullptr);
      else if (State == bundle_state::executable)
        Plugin.call<PiApiKind::piProgramBuild>(PiProgram, 1, &Dev, nullptr,
                                               nullptr, nullptr);
      break;
    case (PI_PROGRAM_BINARY_TYPE_COMPILED_OBJECT):
    case (PI_PROGRAM_BINARY_TYPE_LIBRARY):
      if (State == bundle_state::input)
        // TODO SYCL2020 exception
        throw sycl::runtime_error("Program and kernel_bundle state mismatch",
                                  PI_INVALID_VALUE);
      if (State == bundle_state::executable)
        Plugin.call<PiApiKind::piProgramLink>(ContextImpl->getHandleRef(), 1,
                                              &Dev, nullptr, 1, &PiProgram,
                                              nullptr, nullptr, &PiProgram);
      break;
    case (PI_PROGRAM_BINARY_TYPE_EXECUTABLE):
      if (State == bundle_state::input || State == bundle_state::object)
        // TODO SYCL2020 exception
        throw sycl::runtime_error("Program and kernel_bundle state mismatch",
                                  PI_INVALID_VALUE);
      break;
    }
  }

  std::vector<device> Devices;
  Devices.reserve(ProgramDevices.size());
  std::transform(
      ProgramDevices.begin(), ProgramDevices.end(), std::back_inserter(Devices),
      [&Plugin](const auto &Dev) {
        auto Platform =
            detail::platform_impl::getPlatformFromPiDevice(Dev, Plugin);
        auto DeviceImpl = Platform->getOrMakeDeviceImpl(Dev, Platform);
        return createSyclObjFromImpl<device>(DeviceImpl);
      });

  // Unlike SYCL, other backends, like OpenCL or Level Zero, may not support
  // getting kernel IDs before executable is built. The SYCL Runtime workarounds
  // this by pre-building the device image and extracting kernel info. We can't
  // do the same to user images, since they may contain references to undefined
  // symbols (e.g. when kernel_bundle is supposed to be joined with another).
  std::vector<kernel_id> KernelIDs{};
  auto DevImgImpl = std::make_shared<device_image_impl>(
      nullptr, TargetContext, Devices, State, KernelIDs, PiProgram);
  device_image_plain DevImg{DevImgImpl};

  return std::make_shared<kernel_bundle_impl>(TargetContext, Devices, DevImg);
}

// TODO: Unused. Remove when allowed.
std::shared_ptr<detail::kernel_bundle_impl>
make_kernel_bundle(pi_native_handle NativeHandle, const context &TargetContext,
                   bundle_state State, backend Backend) {
  return make_kernel_bundle(NativeHandle, TargetContext, false, State, Backend);
}

kernel make_kernel(const context &TargetContext,
                   const kernel_bundle<bundle_state::executable> &KernelBundle,
                   pi_native_handle NativeHandle, bool KeepOwnership,
                   backend Backend) {
  const auto &Plugin = getPlugin(Backend);
  const auto &ContextImpl = getSyclObjImpl(TargetContext);
  const auto KernelBundleImpl = getSyclObjImpl(KernelBundle);

  // For Level-Zero expect exactly one device image in the bundle. This is
  // natural for interop kernel to get created out of a single native
  // program/module. This way we don't need to search the exact device image for
  // the kernel, which may not be trivial.
  //
  // Other backends don't need PI program.
  //
  pi::PiProgram PiProgram = nullptr;
  if (Backend == backend::ext_oneapi_level_zero) {
    if (KernelBundleImpl->size() != 1)
      throw sycl::runtime_error{
          "make_kernel: kernel_bundle must have single program image",
          PI_INVALID_PROGRAM};

    const device_image<bundle_state::executable> &DeviceImage =
        *KernelBundle.begin();
    const auto &DeviceImageImpl = getSyclObjImpl(DeviceImage);
    PiProgram = DeviceImageImpl->get_program_ref();
  }

  // Create PI kernel first.
  pi::PiKernel PiKernel = nullptr;
  Plugin.call<PiApiKind::piextKernelCreateWithNativeHandle>(
      NativeHandle, ContextImpl->getHandleRef(), PiProgram, KeepOwnership,
      &PiKernel);

  if (Backend == backend::opencl)
    Plugin.call<PiApiKind::piKernelRetain>(PiKernel);

  // Construct the SYCL queue from PI queue.
  return detail::createSyclObjFromImpl<kernel>(
      std::make_shared<kernel_impl>(PiKernel, ContextImpl, KernelBundleImpl));
}

kernel make_kernel(pi_native_handle NativeHandle, const context &TargetContext,
                   backend Backend) {
  return make_kernel(
      TargetContext,
      get_empty_interop_kernel_bundle<bundle_state::executable>(TargetContext),
      NativeHandle, false, Backend);
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
