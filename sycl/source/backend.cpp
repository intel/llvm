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
#include <sycl/backend.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/export.hpp>
#include <sycl/detail/pi.h>
#include <sycl/detail/pi.hpp>
#include <sycl/exception.hpp>
#include <sycl/exception_list.hpp>
#include <sycl/kernel_bundle.hpp>

#include <algorithm>
#include <memory>

namespace sycl {
inline namespace _V1 {
namespace detail {

static const PluginPtr &getPlugin(backend Backend) {
  switch (Backend) {
  case backend::opencl:
    return pi::getPlugin<backend::opencl>();
  case backend::ext_oneapi_level_zero:
    return pi::getPlugin<backend::ext_oneapi_level_zero>();
  case backend::ext_oneapi_cuda:
    return pi::getPlugin<backend::ext_oneapi_cuda>();
  case backend::ext_oneapi_hip:
    return pi::getPlugin<backend::ext_oneapi_hip>();
  default:
    throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                          "getPlugin: Unsupported backend " +
                              detail::codeToString(PI_ERROR_INVALID_OPERATION));
  }
}

backend convertBackend(pi_platform_backend PiBackend) {
  switch (PiBackend) {
  case PI_EXT_PLATFORM_BACKEND_UNKNOWN:
    return backend::all; // No specific backend
  case PI_EXT_PLATFORM_BACKEND_LEVEL_ZERO:
    return backend::ext_oneapi_level_zero;
  case PI_EXT_PLATFORM_BACKEND_OPENCL:
    return backend::opencl;
  case PI_EXT_PLATFORM_BACKEND_CUDA:
    return backend::ext_oneapi_cuda;
  case PI_EXT_PLATFORM_BACKEND_HIP:
    return backend::ext_oneapi_hip;
  case PI_EXT_PLATFORM_BACKEND_ESIMD:
    return backend::ext_intel_esimd_emulator;
  case PI_EXT_PLATFORM_BACKEND_NATIVE_CPU:
    return backend::ext_oneapi_native_cpu;
  }
  throw sycl::runtime_error{"convertBackend: Unsupported backend",
                            PI_ERROR_INVALID_OPERATION};
}

platform make_platform(pi_native_handle NativeHandle, backend Backend) {
  const auto &Plugin = getPlugin(Backend);

  // Create PI platform first.
  pi::PiPlatform PiPlatform = nullptr;
  Plugin->call<PiApiKind::piextPlatformCreateWithNativeHandle>(NativeHandle,
                                                               &PiPlatform);

  return detail::createSyclObjFromImpl<platform>(
      platform_impl::getOrMakePlatformImpl(PiPlatform, Plugin));
}

__SYCL_EXPORT device make_device(pi_native_handle NativeHandle,
                                 backend Backend) {
  const auto &Plugin = getPlugin(Backend);

  pi::PiDevice PiDevice = nullptr;
  Plugin->call<PiApiKind::piextDeviceCreateWithNativeHandle>(
      NativeHandle, nullptr, &PiDevice);
  // Construct the SYCL device from PI device.
  return detail::createSyclObjFromImpl<device>(
      std::make_shared<device_impl>(PiDevice, Plugin));
}

__SYCL_EXPORT context make_context(pi_native_handle NativeHandle,
                                   const async_handler &Handler,
                                   backend Backend) {
  const auto &Plugin = getPlugin(Backend);

  pi::PiContext PiContext = nullptr;
  Plugin->call<PiApiKind::piextContextCreateWithNativeHandle>(
      NativeHandle, 0, nullptr, false, &PiContext);
  // Construct the SYCL context from PI context.
  return detail::createSyclObjFromImpl<context>(
      std::make_shared<context_impl>(PiContext, Handler, Plugin));
}

__SYCL_EXPORT queue make_queue(pi_native_handle NativeHandle,
                               int32_t NativeHandleDesc, const context &Context,
                               const device *Device, bool KeepOwnership,
                               const property_list &PropList,
                               const async_handler &Handler, backend Backend) {
  sycl::detail::pi::PiDevice PiDevice =
      Device ? getSyclObjImpl(*Device)->getHandleRef() : nullptr;
  const auto &Plugin = getPlugin(Backend);
  const auto &ContextImpl = getSyclObjImpl(Context);

  // Create PI properties from SYCL properties.
  sycl::detail::pi::PiQueueProperties Properties[] = {
      PI_QUEUE_FLAGS,
      queue_impl::createPiQueueProperties(
          PropList, PropList.has_property<property::queue::in_order>()
                        ? QueueOrder::Ordered
                        : QueueOrder::OOO),
      0, 0, 0};
  if (PropList.has_property<ext::intel::property::queue::compute_index>()) {
    throw sycl::exception(
        make_error_code(errc::invalid),
        "Queue create using make_queue cannot have compute_index property.");
  }

  // Create PI queue first.
  pi::PiQueue PiQueue = nullptr;
  Plugin->call<PiApiKind::piextQueueCreateWithNativeHandle>(
      NativeHandle, NativeHandleDesc, ContextImpl->getHandleRef(), PiDevice,
      !KeepOwnership, Properties, &PiQueue);
  // Construct the SYCL queue from PI queue.
  return detail::createSyclObjFromImpl<queue>(
      std::make_shared<queue_impl>(PiQueue, ContextImpl, Handler, PropList));
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
  Plugin->call<PiApiKind::piextEventCreateWithNativeHandle>(
      NativeHandle, ContextImpl->getHandleRef(), !KeepOwnership, &PiEvent);

  event Event = detail::createSyclObjFromImpl<event>(
      std::make_shared<event_impl>(PiEvent, Context));

  if (Backend == backend::opencl)
    Plugin->call<PiApiKind::piEventRetain>(PiEvent);
  return Event;
}

std::shared_ptr<detail::kernel_bundle_impl>
make_kernel_bundle(pi_native_handle NativeHandle, const context &TargetContext,
                   bool KeepOwnership, bundle_state State, backend Backend) {
  const auto &Plugin = getPlugin(Backend);
  const auto &ContextImpl = getSyclObjImpl(TargetContext);

  pi::PiProgram PiProgram = nullptr;
  Plugin->call<PiApiKind::piextProgramCreateWithNativeHandle>(
      NativeHandle, ContextImpl->getHandleRef(), !KeepOwnership, &PiProgram);
  if (ContextImpl->getBackend() == backend::opencl)
    Plugin->call<PiApiKind::piProgramRetain>(PiProgram);

  std::vector<pi::PiDevice> ProgramDevices;
  uint32_t NumDevices = 0;

  Plugin->call<PiApiKind::piProgramGetInfo>(
      PiProgram, PI_PROGRAM_INFO_NUM_DEVICES, sizeof(NumDevices), &NumDevices,
      nullptr);
  ProgramDevices.resize(NumDevices);
  Plugin->call<PiApiKind::piProgramGetInfo>(PiProgram, PI_PROGRAM_INFO_DEVICES,
                                            sizeof(pi::PiDevice) * NumDevices,
                                            ProgramDevices.data(), nullptr);

  for (const auto &Dev : ProgramDevices) {
    size_t BinaryType = 0;
    Plugin->call<PiApiKind::piProgramGetBuildInfo>(
        PiProgram, Dev, PI_PROGRAM_BUILD_INFO_BINARY_TYPE, sizeof(size_t),
        &BinaryType, nullptr);
    switch (BinaryType) {
    case (PI_PROGRAM_BINARY_TYPE_NONE):
      if (State == bundle_state::object)
        Plugin->call<errc::build, PiApiKind::piProgramCompile>(
            PiProgram, 1, &Dev, nullptr, 0, nullptr, nullptr, nullptr, nullptr);
      else if (State == bundle_state::executable)
        Plugin->call<errc::build, PiApiKind::piProgramBuild>(
            PiProgram, 1, &Dev, nullptr, nullptr, nullptr);
      break;
    case (PI_PROGRAM_BINARY_TYPE_COMPILED_OBJECT):
    case (PI_PROGRAM_BINARY_TYPE_LIBRARY):
      if (State == bundle_state::input)
        throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                              "Program and kernel_bundle state mismatch " +
                                  detail::codeToString(PI_ERROR_INVALID_VALUE));
      if (State == bundle_state::executable)
        Plugin->call<errc::build, PiApiKind::piProgramLink>(
            ContextImpl->getHandleRef(), 1, &Dev, nullptr, 1, &PiProgram,
            nullptr, nullptr, &PiProgram);
      break;
    case (PI_PROGRAM_BINARY_TYPE_EXECUTABLE):
      if (State == bundle_state::input || State == bundle_state::object)
        throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                              "Program and kernel_bundle state mismatch " +
                                  detail::codeToString(PI_ERROR_INVALID_VALUE));
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
  auto KernelIDs = std::make_shared<std::vector<kernel_id>>();
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
      throw sycl::exception(
          sycl::make_error_code(sycl::errc::runtime),
          "make_kernel: kernel_bundle must have single program image " +
              detail::codeToString(PI_ERROR_INVALID_PROGRAM));

    const device_image<bundle_state::executable> &DeviceImage =
        *KernelBundle.begin();
    const auto &DeviceImageImpl = getSyclObjImpl(DeviceImage);
    PiProgram = DeviceImageImpl->get_program_ref();
  }

  // Create PI kernel first.
  pi::PiKernel PiKernel = nullptr;
  Plugin->call<PiApiKind::piextKernelCreateWithNativeHandle>(
      NativeHandle, ContextImpl->getHandleRef(), PiProgram, !KeepOwnership,
      &PiKernel);

  if (Backend == backend::opencl)
    Plugin->call<PiApiKind::piKernelRetain>(PiKernel);

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
} // namespace _V1
} // namespace sycl
