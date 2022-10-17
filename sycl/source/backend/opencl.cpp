//==------- opencl.cpp - SYCL OpenCL backend -------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/kernel_impl.hpp>
#include <detail/platform_impl.hpp>
#include <detail/plugin.hpp>
#include <detail/program_impl.hpp>
#include <detail/queue_impl.hpp>
#include <sycl/sycl.hpp>

#include <string_view>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace opencl {
using namespace detail;

//----------------------------------------------------------------------------
// Implementation of opencl::make<platform>
__SYCL_EXPORT platform make_platform(pi_native_handle NativeHandle) {
  return detail::make_platform(NativeHandle, backend::opencl);
}

//----------------------------------------------------------------------------
// Implementation of opencl::make<device>
__SYCL_EXPORT device make_device(pi_native_handle NativeHandle) {
  return detail::make_device(NativeHandle, backend::opencl);
}

//----------------------------------------------------------------------------
// Implementation of opencl::make<context>
__SYCL_EXPORT context make_context(pi_native_handle NativeHandle) {
  return detail::make_context(NativeHandle, async_handler{}, backend::opencl);
}

//----------------------------------------------------------------------------
// Implementation of opencl::make<queue>
__SYCL_EXPORT queue make_queue(const context &Context,
                               pi_native_handle NativeHandle) {
  const auto &ContextImpl = getSyclObjImpl(Context);
  return detail::make_queue(NativeHandle, Context, nullptr, false,
                            ContextImpl->get_async_handler(), backend::opencl);
}

// Free functions to query OpenCL backend extensions
// TODO: Extensions have been deprecated for aspects
__SYCL_EXPORT bool has_extension(const sycl::platform &SyclPlatform,
                                 const std::string &Extension) {
  if (SyclPlatform.get_backend() != sycl::backend::opencl) {
    throw sycl::exception(
        errc::backend_mismatch,
        "has_extension can only be used with an OpenCL backend");
  }

  auto PlatformImpl = getSyclObjImpl(SyclPlatform);
  RT::PiPlatform PluginPlatform = PlatformImpl->getHandleRef();
  plugin Plugin = PlatformImpl->getPlugin();

  // Manual invocation of plugin API to avoid using deprecated
  // info::platform::extensions call
  size_t ResultSize = 0;
  Plugin.call<PiApiKind::piPlatformGetInfo>(
      PluginPlatform, PI_PLATFORM_INFO_EXTENSIONS, 0, nullptr, &ResultSize);
  if (ResultSize == 0) {
    return false;
  }
  std::unique_ptr<char[]> Result(new char[ResultSize]);
  Plugin.call<PiApiKind::piPlatformGetInfo>(PluginPlatform,
                                            PI_PLATFORM_INFO_EXTENSIONS,
                                            ResultSize, Result.get(), nullptr);

  std::string_view ExtensionsString(Result.get());
  return ExtensionsString.find(Extension) != std::string::npos;
}

__SYCL_EXPORT bool has_extension(const sycl::device &SyclDevice,
                                 const std::string &Extension) {
  if (SyclDevice.get_backend() != sycl::backend::opencl) {
    throw sycl::exception(
        errc::backend_mismatch,
        "has_extension can only be used with an OpenCL backend");
  }

  auto DeviceImpl = getSyclObjImpl(SyclDevice);
  RT::PiDevice PluginDevice = DeviceImpl->getHandleRef();
  plugin Plugin = DeviceImpl->getPlugin();

  // Manual invocation of plugin API to avoid using deprecated
  // info::device::extensions call
  size_t ResultSize = 0;
  Plugin.call<PiApiKind::piDeviceGetInfo>(
      PluginDevice, PI_DEVICE_INFO_EXTENSIONS, 0, nullptr, &ResultSize);
  if (ResultSize == 0) {
    return false;
  }
  std::unique_ptr<char[]> Result(new char[ResultSize]);
  Plugin.call<PiApiKind::piDeviceGetInfo>(PluginDevice,
                                          PI_DEVICE_INFO_EXTENSIONS, ResultSize,
                                          Result.get(), nullptr);

  std::string_view ExtensionsString(Result.get());
  return ExtensionsString.find(Extension) != std::string::npos;
}
} // namespace opencl
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
