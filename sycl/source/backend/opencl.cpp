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

#include <memory>
#include <string_view>

namespace sycl {
inline namespace _V1 {
namespace opencl {
using namespace detail;

//----------------------------------------------------------------------------
// Implementation of opencl::make<platform>
__SYCL_EXPORT platform make_platform(ur_native_handle_t NativeHandle) {
  return detail::make_platform(NativeHandle, backend::opencl);
}

//----------------------------------------------------------------------------
// Implementation of opencl::make<device>
__SYCL_EXPORT device make_device(ur_native_handle_t NativeHandle) {
  return detail::make_device(NativeHandle, backend::opencl);
}

//----------------------------------------------------------------------------
// Implementation of opencl::make<context>
__SYCL_EXPORT context make_context(ur_native_handle_t NativeHandle) {
  return detail::make_context(NativeHandle, detail::defaultAsyncHandler,
                              backend::opencl);
}

//----------------------------------------------------------------------------
// Implementation of opencl::make<queue>
__SYCL_EXPORT queue make_queue(const context &Context,
                               ur_native_handle_t NativeHandle) {
  const auto &ContextImpl = getSyclObjImpl(Context);
  return detail::make_queue(NativeHandle, 0, Context, nullptr, false, {},
                            ContextImpl->get_async_handler(), backend::opencl);
}

//----------------------------------------------------------------------------
// Free functions to query OpenCL backend extensions
__SYCL_EXPORT bool has_extension(const sycl::platform &SyclPlatform,
                                 const std::string &Extension) {
  if (SyclPlatform.get_backend() != sycl::backend::opencl) {
    throw sycl::exception(
        errc::backend_mismatch,
        "has_extension can only be used with an OpenCL backend");
  }

  std::shared_ptr<sycl::detail::platform_impl> PlatformImpl =
      getSyclObjImpl(SyclPlatform);
  ur_platform_handle_t PluginPlatform = PlatformImpl->getUrHandleRef();
  const UrPluginPtr &Plugin = PlatformImpl->getUrPlugin();

  // Manual invocation of plugin API to avoid using deprecated
  // info::platform::extensions call.
  size_t ResultSize = 0;
  Plugin->call(urPlatformGetInfo, PluginPlatform, UR_PLATFORM_INFO_EXTENSIONS,
               /*propSize=*/0,
               /*pPropValue=*/nullptr, &ResultSize);
  if (ResultSize == 0)
    return false;

  std::unique_ptr<char[]> Result(new char[ResultSize]);
  Plugin->call(urPlatformGetInfo, PluginPlatform, UR_PLATFORM_INFO_EXTENSIONS,
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

  std::shared_ptr<sycl::detail::device_impl> DeviceImpl =
      getSyclObjImpl(SyclDevice);
  ur_device_handle_t PluginDevice = DeviceImpl->getUrHandleRef();
  const UrPluginPtr &Plugin = DeviceImpl->getUrPlugin();

  // Manual invocation of plugin API to avoid using deprecated
  // info::device::extensions call.
  size_t ResultSize = 0;
  Plugin->call(urDeviceGetInfo, PluginDevice, UR_DEVICE_INFO_EXTENSIONS,
               /*propSize=*/0,
               /*pPropValue=*/nullptr, &ResultSize);
  if (ResultSize == 0)
    return false;

  std::unique_ptr<char[]> Result(new char[ResultSize]);
  Plugin->call(urDeviceGetInfo, PluginDevice, UR_DEVICE_INFO_EXTENSIONS,
               ResultSize, Result.get(), nullptr);

  std::string_view ExtensionsString(Result.get());
  return ExtensionsString.find(Extension) != std::string::npos;
}
} // namespace opencl
} // namespace _V1
} // namespace sycl
