//==------- opencl.cpp - SYCL OpenCL backend -------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/adapter.hpp>
#include <detail/kernel_impl.hpp>
#include <detail/platform_impl.hpp>
#include <detail/queue_impl.hpp>

#include <memory>
#include <string_view>

namespace sycl {
inline namespace _V1 {
namespace opencl {
using namespace detail;

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
  ur_platform_handle_t AdapterPlatform = PlatformImpl->getHandleRef();
  const AdapterPtr &Adapter = PlatformImpl->getAdapter();

  // Manual invocation of UR API to avoid using deprecated
  // info::platform::extensions call.
  size_t ResultSize = 0;
  Adapter->call<UrApiKind::urPlatformGetInfo>(
      AdapterPlatform, UR_PLATFORM_INFO_EXTENSIONS,
      /*propSize=*/0,
      /*pPropValue=*/nullptr, &ResultSize);
  if (ResultSize == 0)
    return false;

  std::unique_ptr<char[]> Result(new char[ResultSize]);
  Adapter->call<UrApiKind::urPlatformGetInfo>(
      AdapterPlatform, UR_PLATFORM_INFO_EXTENSIONS, ResultSize, Result.get(),
      nullptr);

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
  ur_device_handle_t AdapterDevice = DeviceImpl->getHandleRef();
  const AdapterPtr &Adapter = DeviceImpl->getAdapter();

  // Manual invocation of UR API to avoid using deprecated
  // info::device::extensions call.
  size_t ResultSize = 0;
  Adapter->call<UrApiKind::urDeviceGetInfo>(
      AdapterDevice, UR_DEVICE_INFO_EXTENSIONS,
      /*propSize=*/0,
      /*pPropValue=*/nullptr, &ResultSize);
  if (ResultSize == 0)
    return false;

  std::unique_ptr<char[]> Result(new char[ResultSize]);
  Adapter->call<UrApiKind::urDeviceGetInfo>(AdapterDevice,
                                            UR_DEVICE_INFO_EXTENSIONS,
                                            ResultSize, Result.get(), nullptr);

  std::string_view ExtensionsString(Result.get());
  return ExtensionsString.find(Extension) != std::string::npos;
}
} // namespace opencl
} // namespace _V1
} // namespace sycl
