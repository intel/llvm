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
bool has_extension(const sycl::platform &syclPlatform,
                   const std::string &extension) {
  if (syclPlatform.get_backend() != sycl::backend::opencl) {
    throw sycl::runtime_error(errc::backend_mismatch, "Backends mismatch",
                              PI_ERROR_INVALID_OPERATION);
  }

  std::vector<device> devices = syclPlatform.get_devices();
  for (device &currentDevice : devices) {
    std::vector<std::string> deviceExtensions =
        currentDevice.get_info<info::device::extensions>();

    auto findResult =
        std::find(deviceExtensions.begin(), deviceExtensions.end(), extension);
    if (findResult != deviceExtensions.end()) {
      return true;
    }
  }
  return false;
}

bool has_extension(const sycl::device &syclDevice,
                   const std::string &extension) {
  if (syclDevice.get_backend() != sycl::backend::opencl) {
    throw sycl::runtime_error(errc::backend_mismatch, "Backends mismatch",
                              PI_ERROR_INVALID_OPERATION);
  }

  std::vector<std::string> deviceExtensions =
      syclDevice.get_info<info::device::extensions>();

  auto findResult =
      std::find(deviceExtensions.begin(), deviceExtensions.end(), extension);
  return findResult != deviceExtensions.end();
}
} // namespace opencl
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
