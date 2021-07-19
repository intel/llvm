//==------- opencl.cpp - SYCL OpenCL backend -------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <detail/kernel_impl.hpp>
#include <detail/platform_impl.hpp>
#include <detail/plugin.hpp>
#include <detail/program_impl.hpp>
#include <detail/queue_impl.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
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
// Implementation of opencl::make<program>
__SYCL_EXPORT program make_program(const context &Context,
                                   pi_native_handle NativeHandle) {
  // Construct the SYCL program from native program.
  // TODO: move here the code that creates PI program, and remove the
  // native interop constructor.
  return detail::createSyclObjFromImpl<program>(
      std::make_shared<program_impl>(getSyclObjImpl(Context), NativeHandle));
}

//----------------------------------------------------------------------------
// Implementation of opencl::make<queue>
__SYCL_EXPORT queue make_queue(const context &Context,
                               pi_native_handle NativeHandle) {
  const auto &ContextImpl = getSyclObjImpl(Context);
  return detail::make_queue(NativeHandle, Context, false,
                            ContextImpl->get_async_handler(), backend::opencl);
}
} // namespace opencl
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
