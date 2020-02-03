//==---------------- context.cpp - SYCL context ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/context_impl.hpp>
#include <CL/sycl/device.hpp>
#include <CL/sycl/device_selector.hpp>
#include <CL/sycl/exception.hpp>
#include <CL/sycl/exception_list.hpp>
#include <CL/sycl/platform.hpp>
#include <CL/sycl/stl.hpp>
#include <algorithm>
#include <memory>
#include <utility>

// 4.6.2 Context class

__SYCL_INLINE namespace cl {
namespace sycl {
context::context(const async_handler &AsyncHandler)
    : context(default_selector().select_device(), AsyncHandler) {}

context::context(const device &Device, async_handler AsyncHandler)
    : context(vector_class<device>(1, Device), AsyncHandler) {}

context::context(const platform &Platform, async_handler AsyncHandler)
    : context(Platform.get_devices(), AsyncHandler) {}

context::context(const vector_class<device> &DeviceList,
                 async_handler AsyncHandler) {
  if (DeviceList.empty()) {
    throw invalid_parameter_error("DeviceList is empty.");
  }
  auto NonHostDeviceIter = std::find_if_not(
      DeviceList.begin(), DeviceList.end(),
      [&](const device &CurrentDevice) { return CurrentDevice.is_host(); });
  if (NonHostDeviceIter == DeviceList.end())
    impl =
        std::make_shared<detail::context_impl>(DeviceList[0], AsyncHandler);
  else {
    const device &NonHostDevice = *NonHostDeviceIter;
    const auto &NonHostPlatform = NonHostDevice.get_platform().get();
    if (std::any_of(DeviceList.begin(), DeviceList.end(),
                    [&](const device &CurrentDevice) {
                        return (CurrentDevice.is_host() ||
                                (CurrentDevice.get_platform().get() !=
                                 NonHostPlatform));
                    }))
      throw invalid_parameter_error(
          "Can't add devices across platforms to a single context.");
    else
      impl = std::make_shared<detail::context_impl>(DeviceList, AsyncHandler);
  }
}
context::context(cl_context ClContext, async_handler AsyncHandler) {
  impl = std::make_shared<detail::context_impl>(
          detail::pi::cast<detail::RT::PiContext>(ClContext), AsyncHandler);
}

#define PARAM_TRAITS_SPEC(param_type, param, ret_type)                         \
  template <> ret_type context::get_info<info::param_type::param>() const {    \
    return impl->get_info<info::param_type::param>();                          \
  }

#include <CL/sycl/info/context_traits.def>

#undef PARAM_TRAITS_SPEC

cl_context context::get() const { return impl->get(); }

bool context::is_host() const { return impl->is_host(); }

platform context::get_platform() const {
  return impl->get_info<info::context::platform>();
}

vector_class<device> context::get_devices() const {
  return impl->get_info<info::context::devices>();
}

context::context(shared_ptr_class<detail::context_impl> Impl) : impl(Impl) {}

} // namespace sycl
} // namespace cl
