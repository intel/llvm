//==---------------- context.cpp - SYCL context ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/device.hpp>
#include <CL/sycl/device_selector.hpp>
#include <CL/sycl/exception.hpp>
#include <CL/sycl/exception_list.hpp>
#include <CL/sycl/platform.hpp>
#include <CL/sycl/properties/all_properties.hpp>
#include <CL/sycl/stl.hpp>
#include <detail/backend_impl.hpp>
#include <detail/context_impl.hpp>

#include <algorithm>
#include <memory>
#include <utility>

// 4.6.2 Context class

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
context::context(const property_list &PropList)
    : context(default_selector().select_device(), PropList) {}

context::context(const async_handler &AsyncHandler,
                 const property_list &PropList)
    : context(default_selector().select_device(), AsyncHandler, PropList) {}

context::context(const device &Device, const property_list &PropList)
    : context(vector_class<device>(1, Device), PropList) {}

context::context(const device &Device, async_handler AsyncHandler,
                 const property_list &PropList)
    : context(vector_class<device>(1, Device), AsyncHandler, PropList) {}

context::context(const platform &Platform, const property_list &PropList)
    : context(Platform.get_devices(), PropList) {}

context::context(const platform &Platform, async_handler AsyncHandler,
                 const property_list &PropList)
    : context(Platform.get_devices(), AsyncHandler, PropList) {}

context::context(const vector_class<device> &DeviceList,
                 const property_list &PropList)
    : context(DeviceList, async_handler{}, PropList) {}

context::context(const vector_class<device> &DeviceList,
                 async_handler AsyncHandler, const property_list &PropList) {
  if (DeviceList.empty()) {
    throw invalid_parameter_error("DeviceList is empty.", PI_INVALID_VALUE);
  }
  auto NonHostDeviceIter = std::find_if_not(
      DeviceList.begin(), DeviceList.end(),
      [&](const device &CurrentDevice) { return CurrentDevice.is_host(); });
  if (NonHostDeviceIter == DeviceList.end())
    impl = std::make_shared<detail::context_impl>(DeviceList[0], AsyncHandler,
                                                  PropList);
  else {
    const device &NonHostDevice = *NonHostDeviceIter;
    const auto &NonHostPlatform =
        detail::getSyclObjImpl(NonHostDevice.get_platform())->getHandleRef();
    if (std::any_of(DeviceList.begin(), DeviceList.end(),
                    [&](const device &CurrentDevice) {
                      return (
                          CurrentDevice.is_host() ||
                          (detail::getSyclObjImpl(CurrentDevice.get_platform())
                               ->getHandleRef() != NonHostPlatform));
                    }))
      throw invalid_parameter_error(
          "Can't add devices across platforms to a single context.",
          PI_INVALID_DEVICE);
    else
      impl = std::make_shared<detail::context_impl>(DeviceList, AsyncHandler,
                                                    PropList);
  }
}
context::context(cl_context ClContext, async_handler AsyncHandler) {
  const auto &Plugin = RT::getPlugin<backend::opencl>();
  impl = std::make_shared<detail::context_impl>(
      detail::pi::cast<detail::RT::PiContext>(ClContext), AsyncHandler, Plugin);
}

#define __SYCL_PARAM_TRAITS_SPEC(param_type, param, ret_type)                  \
  template <>                                                                  \
  __SYCL_EXPORT ret_type context::get_info<info::param_type::param>() const {  \
    return impl->get_info<info::param_type::param>();                          \
  }

#include <CL/sycl/info/context_traits.def>

#undef __SYCL_PARAM_TRAITS_SPEC

#define __SYCL_PARAM_TRAITS_SPEC(param_type)                                   \
  template <> __SYCL_EXPORT bool context::has_property<param_type>() const {   \
    return impl->has_property<param_type>();                                   \
  }
#include <CL/sycl/detail/properties_traits.def>

#undef __SYCL_PARAM_TRAITS_SPEC

#define __SYCL_PARAM_TRAITS_SPEC(param_type)                                   \
  template <>                                                                  \
  __SYCL_EXPORT param_type context::get_property<param_type>() const {         \
    return impl->get_property<param_type>();                                   \
  }
#include <CL/sycl/detail/properties_traits.def>

#undef __SYCL_PARAM_TRAITS_SPEC

cl_context context::get() const { return impl->get(); }

bool context::is_host() const { return impl->is_host(); }

backend context::get_backend() const noexcept { return getImplBackend(impl); }

platform context::get_platform() const {
  return impl->get_info<info::context::platform>();
}

vector_class<device> context::get_devices() const {
  return impl->get_info<info::context::devices>();
}

context::context(shared_ptr_class<detail::context_impl> Impl) : impl(Impl) {}

pi_native_handle context::getNative() const { return impl->getNative(); }

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
