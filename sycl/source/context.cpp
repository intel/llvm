//==---------------- context.cpp - SYCL context ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/backend_impl.hpp>
#include <detail/context_impl.hpp>
#include <sycl/context.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/device.hpp>
#include <sycl/device_selector.hpp>
#include <sycl/exception.hpp>
#include <sycl/exception_list.hpp>
#include <sycl/platform.hpp>
#include <sycl/properties/all_properties.hpp>

#include <algorithm>
#include <memory>
#include <utility>

// 4.6.2 Context class

namespace sycl {
inline namespace _V1 {

context::context(const property_list &PropList) : context(device{}, PropList) {}

context::context(const async_handler &AsyncHandler,
                 const property_list &PropList)
    : context(device{}, AsyncHandler, PropList) {}

context::context(const device &Device, const property_list &PropList)
    : context(std::vector<device>(1, Device), PropList) {}

context::context(const device &Device, async_handler AsyncHandler,
                 const property_list &PropList)
    : context(std::vector<device>(1, Device), AsyncHandler, PropList) {}

context::context(const platform &Platform, const property_list &PropList)
    : context(Platform.get_devices(), PropList) {}

context::context(const platform &Platform, async_handler AsyncHandler,
                 const property_list &PropList)
    : context(Platform.get_devices(), AsyncHandler, PropList) {}

context::context(const std::vector<device> &DeviceList,
                 const property_list &PropList)
    : context(DeviceList, detail::defaultAsyncHandler, PropList) {}

context::context(const std::vector<device> &DeviceList,
                 async_handler AsyncHandler, const property_list &PropList) {
  if (DeviceList.empty()) {
    throw exception(make_error_code(errc::invalid), "DeviceList is empty.");
  }

  const auto &RefPlatform =
      detail::getSyclObjImpl(DeviceList[0].get_platform())->getHandleRef();
  if (std::any_of(DeviceList.begin(), DeviceList.end(),
                  [&](const device &CurrentDevice) {
                    return (detail::getSyclObjImpl(CurrentDevice.get_platform())
                                ->getHandleRef() != RefPlatform);
                  }))
    throw exception(make_error_code(errc::invalid),
                    "Can't add devices across platforms to a single context.");
  else
    impl = std::make_shared<detail::context_impl>(DeviceList, AsyncHandler,
                                                  PropList);
}
context::context(cl_context ClContext, async_handler AsyncHandler) {
  const auto &Plugin = sycl::detail::pi::getPlugin<backend::opencl>();
  impl = std::make_shared<detail::context_impl>(
      detail::pi::cast<sycl::detail::pi::PiContext>(ClContext), AsyncHandler,
      Plugin);
}

template <typename Param>
typename detail::is_context_info_desc<Param>::return_type
context::get_info() const {
  return impl->template get_info<Param>();
}

#define __SYCL_PARAM_TRAITS_SPEC(DescType, Desc, ReturnT, PiCode)              \
  template __SYCL_EXPORT ReturnT context::get_info<info::DescType::Desc>()     \
      const;

#include <sycl/info/context_traits.def>

#undef __SYCL_PARAM_TRAITS_SPEC

template <typename Param>
typename detail::is_backend_info_desc<Param>::return_type
context::get_backend_info() const {
  return impl->get_backend_info<Param>();
}

#define __SYCL_PARAM_TRAITS_SPEC(DescType, Desc, ReturnT, Picode)              \
  template __SYCL_EXPORT ReturnT                                               \
  context::get_backend_info<info::DescType::Desc>() const;

#include <sycl/info/sycl_backend_traits.def>

#undef __SYCL_PARAM_TRAITS_SPEC

cl_context context::get() const { return impl->get(); }

backend context::get_backend() const noexcept { return impl->getBackend(); }

platform context::get_platform() const {
  return impl->get_info<info::context::platform>();
}

std::vector<device> context::get_devices() const {
  return impl->get_info<info::context::devices>();
}

context::context(std::shared_ptr<detail::context_impl> Impl) : impl(Impl) {}

pi_native_handle context::getNative() const { return impl->getNative(); }

const property_list &context::getPropList() const {
  return impl->getPropList();
}

} // namespace _V1
} // namespace sycl
