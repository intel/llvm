//==------------------- device.cpp -----------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/device_impl.hpp>
#include <CL/sycl/detail/force_device.hpp>
#include <CL/sycl/device.hpp>
#include <CL/sycl/device_selector.hpp>
#include <CL/sycl/info/info_desc.hpp>

namespace cl {
namespace sycl {
namespace detail {
void force_type(info::device_type &t, const info::device_type &ft) {
  if (t == info::device_type::all) {
    t = ft;
  } else if (ft != info::device_type::all && t != ft) {
    throw cl::sycl::invalid_parameter_error("No device of forced type.");
  }
}
} // namespace detail

device::device() : impl(std::make_shared<detail::device_impl>()) {}

device::device(cl_device_id deviceId)
    : impl(std::make_shared<detail::device_impl>(
          detail::pi::cast<detail::RT::PiDevice>(deviceId))) {}

device::device(const device_selector &deviceSelector) {
  *this = deviceSelector.select_device();
}

vector_class<device> device::get_devices(info::device_type deviceType) {
  vector_class<device> devices;
  info::device_type forced_type = detail::get_forced_type();
  // Exclude devices which do not match requested device type
  if (detail::match_types(deviceType, forced_type)) {
    detail::force_type(deviceType, forced_type);
    for (const auto &plt : platform::get_platforms()) {
      vector_class<device> found_devices(plt.get_devices(deviceType));
      if (!found_devices.empty())
        devices.insert(devices.end(), found_devices.begin(),
                       found_devices.end());
    }
  }
  return devices;
}

cl_device_id device::get() const { return impl->get(); }

bool device::is_host() const { return impl->is_host(); }

bool device::is_cpu() const { return impl->is_cpu(); }

bool device::is_gpu() const { return impl->is_gpu(); }

bool device::is_accelerator() const { return impl->is_accelerator(); }

platform device::get_platform() const { return impl->get_platform(); }

template <info::partition_property prop>
vector_class<device> device::create_sub_devices(size_t ComputeUnits) const {
  return impl->create_sub_devices(ComputeUnits);
}
template vector_class<device>
device::create_sub_devices<info::partition_property::partition_equally>(
    size_t ComputeUnits) const;

template <info::partition_property prop>
vector_class<device>
device::create_sub_devices(const vector_class<size_t> &Counts) const {
  return impl->create_sub_devices(Counts);
}
template vector_class<device>
device::create_sub_devices<info::partition_property::partition_by_counts>(
    const vector_class<size_t> &Counts) const;

template <info::partition_property prop>
vector_class<device> device::create_sub_devices(
    info::partition_affinity_domain AffinityDomain) const {
  return impl->create_sub_devices(AffinityDomain);
}
template vector_class<device> device::create_sub_devices<
    info::partition_property::partition_by_affinity_domain>(
    info::partition_affinity_domain AffinityDomain) const;

bool device::has_extension(const string_class &extension_name) const {
  return impl->has_extension(extension_name);
}

template <info::device param>
typename info::param_traits<info::device, param>::return_type
device::get_info() const {
  return impl->template get_info<param>();
}

#define PARAM_TRAITS_SPEC(param_type, param, ret_type)                         \
  template ret_type device::get_info<info::param_type::param>() const;

#include <CL/sycl/info/device_traits.def>

#undef PARAM_TRAITS_SPEC

} // namespace sycl
} // namespace cl
