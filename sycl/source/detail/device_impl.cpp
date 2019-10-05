//==----------------- device_impl.cpp - SYCL device ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/device_impl.hpp>
#include <CL/sycl/device.hpp>

namespace cl {
namespace sycl {
namespace detail {

vector_class<device>
device_impl_pi::create_sub_devices(
  const cl_device_partition_property *Properties,
  size_t SubDevicesCount) const {

  vector_class<RT::PiDevice> SubDevices(SubDevicesCount);
  pi_uint32 ReturnedSubDevices = 0;
  PI_CALL(RT::piDevicePartition(m_device, Properties, SubDevicesCount,
                                SubDevices.data(), &ReturnedSubDevices));
  // TODO: check that returned number of sub-devices matches what was
  // requested, otherwise this walk below is wrong.
  //
  // TODO: Need to describe the subdevice model. Some sub_device management
  // may be necessary. What happens if create_sub_devices is called multiple
  // times with the same arguments?
  //
  vector_class<device> res;
  std::for_each(SubDevices.begin(), SubDevices.end(),
                [&res](const RT::PiDevice &a_pi_device) {
    device sycl_device =
      detail::createSyclObjFromImpl<device>(
        std::make_shared<device_impl_pi>(a_pi_device));
    res.push_back(sycl_device);
  });
  return res;
}

vector_class<device>
device_impl_pi::create_sub_devices(size_t ComputeUnits) const {

  if (!is_partition_supported(info::partition_property::partition_equally)) {
    throw cl::sycl::feature_not_supported();
  }
  size_t SubDevicesCount =
      get_info<info::device::max_compute_units>() / ComputeUnits;
  const cl_device_partition_property Properties[3] = {
      CL_DEVICE_PARTITION_EQUALLY, (cl_device_partition_property)ComputeUnits,
      0};
  return create_sub_devices(Properties, SubDevicesCount);
}

vector_class<device>
device_impl_pi::create_sub_devices(const vector_class<size_t> &Counts) const {

  if (!is_partition_supported(
          info::partition_property::partition_by_counts)) {
    throw cl::sycl::feature_not_supported();
  }
  static const cl_device_partition_property P[] = {
      CL_DEVICE_PARTITION_BY_COUNTS, CL_DEVICE_PARTITION_BY_COUNTS_LIST_END,
      0};
  vector_class<cl_device_partition_property> Properties(P, P + 3);
  Properties.insert(Properties.begin() + 1, Counts.begin(), Counts.end());
  return create_sub_devices(Properties.data(), Counts.size());
}

vector_class<device>
device_impl_pi::create_sub_devices(
  info::partition_affinity_domain AffinityDomain) const {

  if (!is_partition_supported(
          info::partition_property::partition_by_affinity_domain) ||
      !is_affinity_supported(AffinityDomain)) {
    throw cl::sycl::feature_not_supported();
  }
  const cl_device_partition_property Properties[3] = {
      CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN,
      (cl_device_partition_property)AffinityDomain, 0};
  size_t SubDevicesCount =
      get_info<info::device::partition_max_sub_devices>();
  return create_sub_devices(Properties, SubDevicesCount);
}

} // namespace detail
} // namespace sycl
} // namespace cl
