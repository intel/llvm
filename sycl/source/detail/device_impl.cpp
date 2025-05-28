//==----------------- device_impl.cpp - SYCL device ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/device_impl.hpp>
#include <detail/jit_compiler.hpp>
#include <detail/platform_impl.hpp>
#include <detail/ur_info_code.hpp>
#include <sycl/detail/ur.hpp>
#include <sycl/device.hpp>

#include <algorithm>

namespace sycl {
inline namespace _V1 {
namespace detail {

/// Constructs a SYCL device instance using the provided
/// UR device instance.
device_impl::device_impl(ur_device_handle_t Device, platform_impl &Platform,
                         device_impl::private_tag)
    : MDevice(Device), MPlatform(Platform.shared_from_this()),
      // No need to set MRootDevice when MAlwaysRootDevice is true
      MRootDevice(Platform.MAlwaysRootDevice
                      ? nullptr
                      : get_info_impl<UR_DEVICE_INFO_PARENT_DEVICE>()),
      // TODO catch an exception and put it to list of asynchronous exceptions:
      MCache{*this} {
  // Interoperability Constructor already calls DeviceRetain in
  // urDeviceCreateWithNativeHandle.
  getAdapter()->call<UrApiKind::urDeviceRetain>(MDevice);
}

device_impl::~device_impl() {
  try {
    // TODO catch an exception and put it to list of asynchronous exceptions
    const AdapterPtr &Adapter = getAdapter();
    ur_result_t Err =
        Adapter->call_nocheck<UrApiKind::urDeviceRelease>(MDevice);
    __SYCL_CHECK_UR_CODE_NO_EXC(Err);
  } catch (std::exception &e) {
    __SYCL_REPORT_EXCEPTION_TO_STREAM("exception in ~device_impl", e);
  }
}

bool device_impl::is_affinity_supported(
    info::partition_affinity_domain AffinityDomain) const {
  auto SupportedDomains = get_info<info::device::partition_affinity_domains>();
  return std::find(SupportedDomains.begin(), SupportedDomains.end(),
                   AffinityDomain) != SupportedDomains.end();
}

cl_device_id device_impl::get() const {
  // TODO catch an exception and put it to list of asynchronous exceptions
  __SYCL_OCL_CALL(clRetainDevice, ur::cast<cl_device_id>(getNative()));
  return ur::cast<cl_device_id>(getNative());
}

platform device_impl::get_platform() const {
  return createSyclObjFromImpl<platform>(MPlatform);
}

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
template <>
typename info::platform::version::return_type
device_impl::get_backend_info<info::platform::version>() const {
  if (getBackend() != backend::opencl) {
    throw sycl::exception(errc::backend_mismatch,
                          "the info::platform::version info descriptor can "
                          "only be queried with an OpenCL backend");
  }
  return get_platform().get_info<info::platform::version>();
}
#endif

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
template <>
typename info::device::version::return_type
device_impl::get_backend_info<info::device::version>() const {
  if (getBackend() != backend::opencl) {
    throw sycl::exception(errc::backend_mismatch,
                          "the info::device::version info descriptor can only "
                          "be queried with an OpenCL backend");
  }
  return get_info<info::device::version>();
}
#endif

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
template <>
typename info::device::backend_version::return_type
device_impl::get_backend_info<info::device::backend_version>() const {
  if (getBackend() != backend::ext_oneapi_level_zero) {
    throw sycl::exception(errc::backend_mismatch,
                          "the info::device::backend_version info descriptor "
                          "can only be queried with a Level Zero backend");
  }
  return "";
  // Currently The Level Zero backend does not define the value of this
  // information descriptor and implementations are encouraged to return the
  // empty string as per specification.
}
#endif

bool device_impl::has_extension(const std::string &ExtensionName) const {
  std::string AllExtensionNames = get_info_impl<UR_DEVICE_INFO_EXTENSIONS>();

  return (AllExtensionNames.find(ExtensionName) != std::string::npos);
}

bool device_impl::is_partition_supported(info::partition_property Prop) const {
  auto SupportedProperties = get_info<info::device::partition_properties>();
  return std::find(SupportedProperties.begin(), SupportedProperties.end(),
                   Prop) != SupportedProperties.end();
}

std::vector<device> device_impl::create_sub_devices(
    const ur_device_partition_properties_t *Properties,
    size_t SubDevicesCount) const {
  std::vector<ur_device_handle_t> SubDevices(SubDevicesCount);
  uint32_t ReturnedSubDevices = 0;
  const AdapterPtr &Adapter = getAdapter();
  Adapter->call<sycl::errc::invalid, UrApiKind::urDevicePartition>(
      MDevice, Properties, SubDevicesCount, SubDevices.data(),
      &ReturnedSubDevices);
  if (ReturnedSubDevices != SubDevicesCount) {
    throw sycl::exception(
        errc::invalid,
        "Could not partition to the specified number of sub-devices");
  }
  // TODO: Need to describe the subdevice model. Some sub_device management
  // may be necessary. What happens if create_sub_devices is called multiple
  // times with the same arguments?
  //
  std::vector<device> res;
  std::for_each(SubDevices.begin(), SubDevices.end(),
                [&res, this](const ur_device_handle_t &a_ur_device) {
                  device sycl_device = detail::createSyclObjFromImpl<device>(
                      MPlatform->getOrMakeDeviceImpl(a_ur_device));
                  res.push_back(sycl_device);
                });
  return res;
}

std::vector<device> device_impl::create_sub_devices(size_t ComputeUnits) const {
  if (!is_partition_supported(info::partition_property::partition_equally)) {
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Device does not support "
                          "sycl::info::partition_property::partition_equally.");
  }
  // If count exceeds the total number of compute units in the device, an
  // exception with the errc::invalid error code must be thrown.
  auto MaxComputeUnits = get_info<info::device::max_compute_units>();
  if (ComputeUnits > MaxComputeUnits)
    throw sycl::exception(errc::invalid,
                          "Total counts exceed max compute units");

  size_t SubDevicesCount = MaxComputeUnits / ComputeUnits;

  ur_device_partition_property_t Prop{};
  Prop.type = UR_DEVICE_PARTITION_EQUALLY;
  Prop.value.count = static_cast<uint32_t>(ComputeUnits);

  ur_device_partition_properties_t Properties{};
  Properties.stype = UR_STRUCTURE_TYPE_DEVICE_PARTITION_PROPERTIES;
  Properties.PropCount = 1;
  Properties.pProperties = &Prop;

  return create_sub_devices(&Properties, SubDevicesCount);
}

std::vector<device>
device_impl::create_sub_devices(const std::vector<size_t> &Counts) const {
  if (!is_partition_supported(info::partition_property::partition_by_counts)) {
    throw sycl::exception(
        make_error_code(errc::feature_not_supported),
        "Device does not support "
        "sycl::info::partition_property::partition_by_counts.");
  }

  std::vector<ur_device_partition_property_t> Props{};

  // Fill the properties vector with counts and validate it
  size_t TotalCounts = 0;
  size_t NonZeroCounts = 0;
  for (auto Count : Counts) {
    TotalCounts += Count;
    NonZeroCounts += (Count != 0) ? 1 : 0;
    Props.push_back(ur_device_partition_property_t{
        UR_DEVICE_PARTITION_BY_COUNTS, {static_cast<uint32_t>(Count)}});
  }

  ur_device_partition_properties_t Properties{};
  Properties.stype = UR_STRUCTURE_TYPE_DEVICE_PARTITION_PROPERTIES;
  Properties.pProperties = Props.data();
  Properties.PropCount = Props.size();

  // If the number of non-zero values in counts exceeds the device’s maximum
  // number of sub devices (as returned by info::device::
  // partition_max_sub_devices) an exception with the errc::invalid
  // error code must be thrown.
  if (NonZeroCounts > get_info<info::device::partition_max_sub_devices>())
    throw sycl::exception(errc::invalid,
                          "Total non-zero counts exceed max sub-devices");

  // If the total of all the values in the counts vector exceeds the total
  // number of compute units in the device (as returned by
  // info::device::max_compute_units), an exception with the errc::invalid
  // error code must be thrown.
  if (TotalCounts > get_info<info::device::max_compute_units>())
    throw sycl::exception(errc::invalid,
                          "Total counts exceed max compute units");

  return create_sub_devices(&Properties, Counts.size());
}

static inline std::string
affinityDomainToString(info::partition_affinity_domain AffinityDomain) {
  switch (AffinityDomain) {
#define __SYCL_AFFINITY_DOMAIN_STRING_CASE(DOMAIN)                             \
  case DOMAIN:                                                                 \
    return #DOMAIN;

    __SYCL_AFFINITY_DOMAIN_STRING_CASE(
        sycl::info::partition_affinity_domain::numa)
    __SYCL_AFFINITY_DOMAIN_STRING_CASE(
        sycl::info::partition_affinity_domain::L4_cache)
    __SYCL_AFFINITY_DOMAIN_STRING_CASE(
        sycl::info::partition_affinity_domain::L3_cache)
    __SYCL_AFFINITY_DOMAIN_STRING_CASE(
        sycl::info::partition_affinity_domain::L2_cache)
    __SYCL_AFFINITY_DOMAIN_STRING_CASE(
        sycl::info::partition_affinity_domain::L1_cache)
    __SYCL_AFFINITY_DOMAIN_STRING_CASE(
        sycl::info::partition_affinity_domain::next_partitionable)
#undef __SYCL_AFFINITY_DOMAIN_STRING_CASE
  default:
    assert(false && "Missing case for affinity domain.");
    return "unknown";
  }
}

std::vector<device> device_impl::create_sub_devices(
    info::partition_affinity_domain AffinityDomain) const {
  if (!is_partition_supported(
          info::partition_property::partition_by_affinity_domain)) {
    throw sycl::exception(
        make_error_code(errc::feature_not_supported),
        "Device does not support "
        "sycl::info::partition_property::partition_by_affinity_domain.");
  }
  if (!is_affinity_supported(AffinityDomain)) {
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Device does not support " +
                              affinityDomainToString(AffinityDomain) + ".");
  }

  ur_device_partition_property_t Prop;
  Prop.type = UR_DEVICE_PARTITION_BY_AFFINITY_DOMAIN;
  Prop.value.affinity_domain =
      static_cast<ur_device_affinity_domain_flags_t>(AffinityDomain);

  ur_device_partition_properties_t Properties{};
  Properties.stype = UR_STRUCTURE_TYPE_DEVICE_PARTITION_PROPERTIES;
  Properties.PropCount = 1;
  Properties.pProperties = &Prop;

  uint32_t SubDevicesCount = 0;
  const AdapterPtr &Adapter = getAdapter();
  Adapter->call<sycl::errc::invalid, UrApiKind::urDevicePartition>(
      MDevice, &Properties, 0, nullptr, &SubDevicesCount);

  return create_sub_devices(&Properties, SubDevicesCount);
}

std::vector<device> device_impl::create_sub_devices() const {
  if (!is_partition_supported(
          info::partition_property::ext_intel_partition_by_cslice)) {
    throw sycl::exception(
        make_error_code(errc::feature_not_supported),
        "Device does not support "
        "sycl::info::partition_property::ext_intel_partition_by_cslice.");
  }

  ur_device_partition_property_t Prop;
  Prop.type = UR_DEVICE_PARTITION_BY_CSLICE;

  ur_device_partition_properties_t Properties{};
  Properties.stype = UR_STRUCTURE_TYPE_DEVICE_PARTITION_PROPERTIES;
  Properties.pProperties = &Prop;
  Properties.PropCount = 1;

  uint32_t SubDevicesCount = 0;
  const AdapterPtr &Adapter = getAdapter();
  Adapter->call<UrApiKind::urDevicePartition>(MDevice, &Properties, 0, nullptr,
                                              &SubDevicesCount);

  return create_sub_devices(&Properties, SubDevicesCount);
}

ur_native_handle_t device_impl::getNative() const {
  auto Adapter = getAdapter();
  ur_native_handle_t Handle;
  Adapter->call<UrApiKind::urDeviceGetNativeHandle>(getHandleRef(), &Handle);
  if (getBackend() == backend::opencl) {
    __SYCL_OCL_CALL(clRetainDevice, ur::cast<cl_device_id>(Handle));
  }
  return Handle;
}

// On the first call this function queries for device timestamp
// along with host synchronized timestamp and stores it in member variable
// MDeviceHostBaseTime. Subsequent calls to this function would just retrieve
// the host timestamp, compute difference against the host timestamp in
// MDeviceHostBaseTime and calculate the device timestamp based on the
// difference.
//
// The MDeviceHostBaseTime is refreshed with new device and host timestamp
// after a certain interval (determined by TimeTillRefresh) to account for
// clock drift between host and device.
//
uint64_t device_impl::getCurrentDeviceTime() {
  using namespace std::chrono;
  uint64_t HostTime =
      duration_cast<nanoseconds>(steady_clock::now().time_since_epoch())
          .count();

  // To account for potential clock drift between host clock and device clock.
  // The value set is arbitrary: 200 seconds
  constexpr uint64_t TimeTillRefresh = 200e9;
  assert(HostTime >= MDeviceHostBaseTime.second);
  uint64_t Diff = HostTime - MDeviceHostBaseTime.second;

  // If getCurrentDeviceTime is called for the first time or we have to refresh.
  if (!MDeviceHostBaseTime.second || Diff > TimeTillRefresh) {
    const auto &Adapter = getAdapter();
    auto Result = Adapter->call_nocheck<UrApiKind::urDeviceGetGlobalTimestamps>(
        MDevice, &MDeviceHostBaseTime.first, &MDeviceHostBaseTime.second);
    // We have to remember base host timestamp right after UR call and it is
    // going to be used for calculation of the device timestamp at the next
    // getCurrentDeviceTime() call. We need to do it here because getAdapter()
    // and urDeviceGetGlobalTimestamps calls may take significant amount of
    // time, for example on the first call to getAdapter adapters may need to be
    // initialized. If we use timestamp from the beginning of the function then
    // the difference between host timestamps of the current
    // getCurrentDeviceTime and the next getCurrentDeviceTime will be incorrect
    // because it will include execution time of the code before we get device
    // timestamp from urDeviceGetGlobalTimestamps.
    HostTime =
        duration_cast<nanoseconds>(steady_clock::now().time_since_epoch())
            .count();
    if (Result == UR_RESULT_ERROR_INVALID_OPERATION) {
      // NOTE(UR port): Removed the call to GetLastError because  we shouldn't
      // be calling it after ERROR_INVALID_OPERATION: there is no
      // adapter-specific error.
      throw detail::set_ur_error(
          sycl::exception(
              make_error_code(errc::feature_not_supported),
              "Device and/or backend does not support querying timestamp."),
          UR_RESULT_ERROR_INVALID_OPERATION);
    } else {
      Adapter->checkUrResult<errc::feature_not_supported>(Result);
    }
    // Until next sync we will compute device time based on the host time
    // returned in HostTime, so make this our base host time.
    MDeviceHostBaseTime.second = HostTime;
    Diff = 0;
  }
  return MDeviceHostBaseTime.first + Diff;
}

bool device_impl::extOneapiCanBuild(
    ext::oneapi::experimental::source_language Language) {
  try {
    // Get the shared_ptr to this object from the platform that owns it.
    device_impl &Self = MPlatform->getOrMakeDeviceImpl(MDevice);
    return sycl::ext::oneapi::experimental::detail::
        is_source_kernel_bundle_supported(Language,
                                          std::vector<device_impl *>{&Self});

  } catch (sycl::exception &) {
    return false;
  }
}

bool device_impl::extOneapiCanCompile(
    ext::oneapi::experimental::source_language Language) {
  try {
    // Currently only SYCL language is supported for compiling.
    device_impl &Self = MPlatform->getOrMakeDeviceImpl(MDevice);
    return Language == ext::oneapi::experimental::source_language::sycl &&
           sycl::ext::oneapi::experimental::detail::
               is_source_kernel_bundle_supported(
                   Language, std::vector<device_impl *>{&Self});
  } catch (sycl::exception &) {
    return false;
  }
}

// Returns the strongest guarantee that can be provided by the host device for
// threads created at threadScope from a coordination scope given by
// coordinationScope
sycl::ext::oneapi::experimental::forward_progress_guarantee
device_impl::getHostProgressGuarantee(
    ext::oneapi::experimental::execution_scope,
    ext::oneapi::experimental::execution_scope) {
  return sycl::ext::oneapi::experimental::forward_progress_guarantee::
      weakly_parallel;
}

// Returns the strongest progress guarantee that can be provided by this device
// for threads created at threadScope from the coordination scope given by
// coordinationScope.
sycl::ext::oneapi::experimental::forward_progress_guarantee
device_impl::getProgressGuarantee(
    ext::oneapi::experimental::execution_scope threadScope,
    ext::oneapi::experimental::execution_scope coordinationScope) const {
  using forward_progress_guarantee =
      ext::oneapi::experimental::forward_progress_guarantee;
  using execution_scope = ext::oneapi::experimental::execution_scope;
  const int executionScopeSize = 4;
  (void)coordinationScope;
  int threadScopeNum = static_cast<int>(threadScope);
  // we get the immediate progress guarantee that is provided by each scope
  // between root_group and threadScope and then return the weakest of these.
  // Counterintuitively, this corresponds to taking the max of the enum values
  // because of how the forward_progress_guarantee enum values are declared.
  int guaranteeNum = static_cast<int>(
      getImmediateProgressGuarantee(execution_scope::root_group));
  for (int currentScope = executionScopeSize - 2; currentScope > threadScopeNum;
       --currentScope) {
    guaranteeNum = std::max(guaranteeNum,
                            static_cast<int>(getImmediateProgressGuarantee(
                                static_cast<execution_scope>(currentScope))));
  }
  return static_cast<forward_progress_guarantee>(guaranteeNum);
}

bool device_impl::supportsForwardProgress(
    ext::oneapi::experimental::forward_progress_guarantee guarantee,
    ext::oneapi::experimental::execution_scope threadScope,
    ext::oneapi::experimental::execution_scope coordinationScope) const {
  auto guarantees = getProgressGuaranteesUpTo(
      getProgressGuarantee(threadScope, coordinationScope));
  return std::find(guarantees.begin(), guarantees.end(), guarantee) !=
         guarantees.end();
}

// Returns the progress guarantee provided for a coordination scope
// given by coordination_scope for threads created at a scope
// immediately below coordination_scope. For example, for root_group
// coordination scope it returns the progress guarantee provided
// at root_group for threads created at work_group.
ext::oneapi::experimental::forward_progress_guarantee
device_impl::getImmediateProgressGuarantee(
    ext::oneapi::experimental::execution_scope coordination_scope) const {
  using forward_progress_guarantee =
      ext::oneapi::experimental::forward_progress_guarantee;
  using execution_scope = ext::oneapi::experimental::execution_scope;
  if (is_cpu() && getBackend() == backend::opencl) {
    switch (coordination_scope) {
    case execution_scope::root_group:
      return forward_progress_guarantee::parallel;
    case execution_scope::work_group:
    case execution_scope::sub_group:
      return forward_progress_guarantee::weakly_parallel;
    default:
      throw sycl::exception(sycl::errc::invalid,
                            "Work item is not a valid coordination scope!");
    }
  } else if (is_gpu() && getBackend() == backend::ext_oneapi_level_zero) {
    switch (coordination_scope) {
    case execution_scope::root_group:
    case execution_scope::work_group:
      return forward_progress_guarantee::concurrent;
    case execution_scope::sub_group:
      return forward_progress_guarantee::weakly_parallel;
    default:
      throw sycl::exception(sycl::errc::invalid,
                            "Work item is not a valid coordination scope!");
    }
  }
  return forward_progress_guarantee::weakly_parallel;
}

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
#define EXPORT_GET_INFO(PARAM)                                                 \
  template <>                                                                  \
  __SYCL_EXPORT PARAM::return_type device_impl::get_info<PARAM>() const {      \
    return get_info_abi_workaround<PARAM>();                                   \
  }

// clang-format off
EXPORT_GET_INFO(ext::intel::info::device::device_id)
EXPORT_GET_INFO(ext::intel::info::device::pci_address)
EXPORT_GET_INFO(ext::intel::info::device::gpu_eu_count)
EXPORT_GET_INFO(ext::intel::info::device::gpu_eu_simd_width)
EXPORT_GET_INFO(ext::intel::info::device::gpu_slices)
EXPORT_GET_INFO(ext::intel::info::device::gpu_subslices_per_slice)
EXPORT_GET_INFO(ext::intel::info::device::gpu_eu_count_per_subslice)
EXPORT_GET_INFO(ext::intel::info::device::gpu_hw_threads_per_eu)
EXPORT_GET_INFO(ext::intel::info::device::max_mem_bandwidth)
EXPORT_GET_INFO(ext::intel::info::device::uuid)
EXPORT_GET_INFO(ext::intel::info::device::free_memory)
EXPORT_GET_INFO(ext::intel::info::device::memory_clock_rate)
EXPORT_GET_INFO(ext::intel::info::device::memory_bus_width)
EXPORT_GET_INFO(ext::intel::info::device::max_compute_queue_indices)
EXPORT_GET_INFO(ext::intel::esimd::info::device::has_2d_block_io_support)
EXPORT_GET_INFO(ext::intel::info::device::current_clock_throttle_reasons)
EXPORT_GET_INFO(ext::intel::info::device::fan_speed)
EXPORT_GET_INFO(ext::intel::info::device::min_power_limit)
EXPORT_GET_INFO(ext::intel::info::device::max_power_limit)

EXPORT_GET_INFO(ext::codeplay::experimental::info::device::supports_fusion)
EXPORT_GET_INFO(ext::codeplay::experimental::info::device::max_registers_per_work_group)

EXPORT_GET_INFO(ext::oneapi::experimental::info::device::max_global_work_groups)
EXPORT_GET_INFO(ext::oneapi::experimental::info::device::max_work_groups<1>)
EXPORT_GET_INFO(ext::oneapi::experimental::info::device::max_work_groups<2>)
EXPORT_GET_INFO(ext::oneapi::experimental::info::device::max_work_groups<3>)
EXPORT_GET_INFO(ext::oneapi::experimental::info::device::work_group_progress_capabilities<ext::oneapi::experimental::execution_scope::root_group>)
EXPORT_GET_INFO(ext::oneapi::experimental::info::device::sub_group_progress_capabilities<ext::oneapi::experimental::execution_scope::root_group>)
EXPORT_GET_INFO(ext::oneapi::experimental::info::device::sub_group_progress_capabilities<ext::oneapi::experimental::execution_scope::work_group>)
EXPORT_GET_INFO(ext::oneapi::experimental::info::device::work_item_progress_capabilities<ext::oneapi::experimental::execution_scope::root_group>)
EXPORT_GET_INFO(ext::oneapi::experimental::info::device::work_item_progress_capabilities<ext::oneapi::experimental::execution_scope::work_group>)
EXPORT_GET_INFO(ext::oneapi::experimental::info::device::work_item_progress_capabilities<ext::oneapi::experimental::execution_scope::sub_group>)
EXPORT_GET_INFO(ext::oneapi::experimental::info::device::architecture)
EXPORT_GET_INFO(ext::oneapi::experimental::info::device::matrix_combinations)
EXPORT_GET_INFO(ext::oneapi::experimental::info::device::image_row_pitch_align)
EXPORT_GET_INFO(ext::oneapi::experimental::info::device::max_image_linear_row_pitch)
EXPORT_GET_INFO(ext::oneapi::experimental::info::device::max_image_linear_width)
EXPORT_GET_INFO(ext::oneapi::experimental::info::device::max_image_linear_height)
EXPORT_GET_INFO(ext::oneapi::experimental::info::device::mipmap_max_anisotropy)
EXPORT_GET_INFO(ext::oneapi::experimental::info::device::component_devices)
EXPORT_GET_INFO(ext::oneapi::experimental::info::device::composite_device)
EXPORT_GET_INFO(ext::oneapi::info::device::num_compute_units)
// clang-format on

#undef EXPORT_GET_INFO
#endif

} // namespace detail
} // namespace _V1
} // namespace sycl
