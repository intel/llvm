//==----------- platform_impl.cpp ------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/allowlist.hpp>
#include <detail/config.hpp>
#include <detail/device_impl.hpp>
#include <detail/global_handler.hpp>
#include <detail/platform_impl.hpp>
#include <detail/platform_info.hpp>
#include <detail/ur_info_code.hpp>
#include <sycl/backend_types.hpp>
#include <sycl/detail/iostream_proxy.hpp>
#include <sycl/detail/ur.hpp>
#include <sycl/detail/util.hpp>
#include <sycl/device.hpp>
#include <sycl/info/info_desc.hpp>

#include <algorithm>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

namespace sycl {
inline namespace _V1 {
namespace detail {

using PlatformImplPtr = std::shared_ptr<platform_impl>;

PlatformImplPtr
platform_impl::getOrMakePlatformImpl(ur_platform_handle_t UrPlatform,
                                     const AdapterPtr &Adapter) {
  PlatformImplPtr Result;
  {
    const std::lock_guard<std::mutex> Guard(
        GlobalHandler::instance().getPlatformMapMutex());

    std::vector<PlatformImplPtr> &PlatformCache =
        GlobalHandler::instance().getPlatformCache();

    // If we've already seen this platform, return the impl
    for (const auto &PlatImpl : PlatformCache) {
      if (PlatImpl->getHandleRef() == UrPlatform)
        return PlatImpl;
    }

    // Otherwise make the impl
    Result = std::make_shared<platform_impl>(UrPlatform, Adapter);
    PlatformCache.emplace_back(Result);
  }

  return Result;
}

PlatformImplPtr
platform_impl::getPlatformFromUrDevice(ur_device_handle_t UrDevice,
                                       const AdapterPtr &Adapter) {
  ur_platform_handle_t Plt =
      nullptr; // TODO catch an exception and put it to list
  // of asynchronous exceptions
  Adapter->call<UrApiKind::urDeviceGetInfo>(UrDevice, UR_DEVICE_INFO_PLATFORM,
                                            sizeof(Plt), &Plt, nullptr);
  return getOrMakePlatformImpl(Plt, Adapter);
}

static bool IsBannedPlatform(platform Platform) {
  // The NVIDIA OpenCL platform is currently not compatible with DPC++
  // since it is only 1.2 but gets selected by default in many systems
  // There is also no support on the PTX backend for OpenCL consumption,
  // and there have been some internal reports.
  // To avoid problems on default users and deployment of DPC++ on platforms
  // where CUDA is available, the OpenCL support is disabled.
  //
  // There is also no support for the AMD HSA backend for OpenCL consumption,
  // as well as reported problems with device queries, so AMD OpenCL support
  // is disabled as well.
  //
  auto IsMatchingOpenCL = [](platform Platform, const std::string_view name) {
    const bool HasNameMatch = Platform.get_info<info::platform::name>().find(
                                  name) != std::string::npos;
    const auto Backend = detail::getSyclObjImpl(Platform)->getBackend();
    const bool IsMatchingOCL = (HasNameMatch && Backend == backend::opencl);
    if (detail::ur::trace(detail::ur::TraceLevel::TRACE_ALL) && IsMatchingOCL) {
      std::cout << "SYCL_UR_TRACE: " << name
                << " OpenCL platform found but is not compatible." << std::endl;
    }
    return IsMatchingOCL;
  };
  return IsMatchingOpenCL(Platform, "NVIDIA CUDA") ||
         IsMatchingOpenCL(Platform, "AMD Accelerated Parallel Processing");
}

// Get the vector of platforms supported by a given UR adapter
// replace uses of this with a helper in adapter object, the adapter
// objects will own the ur adapter handles and they'll need to pass them to
// urPlatformsGet - so urPlatformsGet will need to be wrapped with a helper
std::vector<platform> platform_impl::getAdapterPlatforms(AdapterPtr &Adapter,
                                                         bool Supported) {
  std::vector<platform> Platforms;

  auto UrPlatforms = Adapter->getUrPlatforms();

  if (UrPlatforms.empty()) {
    return Platforms;
  }

  for (const auto &UrPlatform : UrPlatforms) {
    platform Platform = detail::createSyclObjFromImpl<platform>(
        getOrMakePlatformImpl(UrPlatform, Adapter));
    const bool IsBanned = IsBannedPlatform(Platform);
    const bool HasAnyDevices =
        !Platform.get_devices(info::device_type::all).empty();

    if (!Supported) {
      if (IsBanned || !HasAnyDevices) {
        Platforms.push_back(Platform);
      }
    } else {
      if (IsBanned) {
        continue; // bail as early as possible, otherwise banned platforms may
                  // mess up device counting
      }

      // The SYCL spec says that a platform has one or more devices. ( SYCL
      // 2020 4.6.2 ) If we have an empty platform, we don't report it back
      // from platform::get_platforms().
      if (HasAnyDevices) {
        Platforms.push_back(Platform);
      }
    }
  }
  return Platforms;
}

// This routine has the side effect of registering each platform's last device
// id into each adapter, which is used for device counting.
std::vector<platform> platform_impl::get_platforms() {

  // See which platform we want to be served by which adapter.
  // There should be just one adapter serving each backend.
  std::vector<AdapterPtr> &Adapters = sycl::detail::ur::initializeUr();
  std::vector<std::pair<platform, AdapterPtr>> PlatformsWithAdapter;

  // Then check backend-specific adapters
  for (auto &Adapter : Adapters) {
    const auto &AdapterPlatforms = getAdapterPlatforms(Adapter);
    for (const auto &P : AdapterPlatforms) {
      PlatformsWithAdapter.push_back({P, Adapter});
    }
  }

  // For the selected platforms register them with their adapters
  std::vector<platform> Platforms;
  for (auto &Platform : PlatformsWithAdapter) {
    auto &Adapter = Platform.second;
    std::lock_guard<std::mutex> Guard(*Adapter->getAdapterMutex());
    Adapter->getPlatformId(getSyclObjImpl(Platform.first)->getHandleRef());
    Platforms.push_back(Platform.first);
  }

  // This initializes a function-local variable whose destructor is invoked as
  // the SYCL shared library is first being unloaded.
  GlobalHandler::registerEarlyShutdownHandler();

  return Platforms;
}

// Since ONEAPI_DEVICE_SELECTOR admits negative filters, we use type traits
// to distinguish the case where we are working with ONEAPI_DEVICE_SELECTOR
// in the places where the functionality diverges between these two
// environment variables.
// The return value is a vector that represents the indices of the chosen
// devices.
template <typename ListT, typename FilterT>
std::vector<int>
platform_impl::filterDeviceFilter(std::vector<ur_device_handle_t> &UrDevices,
                                  ListT *FilterList) const {

  constexpr bool is_ods_target = std::is_same_v<FilterT, ods_target>;

  if constexpr (is_ods_target) {

    // Since we are working with ods_target filters ,which can be negative,
    // we sort the filters so that all the negative filters appear before
    // all the positive filters.  This enables us to have the full list of
    // blacklisted devices by the time we get to the positive filters
    // so that if a positive filter matches a blacklisted device we do
    // not add it to the list of available devices.
    std::sort(FilterList->get().begin(), FilterList->get().end(),
              [](const ods_target &filter1, const ods_target &filter2) {
                return filter1.IsNegativeTarget && !filter2.IsNegativeTarget;
              });
  }

  // this map keeps track of devices discarded by negative filters, it is only
  // used in the ONEAPI_DEVICE_SELECTOR implemenation. It cannot be placed
  // in the if statement above because it will then be out of scope in the rest
  // of the function
  std::map<int, bool> Blacklist;
  // original indices keeps track of the device numbers of the chosen
  // devices and is whats returned by the function
  std::vector<int> original_indices;

  // Find out backend of the platform
  ur_platform_backend_t UrBackend = UR_PLATFORM_BACKEND_UNKNOWN;
  MAdapter->call<UrApiKind::urPlatformGetInfo>(
      MPlatform, UR_PLATFORM_INFO_BACKEND, sizeof(ur_platform_backend_t),
      &UrBackend, nullptr);
  backend Backend = convertUrBackend(UrBackend);

  int InsertIDx = 0;
  // DeviceIds should be given consecutive numbers across platforms in the same
  // backend
  std::lock_guard<std::mutex> Guard(*MAdapter->getAdapterMutex());
  int DeviceNum = MAdapter->getStartingDeviceId(MPlatform);
  for (ur_device_handle_t Device : UrDevices) {
    ur_device_type_t UrDevType = UR_DEVICE_TYPE_ALL;
    MAdapter->call<UrApiKind::urDeviceGetInfo>(Device, UR_DEVICE_INFO_TYPE,
                                               sizeof(ur_device_type_t),
                                               &UrDevType, nullptr);
    // Assumption here is that there is 1-to-1 mapping between UrDevType and
    // Sycl device type for GPU, CPU, and ACC.
    info::device_type DeviceType = info::device_type::all;
    switch (UrDevType) {
    default:
    case UR_DEVICE_TYPE_ALL:
      DeviceType = info::device_type::all;
      break;
    case UR_DEVICE_TYPE_GPU:
      DeviceType = info::device_type::gpu;
      break;
    case UR_DEVICE_TYPE_CPU:
      DeviceType = info::device_type::cpu;
      break;
    case UR_DEVICE_TYPE_FPGA:
      DeviceType = info::device_type::accelerator;
      break;
    }

    for (const FilterT &Filter : FilterList->get()) {
      backend FilterBackend = Filter.Backend.value_or(backend::all);
      // First, match the backend entry.
      if (FilterBackend != Backend && FilterBackend != backend::all)
        continue;
      info::device_type FilterDevType =
          Filter.DeviceType.value_or(info::device_type::all);

      // Match the device_num entry.
      if (Filter.DeviceNum && DeviceNum != Filter.DeviceNum.value())
        continue;

      if (FilterDevType != info::device_type::all &&
          FilterDevType != DeviceType)
        continue;

      if constexpr (is_ods_target) {
        // Dealing with ONEAPI_DEVICE_SELECTOR - check for negative filters.
        if (Blacklist[DeviceNum]) // already blacklisted.
          break;

        if (Filter.IsNegativeTarget) {
          // Filter is negative and the device matches the filter so
          // blacklist the device now.
          Blacklist[DeviceNum] = true;
          break;
        }
      }

      UrDevices[InsertIDx++] = Device;
      original_indices.push_back(DeviceNum);
      break;
    }
    DeviceNum++;
  }
  UrDevices.resize(InsertIDx);
  // remember the last backend that has gone through this filter function
  // to assign a unique device id number across platforms that belong to
  // the same backend. For example, opencl:cpu:0, opencl:acc:1, opencl:gpu:2
  MAdapter->setLastDeviceId(MPlatform, DeviceNum);
  return original_indices;
}

std::shared_ptr<device_impl>
platform_impl::getDeviceImpl(ur_device_handle_t UrDevice) {
  const std::lock_guard<std::mutex> Guard(MDeviceMapMutex);
  return getDeviceImplHelper(UrDevice);
}

std::shared_ptr<device_impl> platform_impl::getOrMakeDeviceImpl(
    ur_device_handle_t UrDevice,
    const std::shared_ptr<platform_impl> &PlatformImpl) {
  const std::lock_guard<std::mutex> Guard(MDeviceMapMutex);
  // If we've already seen this device, return the impl
  std::shared_ptr<device_impl> Result = getDeviceImplHelper(UrDevice);
  if (Result)
    return Result;

  // Otherwise make the impl
  Result = std::make_shared<device_impl>(UrDevice, PlatformImpl);
  MDeviceCache.emplace_back(Result);

  return Result;
}

static bool supportsAffinityDomain(const device &dev,
                                   info::partition_property partitionProp,
                                   info::partition_affinity_domain domain) {
  if (partitionProp != info::partition_property::partition_by_affinity_domain) {
    return true;
  }
  auto supported = dev.get_info<info::device::partition_affinity_domains>();
  auto It = std::find(std::begin(supported), std::end(supported), domain);
  return It != std::end(supported);
}

static bool supportsPartitionProperty(const device &dev,
                                      info::partition_property partitionProp) {
  auto supported = dev.get_info<info::device::partition_properties>();
  auto It =
      std::find(std::begin(supported), std::end(supported), partitionProp);
  return It != std::end(supported);
}

static std::vector<device> amendDeviceAndSubDevices(
    backend PlatformBackend, std::vector<device> &DeviceList,
    ods_target_list *OdsTargetList, const std::vector<int> &original_indices,
    PlatformImplPtr PlatformImpl) {
  constexpr info::partition_property partitionProperty =
      info::partition_property::partition_by_affinity_domain;
  constexpr info::partition_affinity_domain affinityDomain =
      info::partition_affinity_domain::next_partitionable;

  std::vector<device> FinalResult;
  // (Only) when amending sub-devices for ONEAPI_DEVICE_SELECTOR, all
  // sub-devices are treated as root.
  TempAssignGuard<bool> TAG(PlatformImpl->MAlwaysRootDevice, true);

  for (unsigned i = 0; i < DeviceList.size(); i++) {
    // device has already been screened. The question is whether it should be a
    // top level device and/or is expected to add its sub-devices to the list.
    device &dev = DeviceList[i];
    bool deviceAdded = false;
    for (ods_target target : OdsTargetList->get()) {
      backend TargetBackend = target.Backend.value_or(backend::all);
      if (PlatformBackend != TargetBackend && TargetBackend != backend::all)
        continue;

      bool deviceMatch = target.HasDeviceWildCard; // opencl:*
      if (target.DeviceType) {                     // opencl:gpu
        deviceMatch =
            ((target.DeviceType == info::device_type::all) ||
             (dev.get_info<info::device::device_type>() == target.DeviceType));

      } else if (target.DeviceNum) { // opencl:0
        deviceMatch = (target.DeviceNum.value() == original_indices[i]);
      }

      if (!deviceMatch)
        continue;

      // Top level matches. Do we add it, or subdevices, or sub-sub-devices?
      bool wantSubDevice = target.SubDeviceNum || target.HasSubDeviceWildCard;
      bool supportsSubPartitioning =
          (supportsPartitionProperty(dev, partitionProperty) &&
           supportsAffinityDomain(dev, partitionProperty, affinityDomain));
      bool wantSubSubDevice =
          target.SubSubDeviceNum || target.HasSubSubDeviceWildCard;

      if (!wantSubDevice) {
        // -- Add top level device only.
        if (!deviceAdded) {
          FinalResult.push_back(dev);
          deviceAdded = true;
        }
        continue;
      }

      if (!supportsSubPartitioning) {
        if (target.DeviceNum ||
            (target.DeviceType &&
             (target.DeviceType.value() != info::device_type::all))) {
          // This device was specifically requested and yet is not
          // partitionable.
          std::cout << "device is not partitionable: " << target << std::endl;
        }
        continue;
      }

      auto subDevices = dev.create_sub_devices<
          info::partition_property::partition_by_affinity_domain>(
          affinityDomain);
      if (target.SubDeviceNum) {
        if (subDevices.size() <= target.SubDeviceNum.value()) {
          std::cout << "subdevice index out of bounds: " << target << std::endl;
          continue;
        }
        subDevices[0] = subDevices[target.SubDeviceNum.value()];
        subDevices.resize(1);
      }

      if (!wantSubSubDevice) {
        // -- Add sub device(s) only.
        FinalResult.insert(FinalResult.end(), subDevices.begin(),
                           subDevices.end());
        continue;
      }

      // -- Add sub sub device(s).
      for (device subDev : subDevices) {
        bool supportsSubSubPartitioning =
            (supportsPartitionProperty(subDev, partitionProperty) &&
             supportsAffinityDomain(subDev, partitionProperty, affinityDomain));
        if (!supportsSubSubPartitioning) {
          if (target.SubDeviceNum) {
            // Parent subdevice was specifically requested, yet is not
            // partitionable.
            std::cout << "sub-device is not partitionable: " << target
                      << std::endl;
          }
          continue;
        }

        // Allright, lets get them sub-sub-devices.
        auto subSubDevices =
            subDev.create_sub_devices<partitionProperty>(affinityDomain);
        if (target.SubSubDeviceNum) {
          if (subSubDevices.size() <= target.SubSubDeviceNum.value()) {
            std::cout << "sub-sub-device index out of bounds: " << target
                      << std::endl;
            continue;
          }
          subSubDevices[0] = subSubDevices[target.SubSubDeviceNum.value()];
          subSubDevices.resize(1);
        }
        FinalResult.insert(FinalResult.end(), subSubDevices.begin(),
                           subSubDevices.end());
      }
    }
  }
  return FinalResult;
}

std::vector<device>
platform_impl::get_devices(info::device_type DeviceType) const {
  std::vector<device> Res;

  ods_target_list *OdsTargetList = SYCLConfig<ONEAPI_DEVICE_SELECTOR>::get();
  if (DeviceType == info::device_type::host)
    return Res;

  ur_device_type_t UrDeviceType = UR_DEVICE_TYPE_ALL;

  switch (DeviceType) {
  default:
  case info::device_type::all:
    UrDeviceType = UR_DEVICE_TYPE_ALL;
    break;
  case info::device_type::gpu:
    UrDeviceType = UR_DEVICE_TYPE_GPU;
    break;
  case info::device_type::cpu:
    UrDeviceType = UR_DEVICE_TYPE_CPU;
    break;
  case info::device_type::accelerator:
    UrDeviceType = UR_DEVICE_TYPE_FPGA;
    break;
  }

  uint32_t NumDevices = 0;
  MAdapter->call<UrApiKind::urDeviceGet>(MPlatform, UrDeviceType,
                                         0, // CP info::device_type::all
                                         nullptr, &NumDevices);
  const backend Backend = getBackend();

  if (NumDevices == 0) {
    // If platform doesn't have devices (even without filter)
    // LastDeviceIds[PlatformId] stay 0 that affects next platform devices num
    // analysis. Doing adjustment by simple copy of last device num from
    // previous platform.
    // Needs non const adapter reference.
    std::vector<AdapterPtr> &Adapters = sycl::detail::ur::initializeUr();
    auto It = std::find_if(Adapters.begin(), Adapters.end(),
                           [&Platform = MPlatform](AdapterPtr &Adapter) {
                             return Adapter->containsUrPlatform(Platform);
                           });
    if (It != Adapters.end()) {
      AdapterPtr &Adapter = *It;
      std::lock_guard<std::mutex> Guard(*Adapter->getAdapterMutex());
      Adapter->adjustLastDeviceId(MPlatform);
    }
    return Res;
  }

  std::vector<ur_device_handle_t> UrDevices(NumDevices);
  // TODO catch an exception and put it to list of asynchronous exceptions
  MAdapter->call<UrApiKind::urDeviceGet>(
      MPlatform,
      UrDeviceType, // CP info::device_type::all
      NumDevices, UrDevices.data(), nullptr);

  // Some elements of UrDevices vector might be filtered out, so make a copy of
  // handles to do a cleanup later
  std::vector<ur_device_handle_t> UrDevicesToCleanUp = UrDevices;

  // Filter out devices that are not present in the SYCL_DEVICE_ALLOWLIST
  if (SYCLConfig<SYCL_DEVICE_ALLOWLIST>::get())
    applyAllowList(UrDevices, MPlatform, MAdapter);

  // The first step is to filter out devices that are not compatible with
  // ONEAPI_DEVICE_SELECTOR. This is also the mechanism by which top level
  // device ids are assigned.
  std::vector<int> PlatformDeviceIndices;
  if (OdsTargetList) {
    PlatformDeviceIndices = filterDeviceFilter<ods_target_list, ods_target>(
        UrDevices, OdsTargetList);
  }

  // The next step is to inflate the filtered UrDevices into SYCL Device
  // objects.
  PlatformImplPtr PlatformImpl = getOrMakePlatformImpl(MPlatform, MAdapter);
  std::transform(
      UrDevices.begin(), UrDevices.end(), std::back_inserter(Res),
      [PlatformImpl](const ur_device_handle_t UrDevice) -> device {
        return detail::createSyclObjFromImpl<device>(
            PlatformImpl->getOrMakeDeviceImpl(UrDevice, PlatformImpl));
      });

  // The reference counter for handles, that we used to create sycl objects, is
  // incremented, so we need to call release here.
  for (ur_device_handle_t &UrDev : UrDevicesToCleanUp)
    MAdapter->call<UrApiKind::urDeviceRelease>(UrDev);

  // If we aren't using ONEAPI_DEVICE_SELECTOR, then we are done.
  // and if there are no devices so far, there won't be any need to replace them
  // with subdevices.
  if (!OdsTargetList || Res.size() == 0)
    return Res;

  // Otherwise, our last step is to revisit the devices, possibly replacing
  // them with subdevices (which have been ignored until now)
  return amendDeviceAndSubDevices(Backend, Res, OdsTargetList,
                                  PlatformDeviceIndices, PlatformImpl);
}

bool platform_impl::has_extension(const std::string &ExtensionName) const {
  std::string AllExtensionNames = get_platform_info_string_impl(
      MPlatform, getAdapter(),
      detail::UrInfoCode<info::platform::extensions>::value);
  return (AllExtensionNames.find(ExtensionName) != std::string::npos);
}

bool platform_impl::supports_usm() const {
  return getBackend() != backend::opencl ||
         has_extension("cl_intel_unified_shared_memory");
}

ur_native_handle_t platform_impl::getNative() const {
  const auto &Adapter = getAdapter();
  ur_native_handle_t Handle = 0;
  Adapter->call<UrApiKind::urPlatformGetNativeHandle>(getHandleRef(), &Handle);
  return Handle;
}

template <typename Param>
typename Param::return_type platform_impl::get_info() const {
  return get_platform_info<Param>(this->getHandleRef(), getAdapter());
}

template <>
typename info::platform::version::return_type
platform_impl::get_backend_info<info::platform::version>() const {
  if (getBackend() != backend::opencl) {
    throw sycl::exception(errc::backend_mismatch,
                          "the info::platform::version info descriptor can "
                          "only be queried with an OpenCL backend");
  }
  return get_info<info::platform::version>();
}

device select_device(DSelectorInvocableType DeviceSelectorInvocable,
                     std::vector<device> &Devices);

template <>
typename info::device::version::return_type
platform_impl::get_backend_info<info::device::version>() const {
  if (getBackend() != backend::opencl) {
    throw sycl::exception(errc::backend_mismatch,
                          "the info::device::version info descriptor can only "
                          "be queried with an OpenCL backend");
  }
  auto Devices = get_devices();
  if (Devices.empty()) {
    return "No available device";
  }
  // Use default selector to pick a device.
  return select_device(default_selector_v, Devices)
      .get_info<info::device::version>();
}

template <>
typename info::device::backend_version::return_type
platform_impl::get_backend_info<info::device::backend_version>() const {
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

// All devices on the platform must have the given aspect.
bool platform_impl::has(aspect Aspect) const {
  for (const auto &dev : get_devices()) {
    if (dev.has(Aspect) == false) {
      return false;
    }
  }
  return true;
}

std::shared_ptr<device_impl>
platform_impl::getDeviceImplHelper(ur_device_handle_t UrDevice) {
  for (const std::weak_ptr<device_impl> &DeviceWP : MDeviceCache) {
    if (std::shared_ptr<device_impl> Device = DeviceWP.lock()) {
      if (Device->getHandleRef() == UrDevice)
        return Device;
    }
  }
  return nullptr;
}

#define __SYCL_PARAM_TRAITS_SPEC(DescType, Desc, ReturnT, PiCode)              \
  template ReturnT platform_impl::get_info<info::platform::Desc>() const;

#include <sycl/info/platform_traits.def>
#undef __SYCL_PARAM_TRAITS_SPEC

} // namespace detail
} // namespace _V1
} // namespace sycl
