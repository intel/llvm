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
#include <sycl/backend.hpp>
#include <sycl/detail/iostream_proxy.hpp>
#include <sycl/detail/util.hpp>
#include <sycl/device.hpp>

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

PlatformImplPtr platform_impl::getHostPlatformImpl() {
  static PlatformImplPtr HostImpl = std::make_shared<platform_impl>();

  return HostImpl;
}

PlatformImplPtr
platform_impl::getOrMakePlatformImpl(sycl::detail::pi::PiPlatform PiPlatform,
                                     const PluginPtr &Plugin) {
  PlatformImplPtr Result;
  {
    const std::lock_guard<std::mutex> Guard(
        GlobalHandler::instance().getPlatformMapMutex());

    std::vector<PlatformImplPtr> &PlatformCache =
        GlobalHandler::instance().getPlatformCache();

    // If we've already seen this platform, return the impl
    for (const auto &PlatImpl : PlatformCache) {
      if (PlatImpl->getHandleRef() == PiPlatform)
        return PlatImpl;
    }

    // Otherwise make the impl
    Result = std::make_shared<platform_impl>(PiPlatform, Plugin);
    PlatformCache.emplace_back(Result);
  }

  return Result;
}

PlatformImplPtr
platform_impl::getPlatformFromPiDevice(sycl::detail::pi::PiDevice PiDevice,
                                       const PluginPtr &Plugin) {
  sycl::detail::pi::PiPlatform Plt =
      nullptr; // TODO catch an exception and put it to list
  // of asynchronous exceptions
  Plugin->call<PiApiKind::piDeviceGetInfo>(PiDevice, PI_DEVICE_INFO_PLATFORM,
                                           sizeof(Plt), &Plt, nullptr);
  return getOrMakePlatformImpl(Plt, Plugin);
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
    if (getSyclObjImpl(Platform)->is_host())
      return false;

    const bool HasNameMatch = Platform.get_info<info::platform::name>().find(
                                  name) != std::string::npos;
    const auto Backend = detail::getSyclObjImpl(Platform)->getBackend();
    const bool IsMatchingOCL = (HasNameMatch && Backend == backend::opencl);
    if (detail::pi::trace(detail::pi::TraceLevel::PI_TRACE_ALL) &&
        IsMatchingOCL) {
      std::cout << "SYCL_PI_TRACE[all]: " << name
                << " OpenCL platform found but is not compatible." << std::endl;
    }
    return IsMatchingOCL;
  };
  return IsMatchingOpenCL(Platform, "NVIDIA CUDA") ||
         IsMatchingOpenCL(Platform, "AMD Accelerated Parallel Processing");
}

// This routine has the side effect of registering each platform's last device
// id into each plugin, which is used for device counting.
std::vector<platform> platform_impl::get_platforms() {

  // Get the vector of platforms supported by a given PI plugin
  auto getPluginPlatforms = [](PluginPtr &Plugin) {
    std::vector<platform> Platforms;
    pi_uint32 NumPlatforms = 0;
    if (Plugin->call_nocheck<PiApiKind::piPlatformsGet>(
            0, nullptr, &NumPlatforms) != PI_SUCCESS)
      return Platforms;

    if (NumPlatforms) {
      std::vector<sycl::detail::pi::PiPlatform> PiPlatforms(NumPlatforms);
      if (Plugin->call_nocheck<PiApiKind::piPlatformsGet>(
              NumPlatforms, PiPlatforms.data(), nullptr) != PI_SUCCESS)
        return Platforms;

      for (const auto &PiPlatform : PiPlatforms) {
        platform Platform = detail::createSyclObjFromImpl<platform>(
            getOrMakePlatformImpl(PiPlatform, Plugin));
        if (IsBannedPlatform(Platform)) {
          continue; // bail as early as possible, otherwise banned platforms may
                    // mess up device counting
        }

        // The SYCL spec says that a platform has one or more devices. ( SYCL
        // 2020 4.6.2 ) If we have an empty platform, we don't report it back
        // from platform::get_platforms().
        if (!Platform.get_devices(info::device_type::all).empty()) {
          Platforms.push_back(Platform);
        }
      }
    }
    return Platforms;
  };

  static const bool PreferUR = [] {
    const char *PreferURStr = std::getenv("SYCL_PREFER_UR");
    return (PreferURStr && (std::stoi(PreferURStr) != 0));
  }();

  // See which platform we want to be served by which plugin.
  // There should be just one plugin serving each backend.
  std::vector<PluginPtr> &Plugins = sycl::detail::pi::initialize();
  std::vector<std::pair<platform, PluginPtr>> PlatformsWithPlugin;

  // First check Unified Runtime
  // Keep track of backends covered by UR
  std::unordered_set<backend> BackendsUR;
  if (PreferUR) {
    PluginPtr *PluginUR = nullptr;
    for (PluginPtr &Plugin : Plugins) {
      if (Plugin->hasBackend(backend::all)) { // this denotes UR
        PluginUR = &Plugin;
        break;
      }
    }
    if (PluginUR) {
      for (const auto &P : getPluginPlatforms(*PluginUR)) {
        PlatformsWithPlugin.push_back({P, *PluginUR});
        BackendsUR.insert(getSyclObjImpl(P)->getBackend());
      }
    }
  }

  // Then check backend-specific plugins
  for (auto &Plugin : Plugins) {
    if (Plugin->hasBackend(backend::all)) {
      continue; // skip UR on this pass
    }
    const auto &PluginPlatforms = getPluginPlatforms(Plugin);
    for (const auto &P : PluginPlatforms) {
      // Only add those not already covered by UR
      if (BackendsUR.find(getSyclObjImpl(P)->getBackend()) ==
          BackendsUR.end()) {
        PlatformsWithPlugin.push_back({P, Plugin});
      }
    }
  }

  // For the selected platforms register them with their plugins
  std::vector<platform> Platforms;
  for (auto &Platform : PlatformsWithPlugin) {
    auto &Plugin = Platform.second;
    std::lock_guard<std::mutex> Guard(*Plugin->getPluginMutex());
    Plugin->getPlatformId(getSyclObjImpl(Platform.first)->getHandleRef());
    Platforms.push_back(Platform.first);
  }

  // Register default context release handler after plugins have been loaded and
  // after the first calls to each plugin. This initializes a function-local
  // variable that should be destroyed before any global variables in the
  // plugins are destroyed. This is done after the first call to the backends to
  // ensure any lazy-loaded dependencies are loaded prior to the handler
  // variable's initialization. Note: The default context release handler is not
  // guaranteed to be destroyed before function-local static variables as they
  // may be initialized after.
  GlobalHandler::registerDefaultContextReleaseHandler();

  return Platforms;
}

// Since ONEAPI_DEVICE_SELECTOR admits negative filters, we use type traits
// to distinguish the case where we are working with ONEAPI_DEVICE_SELECTOR
// in the places where the functionality diverges between these two
// environment variables.
// The return value is a vector that represents the indices of the chosen
// devices.
template <typename ListT, typename FilterT>
std::vector<int> platform_impl::filterDeviceFilter(
    std::vector<sycl::detail::pi::PiDevice> &PiDevices,
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
  sycl::detail::pi::PiPlatformBackend PiBackend;
  MPlugin->call<PiApiKind::piPlatformGetInfo>(
      MPlatform, PI_EXT_PLATFORM_INFO_BACKEND,
      sizeof(sycl::detail::pi::PiPlatformBackend), &PiBackend, nullptr);
  backend Backend = convertBackend(PiBackend);

  int InsertIDx = 0;
  // DeviceIds should be given consecutive numbers across platforms in the same
  // backend
  std::lock_guard<std::mutex> Guard(*MPlugin->getPluginMutex());
  int DeviceNum = MPlugin->getStartingDeviceId(MPlatform);
  for (sycl::detail::pi::PiDevice Device : PiDevices) {
    sycl::detail::pi::PiDeviceType PiDevType;
    MPlugin->call<PiApiKind::piDeviceGetInfo>(
        Device, PI_DEVICE_INFO_TYPE, sizeof(sycl::detail::pi::PiDeviceType),
        &PiDevType, nullptr);
    // Assumption here is that there is 1-to-1 mapping between PiDevType and
    // Sycl device type for GPU, CPU, and ACC.
    info::device_type DeviceType = pi::cast<info::device_type>(PiDevType);

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

      PiDevices[InsertIDx++] = Device;
      original_indices.push_back(DeviceNum);
      break;
    }
    DeviceNum++;
  }
  PiDevices.resize(InsertIDx);
  // remember the last backend that has gone through this filter function
  // to assign a unique device id number across platforms that belong to
  // the same backend. For example, opencl:cpu:0, opencl:acc:1, opencl:gpu:2
  MPlugin->setLastDeviceId(MPlatform, DeviceNum);
  return original_indices;
}

std::shared_ptr<device_impl>
platform_impl::getDeviceImpl(sycl::detail::pi::PiDevice PiDevice) {
  const std::lock_guard<std::mutex> Guard(MDeviceMapMutex);
  return getDeviceImplHelper(PiDevice);
}

std::shared_ptr<device_impl> platform_impl::getOrMakeDeviceImpl(
    sycl::detail::pi::PiDevice PiDevice,
    const std::shared_ptr<platform_impl> &PlatformImpl) {
  const std::lock_guard<std::mutex> Guard(MDeviceMapMutex);
  // If we've already seen this device, return the impl
  std::shared_ptr<device_impl> Result = getDeviceImplHelper(PiDevice);
  if (Result)
    return Result;

  // Otherwise make the impl
  Result = std::make_shared<device_impl>(PiDevice, PlatformImpl);
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
#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
  // Will we be filtering with SYCL_DEVICE_FILTER or ONEAPI_DEVICE_SELECTOR ?
  // We do NOT attempt to support both simultaneously.
  device_filter_list *FilterList = SYCLConfig<SYCL_DEVICE_FILTER>::get();
#endif

  if (is_host() && (DeviceType == info::device_type::host ||
                    DeviceType == info::device_type::all)) {
    Res.push_back(
        createSyclObjFromImpl<device>(device_impl::getHostDeviceImpl()));
  }

  // If any DeviceType other than host was requested for host platform,
  // an empty vector will be returned.
  if (is_host() || DeviceType == info::device_type::host)
    return Res;

  pi_uint32 NumDevices = 0;
  MPlugin->call<PiApiKind::piDevicesGet>(
      MPlatform, pi::cast<sycl::detail::pi::PiDeviceType>(DeviceType),
      0, // CP info::device_type::all
      pi::cast<sycl::detail::pi::PiDevice *>(nullptr), &NumDevices);
  const backend Backend = getBackend();

  if (NumDevices == 0) {
    // If platform doesn't have devices (even without filter)
    // LastDeviceIds[PlatformId] stay 0 that affects next platform devices num
    // analysis. Doing adjustment by simple copy of last device num from
    // previous platform.
    // Needs non const plugin reference.
    std::vector<PluginPtr> &Plugins = sycl::detail::pi::initialize();
    auto It = std::find_if(Plugins.begin(), Plugins.end(),
                           [&Platform = MPlatform](PluginPtr &Plugin) {
                             return Plugin->containsPiPlatform(Platform);
                           });
    if (It != Plugins.end()) {
      PluginPtr &Plugin = *It;
      std::lock_guard<std::mutex> Guard(*Plugin->getPluginMutex());
      Plugin->adjustLastDeviceId(MPlatform);
    }
    return Res;
  }

  std::vector<sycl::detail::pi::PiDevice> PiDevices(NumDevices);
  // TODO catch an exception and put it to list of asynchronous exceptions
  MPlugin->call<PiApiKind::piDevicesGet>(
      MPlatform,
      pi::cast<sycl::detail::pi::PiDeviceType>(
          DeviceType), // CP info::device_type::all
      NumDevices, PiDevices.data(), nullptr);

  // Some elements of PiDevices vector might be filtered out, so make a copy of
  // handles to do a cleanup later
  std::vector<sycl::detail::pi::PiDevice> PiDevicesToCleanUp = PiDevices;

  // Filter out devices that are not present in the SYCL_DEVICE_ALLOWLIST
  if (SYCLConfig<SYCL_DEVICE_ALLOWLIST>::get())
    applyAllowList(PiDevices, MPlatform, MPlugin);

  // The first step is to filter out devices that are not compatible with
  // ONEAPI_DEVICE_SELECTOR. This is also the mechanism by which top level
  // device ids are assigned.
  std::vector<int> PlatformDeviceIndices;
  if (OdsTargetList) {
#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
    if (FilterList) {
      throw sycl::exception(sycl::make_error_code(errc::invalid),
                            "ONEAPI_DEVICE_SELECTOR cannot be used in "
                            "conjunction with SYCL_DEVICE_FILTER");
    }
#endif
    PlatformDeviceIndices = filterDeviceFilter<ods_target_list, ods_target>(
        PiDevices, OdsTargetList);
#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
  } else if (FilterList) {
    PlatformDeviceIndices =
        filterDeviceFilter<device_filter_list, device_filter>(PiDevices,
                                                              FilterList);
#endif
  }

  // The next step is to inflate the filtered PIDevices into SYCL Device
  // objects.
  PlatformImplPtr PlatformImpl = getOrMakePlatformImpl(MPlatform, MPlugin);
  std::transform(
      PiDevices.begin(), PiDevices.end(), std::back_inserter(Res),
      [PlatformImpl](const sycl::detail::pi::PiDevice &PiDevice) -> device {
        return detail::createSyclObjFromImpl<device>(
            PlatformImpl->getOrMakeDeviceImpl(PiDevice, PlatformImpl));
      });

  // The reference counter for handles, that we used to create sycl objects, is
  // incremented, so we need to call release here.
  for (sycl::detail::pi::PiDevice &PiDev : PiDevicesToCleanUp)
    MPlugin->call<PiApiKind::piDeviceRelease>(PiDev);

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
  if (is_host())
    return false;

  std::string AllExtensionNames = get_platform_info_string_impl(
      MPlatform, getPlugin(),
      detail::PiInfoCode<info::platform::extensions>::value);
  return (AllExtensionNames.find(ExtensionName) != std::string::npos);
}

bool platform_impl::supports_usm() const {
  return getBackend() != backend::opencl ||
         has_extension("cl_intel_unified_shared_memory");
}

pi_native_handle platform_impl::getNative() const {
  const auto &Plugin = getPlugin();
  pi_native_handle Handle;
  Plugin->call<PiApiKind::piextPlatformGetNativeHandle>(getHandleRef(),
                                                        &Handle);
  return Handle;
}

template <typename Param>
typename Param::return_type platform_impl::get_info() const {
  if (is_host())
    return get_platform_info_host<Param>();

  return get_platform_info<Param>(this->getHandleRef(), getPlugin());
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
platform_impl::getDeviceImplHelper(sycl::detail::pi::PiDevice PiDevice) {
  for (const std::weak_ptr<device_impl> &DeviceWP : MDeviceCache) {
    if (std::shared_ptr<device_impl> Device = DeviceWP.lock()) {
      if (Device->getHandleRef() == PiDevice)
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
