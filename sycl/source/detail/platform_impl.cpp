//==----------- platform_impl.cpp ------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/device.hpp>
#include <detail/config.hpp>
#include <detail/device_impl.hpp>
#include <detail/force_device.hpp>
#include <detail/global_handler.hpp>
#include <detail/platform_impl.hpp>
#include <detail/platform_info.hpp>

#include <algorithm>
#include <cstring>
#include <regex>
#include <string>
#include <vector>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

using PlatformImplPtr = std::shared_ptr<platform_impl>;

PlatformImplPtr platform_impl::getHostPlatformImpl() {
  static PlatformImplPtr HostImpl = std::make_shared<platform_impl>();

  return HostImpl;
}

PlatformImplPtr platform_impl::getOrMakePlatformImpl(RT::PiPlatform PiPlatform,
                                                     const plugin &Plugin) {
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

PlatformImplPtr platform_impl::getPlatformFromPiDevice(RT::PiDevice PiDevice,
                                                       const plugin &Plugin) {
  RT::PiPlatform Plt = nullptr; // TODO catch an exception and put it to list
  // of asynchronous exceptions
  Plugin.call<PiApiKind::piDeviceGetInfo>(PiDevice, PI_DEVICE_INFO_PLATFORM,
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
  auto IsNVIDIAOpenCL = [](platform Platform) {
    if (Platform.is_host())
      return false;

    const bool HasCUDA = Platform.get_info<info::platform::name>().find(
                             "NVIDIA CUDA") != std::string::npos;
    const auto Backend =
        detail::getSyclObjImpl(Platform)->getPlugin().getBackend();
    const bool IsCUDAOCL = (HasCUDA && Backend == backend::opencl);
    if (detail::pi::trace(detail::pi::TraceLevel::PI_TRACE_ALL) && IsCUDAOCL) {
      std::cout << "SYCL_PI_TRACE[all]: "
                << "NVIDIA CUDA OpenCL platform found but is not compatible."
                << std::endl;
    }
    return IsCUDAOCL;
  };
  return IsNVIDIAOpenCL(Platform);
}

vector_class<platform> platform_impl::get_platforms() {
  vector_class<platform> Platforms;
  const vector_class<plugin> &Plugins = RT::initialize();

  info::device_type ForcedType = detail::get_forced_type();
  for (unsigned int i = 0; i < Plugins.size(); i++) {

    pi_uint32 NumPlatforms = 0;
    // Move to the next plugin if the plugin fails to initialize.
    // This way platforms from other plugins get a chance to be discovered.
    if (Plugins[i].call_nocheck<PiApiKind::piPlatformsGet>(
            0, nullptr, &NumPlatforms) != PI_SUCCESS)
      continue;

    if (NumPlatforms) {
      vector_class<RT::PiPlatform> PiPlatforms(NumPlatforms);
      if (Plugins[i].call_nocheck<PiApiKind::piPlatformsGet>(
              NumPlatforms, PiPlatforms.data(), nullptr) != PI_SUCCESS)
        return Platforms;

      for (const auto &PiPlatform : PiPlatforms) {
        platform Platform = detail::createSyclObjFromImpl<platform>(
            getOrMakePlatformImpl(PiPlatform, Plugins[i]));
        // Skip platforms which do not contain requested device types
        if (!Platform.get_devices(ForcedType).empty() &&
            !IsBannedPlatform(Platform))
          Platforms.push_back(Platform);
      }
    }
  }

  // The host platform should always be available unless not allowed by the
  // SYCL_DEVICE_FILTER
  detail::device_filter_list *FilterList =
      detail::SYCLConfig<detail::SYCL_DEVICE_FILTER>::get();
  if (!FilterList || FilterList->backendCompatible(backend::host))
    Platforms.emplace_back(platform());

  return Platforms;
}

constexpr char BackendNameKeyName[] = "BackendName";
constexpr char DeviceTypeKeyName[] = "DeviceType";
constexpr char DeviceVendorIdKeyName[] = "DeviceVendorId";
constexpr char DriverVersionKeyName[] = "DriverVersion";
constexpr char PlatformVersionKeyName[] = "PlatformVersion";
constexpr char DeviceNameKeyName[] = "DeviceName";
constexpr char PlatformNameKeyName[] = "PlatformName";

// change to constexpr std::vector after switching DPC++ RT to C++20
const std::vector<std::string> SupportedAllowListKeyNames{
    BackendNameKeyName,   DeviceTypeKeyName,      DeviceVendorIdKeyName,
    DriverVersionKeyName, PlatformVersionKeyName, DeviceNameKeyName,
    PlatformNameKeyName};

// Parsing and validating SYCL_DEVICE_ALLOWLIST variable value.
//
// The value has the following form:
// DeviceDesc1|DeviceDesc2|<...>|DeviceDescN
// DeviceDescN is the set of descriptions for the device which should be
// allowed. The sets of device descriptions are separated by '|' symbol. The set
// of descriptions has the following structure:
// DeviceDescN = Key1:Value1,Key2:Value2,...,KeyN:ValueN
// Device descriptions are separated by ',' symbol.
// Key and value of a device description are separated by ":" symbol.
// KeyN is the key of a device description, it could be one of the following
// from SupportedAllowListKeyNames vector above.
// DeviceName and PlatformName device descriptions are deprecated and will be
// removed in one of the future releases.
// ValueN is the value of a device description, it could be regex and some fixed
// string.
// Function should return parsed SYCL_DEVICE_ALLOWLIST variable value as
// AllowListParsedT type (vector of maps), e.g.:
// {{Key1: Value1, Key2: Value2}, ..., {Key1: Value1, ..., KeyN: ValueN}}
AllowListParsedT parseAllowList(const std::string &AllowListRaw) {
  if (AllowListRaw.empty())
    return {};

  AllowListParsedT AllowListParsed;
  AllowListParsed.emplace_back();

  const std::vector<std::string> SupportedKeyNamesHaveFixedValue{
      BackendNameKeyName, DeviceTypeKeyName, DeviceVendorIdKeyName};
  const std::vector<std::string> SupportedKeyNamesRequireRegexValue{
      DriverVersionKeyName, PlatformVersionKeyName, DeviceNameKeyName,
      PlatformNameKeyName};

  size_t KeyStart = 0, KeyEnd = 0, ValueStart = 0, ValueEnd = 0,
         DeviceDescIndex = 0;

  while ((KeyEnd = AllowListRaw.find(':', KeyStart)) != std::string::npos) {
    if ((ValueStart = AllowListRaw.find_first_not_of(":", KeyEnd)) ==
        std::string::npos)
      break;
    const std::string &Key = AllowListRaw.substr(KeyStart, KeyEnd - KeyStart);

    // check that provided key is supported
    if (std::find(SupportedAllowListKeyNames.begin(),
                  SupportedAllowListKeyNames.end(),
                  Key) == SupportedAllowListKeyNames.end()) {
      throw sycl::runtime_error("Unrecognized key in SYCL_DEVICE_ALLOWLIST",
                                PI_INVALID_VALUE);
    }

    bool ShouldAllocateNewDeviceDescMap = false;

    ValueEnd = AllowListRaw.find(',', ValueStart);
    if (ValueEnd == std::string::npos) {
      ValueEnd = AllowListRaw.length();
    }
    for (const auto &SupportedKeyName : SupportedAllowListKeyNames) {
      // check if it is the last Key:Value pair in the device description, and
      // correct end position of that value
      if (size_t ValueEndCand =
              AllowListRaw.find("|" + SupportedKeyName, ValueStart);
          (ValueEndCand != std::string::npos) && (ValueEndCand < ValueEnd)) {
        ValueEnd = ValueEndCand;
        ShouldAllocateNewDeviceDescMap = true;
      }
    }
    auto &DeviceDescMap = AllowListParsed[DeviceDescIndex];

    // check if Key is not already defined in DeviceDescMap, e.g., caused by the
    // following invalid syntax: Key1:Value1,Key2:Value2,Key1:Value3
    if (DeviceDescMap.find(Key) == DeviceDescMap.end()) {
      // check that regex values have double curly braces at the beginning and
      // at the end
      size_t CurlyBracesStartSize = 0, CurlyBracesEndSize = 0;
      if (std::find(SupportedKeyNamesRequireRegexValue.begin(),
                    SupportedKeyNamesRequireRegexValue.end(),
                    Key) != SupportedKeyNamesRequireRegexValue.end()) {
        const std::string &ValueRaw =
            AllowListRaw.substr(ValueStart, ValueEnd - ValueStart);
        std::string Prefix("{{");
        // can be changed to string_view::starts_with after switching DPC++ RT
        // to C++20
        if (Prefix != ValueRaw.substr(0, Prefix.length())) {
          throw sycl::runtime_error("Key " + Key +
                                        " of SYCL_DEVICE_ALLOWLIST should have "
                                        "value which starts with {{",
                                    PI_INVALID_VALUE);
        }
        std::string Postfix("}}");
        // can be changed to string_view::ends_with after switching DPC++ RT to
        // C++20
        if (Postfix != ValueRaw.substr(ValueRaw.length() - Postfix.length(),
                                       ValueRaw.length())) {
          throw sycl::runtime_error("Key " + Key +
                                        " of SYCL_DEVICE_ALLOWLIST should have "
                                        "value which ends with }}",
                                    PI_INVALID_VALUE);
        }
        CurlyBracesStartSize = Prefix.length();
        CurlyBracesEndSize = Postfix.length();
      }
      // if value has curly braces {{ and }} at the beginning and at the end,
      // CurlyBracesStartSize and CurlyBracesEndSize != 0, so we move boundaries
      // to remove these braces
      const std::string &Value =
          AllowListRaw.substr(ValueStart + CurlyBracesStartSize,
                              (ValueEnd - CurlyBracesEndSize) -
                                  (ValueStart + CurlyBracesStartSize));
      // check that values of keys, which should have some fixed format, are
      // valid. E.g., for BackendName key, the allowed values are only ones
      // described in SyclBeMap
      if (std::find(SupportedKeyNamesHaveFixedValue.begin(),
                    SupportedKeyNamesHaveFixedValue.end(),
                    Key) != SupportedKeyNamesHaveFixedValue.end()) {
        if (Key == BackendNameKeyName) {
          bool ValueForBackendNameIsValid = false;
          for (const auto &SyclBe : SyclBeMap) {
            if (Value == SyclBe.first) {
              ValueForBackendNameIsValid = true;
              break;
            }
          }
          if (!ValueForBackendNameIsValid) {
            throw sycl::runtime_error(
                "Value " + Value + " for key " + Key +
                    " is not valid in "
                    "SYCL_DEVICE_ALLOWLIST. For details, please refer to "
                    "https://github.com/intel/llvm/blob/sycl/sycl/doc/"
                    "EnvironmentVariables.md",
                PI_INVALID_VALUE);
          }
        }
        if (Key == DeviceTypeKeyName) {
          bool ValueForDeviceTypeIsValid = false;
          for (const auto &SyclDeviceType : SyclDeviceTypeMap) {
            if (Value == SyclDeviceType.first) {
              ValueForDeviceTypeIsValid = true;
              break;
            }
          }
          if (!ValueForDeviceTypeIsValid) {
            throw sycl::runtime_error(
                "Value " + Value + " for key " + Key +
                    " is not valid in "
                    "SYCL_DEVICE_ALLOWLIST. For details, please refer to "
                    "https://github.com/intel/llvm/blob/sycl/sycl/doc/"
                    "EnvironmentVariables.md",
                PI_INVALID_VALUE);
          }
        }
        if (Key == DeviceVendorIdKeyName) {
          // DeviceVendorId should have hex format
          if (!std::regex_match(Value, std::regex("0[xX][0-9a-fA-F]+"))) {
            throw sycl::runtime_error(
                "Value " + Value + " for key " + Key +
                    " is not valid in "
                    "SYCL_DEVICE_ALLOWLIST. It should have hex format. For "
                    "details, please refer to "
                    "https://github.com/intel/llvm/blob/sycl/sycl/doc/"
                    "EnvironmentVariables.md",
                PI_INVALID_VALUE);
          }
        }
      }

      // add key and value to the map
      DeviceDescMap.emplace(Key, Value);
    } else {
      throw sycl::runtime_error("Re-definition of key " + Key +
                                    " is not allowed in "
                                    "SYCL_DEVICE_ALLOWLIST",
                                PI_INVALID_VALUE);
    }

    KeyStart = ValueEnd;
    if (KeyStart != std::string::npos)
      ++KeyStart;
    if (ShouldAllocateNewDeviceDescMap) {
      ++DeviceDescIndex;
      AllowListParsed.emplace_back();
    }
  }

  return AllowListParsed;
}

// Checking if we can allow device with device description DeviceDesc
bool DeviceIsAllowed(const DeviceDescT &DeviceDesc,
                     const AllowListParsedT &AllowListParsed) {
  for (const auto &SupportedKeyName : SupportedAllowListKeyNames)
    assert((DeviceDesc.find(SupportedKeyName) != DeviceDesc.end()) &&
           "DeviceDesc map should have all supported keys for "
           "SYCL_DEVICE_ALLOWLIST.");
  auto EqualityComp = [&](const std::string &KeyName,
                          const DeviceDescT &AllowListDeviceDesc) {
    // change to map::contains after switching DPC++ RT to C++20
    if (AllowListDeviceDesc.find(KeyName) != AllowListDeviceDesc.end())
      if (AllowListDeviceDesc.at(KeyName) != DeviceDesc.at(KeyName))
        return false;
    return true;
  };
  auto RegexComp = [&](const std::string &KeyName,
                       const DeviceDescT &AllowListDeviceDesc) {
    if (AllowListDeviceDesc.find(KeyName) != AllowListDeviceDesc.end())
      if (!std::regex_match(DeviceDesc.at(KeyName),
                            std::regex(AllowListDeviceDesc.at(KeyName))))
        return false;
    return true;
  };

  bool ShouldDeviceBeAllowed = false;

  for (const auto &AllowListDeviceDesc : AllowListParsed) {
    if (!EqualityComp(BackendNameKeyName, AllowListDeviceDesc))
      continue;
    if (!EqualityComp(DeviceTypeKeyName, AllowListDeviceDesc))
      continue;
    if (!EqualityComp(DeviceVendorIdKeyName, AllowListDeviceDesc))
      continue;
    if (!RegexComp(DriverVersionKeyName, AllowListDeviceDesc))
      continue;
    if (!RegexComp(PlatformVersionKeyName, AllowListDeviceDesc))
      continue;
    if (!RegexComp(DeviceNameKeyName, AllowListDeviceDesc))
      continue;
    if (!RegexComp(PlatformNameKeyName, AllowListDeviceDesc))
      continue;

    // no any continue was called on this iteration, so all parameters matched
    // successfully, so allow this device to use
    ShouldDeviceBeAllowed = true;
    break;
  }

  return ShouldDeviceBeAllowed;
}

static void applyAllowList(std::vector<RT::PiDevice> &PiDevices,
                           RT::PiPlatform PiPlatform, const plugin &Plugin) {
  AllowListParsedT AllowListParsed =
      parseAllowList(SYCLConfig<SYCL_DEVICE_ALLOWLIST>::get());
  if (AllowListParsed.empty())
    return;

  DeviceDescT DeviceDesc;

  // get BackendName value and put it to DeviceDesc
  sycl::backend Backend = Plugin.getBackend();
  for (const auto &SyclBe : SyclBeMap) {
    if (SyclBe.second == Backend) {
      DeviceDesc.emplace(BackendNameKeyName, SyclBe.first);
    }
  }
  // get PlatformVersion value and put it to DeviceDesc
  DeviceDesc.emplace(
      PlatformVersionKeyName,
      sycl::detail::get_platform_info<std::string,
                                      info::platform::version>::get(PiPlatform,
                                                                    Plugin));
  // get PlatformName value and put it to DeviceDesc
  DeviceDesc.emplace(
      PlatformNameKeyName,
      sycl::detail::get_platform_info<std::string, info::platform::name>::get(
          PiPlatform, Plugin));

  int InsertIDx = 0;
  for (RT::PiDevice Device : PiDevices) {
    bool IsInserted = false;
    // get DeviceType value and put it to DeviceDesc
    RT::PiDeviceType PiDevType;
    Plugin.call<PiApiKind::piDeviceGetInfo>(Device, PI_DEVICE_INFO_TYPE,
                                            sizeof(RT::PiDeviceType),
                                            &PiDevType, nullptr);
    sycl::info::device_type DeviceType = pi::cast<info::device_type>(PiDevType);
    for (const auto &SyclDeviceType : SyclDeviceTypeMap) {
      if (SyclDeviceType.second == DeviceType) {
        const auto &DeviceTypeValue = SyclDeviceType.first;
        std::tie(std::ignore, IsInserted) =
            DeviceDesc.emplace(DeviceTypeKeyName, DeviceTypeValue);
        if (!IsInserted)
          DeviceDesc.at(DeviceTypeKeyName) = DeviceTypeValue;
        break;
      }
    }
    // get DeviceVendorId value and put it to DeviceDesc
    uint32_t DeviceVendorIdUInt =
        sycl::detail::get_device_info<uint32_t, info::device::vendor_id>::get(
            Device, Plugin);
    std::stringstream DeviceVendorIdHexStringStream;
    DeviceVendorIdHexStringStream << "0x" << std::hex << DeviceVendorIdUInt;
    const auto &DeviceVendorIdValue = DeviceVendorIdHexStringStream.str();
    std::tie(std::ignore, IsInserted) = DeviceDesc.emplace(
        DeviceVendorIdKeyName, DeviceVendorIdHexStringStream.str());
    if (!IsInserted)
      DeviceDesc.at(DeviceVendorIdKeyName) = DeviceVendorIdValue;
    // get DriverVersion value and put it to DeviceDesc
    const auto &DriverVersionValue = sycl::detail::get_device_info<
        std::string, info::device::driver_version>::get(Device, Plugin);
    std::tie(std::ignore, IsInserted) =
        DeviceDesc.emplace(DriverVersionKeyName, DriverVersionValue);
    if (!IsInserted)
      DeviceDesc.at(DriverVersionKeyName) = DriverVersionValue;
    // get DeviceName value and put it to DeviceDesc
    const auto &DeviceNameValue =
        sycl::detail::get_device_info<std::string, info::device::name>::get(
            Device, Plugin);
    std::tie(std::ignore, IsInserted) =
        DeviceDesc.emplace(DeviceNameKeyName, DeviceNameValue);
    if (!IsInserted)
      DeviceDesc.at(DeviceNameKeyName) = DeviceNameValue;

    // check if we can allow device with such device description DeviceDesc
    if (DeviceIsAllowed(DeviceDesc, AllowListParsed)) {
      PiDevices[InsertIDx++] = Device;
    }
  }
  PiDevices.resize(InsertIDx);
}

// Filter out the devices that are not compatible with SYCL_DEVICE_FILTER.
// All three entries (backend:device_type:device_num) are optional.
// The missing entries are constructed using '*', which means 'any' | 'all'
// by the device_filter constructor.
// This function matches devices in the order of backend, device_type, and
// device_num.
static void filterDeviceFilter(vector_class<RT::PiDevice> &PiDevices,
                               const plugin &Plugin) {
  device_filter_list *FilterList = SYCLConfig<SYCL_DEVICE_FILTER>::get();
  if (!FilterList)
    return;

  backend Backend = Plugin.getBackend();
  int InsertIDx = 0;
  int DeviceNum = 0;
  for (RT::PiDevice Device : PiDevices) {
    RT::PiDeviceType PiDevType;
    Plugin.call<PiApiKind::piDeviceGetInfo>(Device, PI_DEVICE_INFO_TYPE,
                                            sizeof(RT::PiDeviceType),
                                            &PiDevType, nullptr);
    // Assumption here is that there is 1-to-1 mapping between PiDevType and
    // Sycl device type for GPU, CPU, and ACC.
    info::device_type DeviceType = pi::cast<info::device_type>(PiDevType);

    for (const device_filter &Filter : FilterList->get()) {
      backend FilterBackend = Filter.Backend;
      // First, match the backend entry
      if (FilterBackend == Backend || FilterBackend == backend::all) {
        info::device_type FilterDevType = Filter.DeviceType;
        // Next, match the device_type entry
        if (FilterDevType == info::device_type::all) {
          // Last, match the device_num entry
          if (!Filter.HasDeviceNum || DeviceNum == Filter.DeviceNum) {
            PiDevices[InsertIDx++] = Device;
            break;
          }
        } else if (FilterDevType == DeviceType) {
          if (!Filter.HasDeviceNum || DeviceNum == Filter.DeviceNum) {
            PiDevices[InsertIDx++] = Device;
            break;
          }
        }
      }
    }
    DeviceNum++;
  }
  PiDevices.resize(InsertIDx);
}

std::shared_ptr<device_impl> platform_impl::getOrMakeDeviceImpl(
    RT::PiDevice PiDevice, const std::shared_ptr<platform_impl> &PlatformImpl) {
  const std::lock_guard<std::mutex> Guard(MDeviceMapMutex);

  // If we've already seen this device, return the impl
  for (const std::weak_ptr<device_impl> &DeviceWP : MDeviceCache) {
    if (std::shared_ptr<device_impl> Device = DeviceWP.lock()) {
      if (Device->getHandleRef() == PiDevice)
        return Device;
    }
  }

  // Otherwise make the impl
  std::shared_ptr<device_impl> Result =
      std::make_shared<device_impl>(PiDevice, PlatformImpl);
  MDeviceCache.emplace_back(Result);

  return Result;
}

vector_class<device>
platform_impl::get_devices(info::device_type DeviceType) const {
  vector_class<device> Res;
  if (is_host() && (DeviceType == info::device_type::host ||
                    DeviceType == info::device_type::all)) {
    // If SYCL_DEVICE_FILTER is set, check if filter contains host.
    device_filter_list *FilterList = SYCLConfig<SYCL_DEVICE_FILTER>::get();
    if (!FilterList || FilterList->containsHost()) {
      Res.push_back(device());
    }
  }

  // If any DeviceType other than host was requested for host platform,
  // an empty vector will be returned.
  if (is_host() || DeviceType == info::device_type::host)
    return Res;

  pi_uint32 NumDevices = 0;
  const detail::plugin &Plugin = getPlugin();
  Plugin.call<PiApiKind::piDevicesGet>(
      MPlatform, pi::cast<RT::PiDeviceType>(DeviceType), 0,
      pi::cast<RT::PiDevice *>(nullptr), &NumDevices);

  if (NumDevices == 0)
    return Res;

  vector_class<RT::PiDevice> PiDevices(NumDevices);
  // TODO catch an exception and put it to list of asynchronous exceptions
  Plugin.call<PiApiKind::piDevicesGet>(MPlatform,
                                       pi::cast<RT::PiDeviceType>(DeviceType),
                                       NumDevices, PiDevices.data(), nullptr);

  // Filter out devices that are not present in the SYCL_DEVICE_ALLOWLIST
  if (SYCLConfig<SYCL_DEVICE_ALLOWLIST>::get())
    applyAllowList(PiDevices, MPlatform, this->getPlugin());

  // Filter out devices that are not compatible with SYCL_DEVICE_FILTER
  filterDeviceFilter(PiDevices, Plugin);

  PlatformImplPtr PlatformImpl = getOrMakePlatformImpl(MPlatform, *MPlugin);
  std::transform(
      PiDevices.begin(), PiDevices.end(), std::back_inserter(Res),
      [PlatformImpl](const RT::PiDevice &PiDevice) -> device {
        return detail::createSyclObjFromImpl<device>(
            PlatformImpl->getOrMakeDeviceImpl(PiDevice, PlatformImpl));
      });

  return Res;
}

bool platform_impl::has_extension(const string_class &ExtensionName) const {
  if (is_host())
    return false;

  string_class AllExtensionNames =
      get_platform_info<string_class, info::platform::extensions>::get(
          MPlatform, getPlugin());
  return (AllExtensionNames.find(ExtensionName) != std::string::npos);
}

pi_native_handle platform_impl::getNative() const {
  const auto &Plugin = getPlugin();
  pi_native_handle Handle;
  Plugin.call<PiApiKind::piextPlatformGetNativeHandle>(getHandleRef(), &Handle);
  return Handle;
}

template <info::platform param>
typename info::param_traits<info::platform, param>::return_type
platform_impl::get_info() const {
  if (is_host())
    return get_platform_info_host<param>();

  return get_platform_info<
      typename info::param_traits<info::platform, param>::return_type,
      param>::get(this->getHandleRef(), getPlugin());
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

#define __SYCL_PARAM_TRAITS_SPEC(param_type, param, ret_type)                  \
  template ret_type platform_impl::get_info<info::param_type::param>() const;

#include <CL/sycl/info/platform_traits.def>
#undef __SYCL_PARAM_TRAITS_SPEC

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
