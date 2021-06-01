//==-------------- allowlist.cpp - SYCL_DEVICE_ALLOWLIST -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/allowlist.hpp>
#include <detail/config.hpp>
#include <detail/device_impl.hpp>
#include <detail/platform_info.hpp>

#include <regex>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

constexpr char BackendNameKeyName[] = "BackendName";
constexpr char DeviceTypeKeyName[] = "DeviceType";
constexpr char DeviceVendorIdKeyName[] = "DeviceVendorId";
constexpr char DriverVersionKeyName[] = "DriverVersion";
constexpr char PlatformVersionKeyName[] = "PlatformVersion";
constexpr char DeviceNameKeyName[] = "DeviceName";
constexpr char PlatformNameKeyName[] = "PlatformName";

constexpr std::array<const char *, 7> SupportedAllowListKeyNames{
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

  constexpr std::array<const char *, 3> SupportedKeyNamesHaveFixedValue{
      BackendNameKeyName, DeviceTypeKeyName, DeviceVendorIdKeyName};
  constexpr std::array<const char *, 4> SupportedKeyNamesRequireRegexValue{
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
      if (size_t ValueEndCand = AllowListRaw.find(
              "|" + std::string(SupportedKeyName), ValueStart);
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
bool deviceIsAllowed(const DeviceDescT &DeviceDesc,
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

void applyAllowList(std::vector<RT::PiDevice> &PiDevices,
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
    if (deviceIsAllowed(DeviceDesc, AllowListParsed)) {
      PiDevices[InsertIDx++] = Device;
    }
  }
  PiDevices.resize(InsertIDx);
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
