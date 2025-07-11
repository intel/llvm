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
#include <sycl/backend_types.hpp>

#include <algorithm>
#include <regex>
#include <sstream>

namespace sycl {
inline namespace _V1 {
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

  const char DelimiterBtwKeyAndValue = ':';
  const char DelimiterBtwItemsInDeviceDesc = ',';
  const char DelimiterBtwDeviceDescs = '|';

  if (AllowListRaw.find(DelimiterBtwKeyAndValue, KeyStart) == std::string::npos)
    throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                          "SYCL_DEVICE_ALLOWLIST has incorrect format. For "
                          "details, please refer to "
                          "https://github.com/intel/llvm/blob/sycl/sycl/"
                          "doc/EnvironmentVariables.md " +
                              codeToString(UR_RESULT_ERROR_INVALID_VALUE));

  const std::string &DeprecatedKeyNameDeviceName = DeviceNameKeyName;
  const std::string &DeprecatedKeyNamePlatformName = PlatformNameKeyName;

  bool IsDeprecatedKeyNameDeviceNameWasUsed = false;
  bool IsDeprecatedKeyNamePlatformNameWasUsed = false;

  while ((KeyEnd = AllowListRaw.find(DelimiterBtwKeyAndValue, KeyStart)) !=
         std::string::npos) {
    if ((ValueStart = AllowListRaw.find_first_not_of(
             DelimiterBtwKeyAndValue, KeyEnd)) == std::string::npos)
      break;
    const std::string &Key = AllowListRaw.substr(KeyStart, KeyEnd - KeyStart);

    // check that provided key is supported
    if (std::find(SupportedAllowListKeyNames.begin(),
                  SupportedAllowListKeyNames.end(),
                  Key) == SupportedAllowListKeyNames.end()) {
      throw sycl::exception(
          sycl::make_error_code(sycl::errc::runtime),
          "Unrecognized key in SYCL_DEVICE_ALLOWLIST. For details, please "
          "refer to "
          "https://github.com/intel/llvm/blob/sycl/sycl/doc/"
          "EnvironmentVariables.md " +
              codeToString(UR_RESULT_ERROR_INVALID_VALUE));
    }

    if (Key == DeprecatedKeyNameDeviceName) {
      IsDeprecatedKeyNameDeviceNameWasUsed = true;
    }
    if (Key == DeprecatedKeyNamePlatformName) {
      IsDeprecatedKeyNamePlatformNameWasUsed = true;
    }

    bool ShouldAllocateNewDeviceDescMap = false;

    std::string Value;

    auto &DeviceDescMap = AllowListParsed[DeviceDescIndex];

    // check if Key is not already defined in DeviceDescMap, e.g., caused by the
    // following invalid syntax: Key1:Value1,Key2:Value2,Key1:Value3
    if (DeviceDescMap.find(Key) == DeviceDescMap.end()) {
      // calculate and validate value which has fixed format
      if (std::find(SupportedKeyNamesHaveFixedValue.begin(),
                    SupportedKeyNamesHaveFixedValue.end(),
                    Key) != SupportedKeyNamesHaveFixedValue.end()) {
        ValueEnd = AllowListRaw.find(DelimiterBtwItemsInDeviceDesc, ValueStart);
        // check if it is the last Key:Value pair in the device description, and
        // correct end position of that value
        if (size_t ValueEndCand =
                AllowListRaw.find(DelimiterBtwDeviceDescs, ValueStart);
            (ValueEndCand != std::string::npos) && (ValueEndCand < ValueEnd)) {
          ValueEnd = ValueEndCand;
          ShouldAllocateNewDeviceDescMap = true;
        }
        if (ValueEnd == std::string::npos)
          ValueEnd = AllowListRaw.length();

        Value = AllowListRaw.substr(ValueStart, ValueEnd - ValueStart);

        // post-processing checks for some values

        auto ValidateEnumValues = [&](std::string CheckingKeyName,
                                      auto SourceOfSupportedValues) {
          if (Key == CheckingKeyName) {
            bool ValueIsValid = false;
            for (const auto &Item : SourceOfSupportedValues)
              if (Value == Item.first) {
                ValueIsValid = true;
                break;
              }
            if (!ValueIsValid)
              throw sycl::exception(
                  sycl::make_error_code(sycl::errc::runtime),
                  "Value " + Value + " for key " + Key +
                      " is not valid in "
                      "SYCL_DEVICE_ALLOWLIST. For details, please refer to "
                      "https://github.com/intel/llvm/blob/sycl/sycl/doc/"
                      "EnvironmentVariables.md " +
                      codeToString(UR_RESULT_ERROR_INVALID_VALUE));
          }
        };

        // check that values of keys, which should have some fixed format, are
        // valid. E.g., for BackendName key, the allowed values are only ones
        // described in SyclBeMap
        ValidateEnumValues(BackendNameKeyName, getSyclBeMap());
        ValidateEnumValues(DeviceTypeKeyName,
                           getSyclDeviceTypeMap<true /*Enable 'acc'*/>());

        if (Key == DeviceVendorIdKeyName) {
          // DeviceVendorId should have hex format
          if (!std::regex_match(Value, std::regex("0[xX][0-9a-fA-F]+"))) {
            throw sycl::exception(
                sycl::make_error_code(sycl::errc::runtime),
                "Value " + Value + " for key " + Key +
                    " is not valid in "
                    "SYCL_DEVICE_ALLOWLIST. It should have the hex format. For "
                    "details, please refer to "
                    "https://github.com/intel/llvm/blob/sycl/sycl/doc/"
                    "EnvironmentVariables.md " +
                    codeToString(UR_RESULT_ERROR_INVALID_VALUE));
          }
        }
      }
      // calculate and validate value which has regex format
      else if (std::find(SupportedKeyNamesRequireRegexValue.begin(),
                         SupportedKeyNamesRequireRegexValue.end(),
                         Key) != SupportedKeyNamesRequireRegexValue.end()) {
        const std::string Prefix("{{");
        // TODO: can be changed to string_view::starts_with after switching
        // DPC++ RT to C++20
        if (Prefix != AllowListRaw.substr(ValueStart, Prefix.length())) {
          throw sycl::exception(
              sycl::make_error_code(sycl::errc::runtime),
              "Key " + Key +
                  " of SYCL_DEVICE_ALLOWLIST should have "
                  "value which starts with " +
                  Prefix + " " +
                  detail::codeToString(UR_RESULT_ERROR_INVALID_VALUE));
        }
        // cut off prefix from the value
        ValueStart += Prefix.length();

        ValueEnd = ValueStart;
        const std::string Postfix("}}");
        for (; ValueEnd < AllowListRaw.length() - Postfix.length() + 1;
             ++ValueEnd) {
          if (Postfix == AllowListRaw.substr(ValueEnd, Postfix.length()))
            break;
          // if it is the last iteration and next 2 symbols are not a postfix,
          // throw exception
          if (ValueEnd == AllowListRaw.length() - Postfix.length())
            throw sycl::exception(
                sycl::make_error_code(sycl::errc::runtime),
                "Key " + Key +
                    " of SYCL_DEVICE_ALLOWLIST should have "
                    "value which ends with " +
                    Postfix + " " +
                    detail::codeToString(UR_RESULT_ERROR_INVALID_VALUE));
        }
        size_t NextExpectedDelimiterPos = ValueEnd + Postfix.length();
        // if it is not the end of the string, check that symbol next to a
        // postfix is a delimiter (, or ;)
        if ((AllowListRaw.length() != NextExpectedDelimiterPos) &&
            (AllowListRaw[NextExpectedDelimiterPos] !=
             DelimiterBtwItemsInDeviceDesc) &&
            (AllowListRaw[NextExpectedDelimiterPos] != DelimiterBtwDeviceDescs))
          throw sycl::exception(
              sycl::make_error_code(sycl::errc::runtime),
              "Unexpected symbol on position " +
                  std::to_string(NextExpectedDelimiterPos) + ": " +
                  AllowListRaw[NextExpectedDelimiterPos] +
                  ". Should be either " + DelimiterBtwItemsInDeviceDesc +
                  " or " + DelimiterBtwDeviceDescs +
                  codeToString(UR_RESULT_ERROR_INVALID_VALUE));

        if (AllowListRaw[NextExpectedDelimiterPos] == DelimiterBtwDeviceDescs)
          ShouldAllocateNewDeviceDescMap = true;

        Value = AllowListRaw.substr(ValueStart, ValueEnd - ValueStart);

        ValueEnd += Postfix.length();
      } else
        assert(false &&
               "Key should be either in SupportedKeyNamesHaveFixedValue "
               "or SupportedKeyNamesRequireRegexValue");

      // add key and value to the map
      DeviceDescMap.emplace(Key, Value);
    } else
      throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                            "Re-definition of key " + Key +
                                " is not allowed in "
                                "SYCL_DEVICE_ALLOWLIST " +
                                codeToString(UR_RESULT_ERROR_INVALID_VALUE));

    KeyStart = ValueEnd;
    if (KeyStart != std::string::npos)
      ++KeyStart;
    if (ShouldAllocateNewDeviceDescMap) {
      ++DeviceDescIndex;
      AllowListParsed.emplace_back();
    }
  }

  if (IsDeprecatedKeyNameDeviceNameWasUsed &&
      IsDeprecatedKeyNamePlatformNameWasUsed) {
    std::cout << "\nWARNING: " << DeprecatedKeyNameDeviceName << " and "
              << DeprecatedKeyNamePlatformName
              << " in SYCL_DEVICE_ALLOWLIST are deprecated. ";
  } else if (IsDeprecatedKeyNameDeviceNameWasUsed) {
    std::cout << "\nWARNING: " << DeprecatedKeyNameDeviceName
              << " in SYCL_DEVICE_ALLOWLIST is deprecated. ";
  } else if (IsDeprecatedKeyNamePlatformNameWasUsed) {
    std::cout << "\nWARNING: " << DeprecatedKeyNamePlatformName
              << " in SYCL_DEVICE_ALLOWLIST is deprecated. ";
  }
  if (IsDeprecatedKeyNameDeviceNameWasUsed ||
      IsDeprecatedKeyNamePlatformNameWasUsed) {
    std::cout << "Please use " << BackendNameKeyName << ", "
              << DeviceTypeKeyName << " and " << DeviceVendorIdKeyName
              << " instead. For details, please refer to "
                 "https://github.com/intel/llvm/blob/sycl/sycl/doc/"
                 "EnvironmentVariables.md\n\n";
  }

  return AllowListParsed;
}

static void traceAllowFiltering(const DeviceDescT &DeviceDesc, bool Allowed) {
  bool shouldTrace = false;
  if (Allowed) {
    shouldTrace = detail::ur::trace(detail::ur::TraceLevel::TRACE_BASIC);
  } else {
    shouldTrace = detail::ur::trace(detail::ur::TraceLevel::TRACE_ALL);
  }

  if (shouldTrace) {
    auto selectionMsg = Allowed ? "allowed" : "filtered";
    std::cout << "SYCL_UR_TRACE: Device " << selectionMsg
              << " by SYCL_DEVICE_ALLOWLIST" << std::endl
              << "SYCL_UR_TRACE: "
              << "  platform: " << DeviceDesc.at(PlatformNameKeyName)
              << std::endl
              << "SYCL_UR_TRACE: "
              << "  device: " << DeviceDesc.at(DeviceNameKeyName) << std::endl;
  }
}

// Checking if we can allow device with device description DeviceDesc
bool deviceIsAllowed(const DeviceDescT &DeviceDesc,
                     const AllowListParsedT &AllowListParsed) {
  assert(std::all_of(SupportedAllowListKeyNames.begin(),
                     SupportedAllowListKeyNames.end(),
                     [&DeviceDesc](const auto &SupportedKeyName) {
                       return DeviceDesc.find(SupportedKeyName) !=
                              DeviceDesc.end();
                     }) &&
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

void applyAllowList(std::vector<ur_device_handle_t> &UrDevices,
                    ur_platform_handle_t UrPlatform, adapter_impl &Adapter) {

  AllowListParsedT AllowListParsed =
      parseAllowList(SYCLConfig<SYCL_DEVICE_ALLOWLIST>::get());
  if (AllowListParsed.empty())
    return;

  // Get platform's backend and put it to DeviceDesc
  DeviceDescT DeviceDesc;
  platform_impl &PlatformImpl =
      platform_impl::getOrMakePlatformImpl(UrPlatform, Adapter);
  backend Backend = PlatformImpl.getBackend();

  for (const auto &SyclBe : getSyclBeMap()) {
    if (SyclBe.second == Backend) {
      DeviceDesc.emplace(BackendNameKeyName, SyclBe.first);
      break;
    }
  }
  // get PlatformVersion value and put it to DeviceDesc
  DeviceDesc.emplace(PlatformVersionKeyName,
                     PlatformImpl.get_info<info::platform::version>());
  // get PlatformName value and put it to DeviceDesc
  DeviceDesc.emplace(PlatformNameKeyName,
                     PlatformImpl.get_info<info::platform::name>());

  int InsertIDx = 0;
  for (ur_device_handle_t Device : UrDevices) {
    device_impl &DeviceImpl = PlatformImpl.getOrMakeDeviceImpl(Device);
    // get DeviceType value and put it to DeviceDesc
    ur_device_type_t UrDevType = UR_DEVICE_TYPE_ALL;
    Adapter.call<UrApiKind::urDeviceGetInfo>(
        Device, UR_DEVICE_INFO_TYPE, sizeof(UrDevType), &UrDevType, nullptr);
    // TODO need mechanism to do these casts, there's a bunch of this sort of
    // thing
    sycl::info::device_type DeviceType = info::device_type::all;
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
    for (const auto &SyclDeviceType :
         getSyclDeviceTypeMap<true /*Enable 'acc'*/>()) {
      if (SyclDeviceType.second == DeviceType) {
        const auto &DeviceTypeValue = SyclDeviceType.first;
        DeviceDesc[DeviceTypeKeyName] = DeviceTypeValue;
        break;
      }
    }
    // get DeviceVendorId value and put it to DeviceDesc
    uint32_t DeviceVendorIdUInt =
        DeviceImpl.get_info<info::device::vendor_id>();
    std::stringstream DeviceVendorIdHexStringStream;
    // To avoid commas or other locale-specific modifications, call imbue().
    DeviceVendorIdHexStringStream.imbue(std::locale::classic());
    DeviceVendorIdHexStringStream << "0x" << std::hex << DeviceVendorIdUInt;
    const auto &DeviceVendorIdValue = DeviceVendorIdHexStringStream.str();
    DeviceDesc[DeviceVendorIdKeyName] = DeviceVendorIdValue;
    // get DriverVersion value and put it to DeviceDesc
    const std::string &DriverVersionValue =
        DeviceImpl.get_info<info::device::driver_version>();
    DeviceDesc[DriverVersionKeyName] = DriverVersionValue;
    // get DeviceName value and put it to DeviceDesc
    const std::string &DeviceNameValue =
        DeviceImpl.get_info<info::device::name>();
    DeviceDesc[DeviceNameKeyName] = DeviceNameValue;

    // check if we can allow device with such device description DeviceDesc
    bool isAllowed = deviceIsAllowed(DeviceDesc, AllowListParsed);
    if (isAllowed) {
      UrDevices[InsertIDx++] = Device;
    }
    traceAllowFiltering(DeviceDesc, isAllowed);
  }
  UrDevices.resize(InsertIDx);
}

} // namespace detail
} // namespace _V1
} // namespace sycl
