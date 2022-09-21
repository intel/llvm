//==------------------- device_filter.cpp ----------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/config.hpp>
#include <detail/device_impl.hpp>
#include <sycl/detail/device_filter.hpp>
#include <sycl/info/info_desc.hpp>

#include <cstring>
#include <sstream>
#include <string_view>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {

std::vector<std::string_view> tokenize(const std::string_view &Filter,
                                       const std::string &Delim) {
  std::vector<std::string_view> Tokens;
  size_t Pos = 0;
  size_t LastPos = 0;

  while ((Pos = Filter.find(Delim, LastPos)) != std::string::npos) {
    std::string_view Tok(Filter.data() + LastPos, (Pos - LastPos));

    if (!Tok.empty()) {
      Tokens.push_back(Tok);
    }
    // move the search starting index
    LastPos = Pos + 1;
  }

  // Add remainder if any
  if (LastPos < Filter.size()) {
    std::string_view Tok(Filter.data() + LastPos, Filter.size() - LastPos);
    Tokens.push_back(Tok);
  }
  return Tokens;
}

// ---------------------------------------
// ONEAPI_DEVICE_SELECTOR support

static backend Parse_ODS_Backend(const std::string_view &BackendStr,
                                 const std::string_view &FullEntry) {
  // Check if the first entry matches with a known backend type
  auto SyclBeMap =
      getSyclBeMap(); // <-- std::array<std::pair<std::string, backend>>
                      // [{"level_zero", backend::level_zero}, {"*", ::all}, ...
  auto It = std::find_if(
      std::begin(SyclBeMap), std::end(SyclBeMap), [&](auto BePair) {
        return std::string::npos != BackendStr.find(BePair.first);
      });

  if (It == SyclBeMap.end()) {
    // backend is required
    std::stringstream ss;
    ss << "ONEAPI_DEVICE_SELECTOR parsing error. Backend is required but "
          "missing from \""
       << FullEntry << "\"";
    throw sycl::exception(sycl::make_error_code(errc::invalid), ss.str());
  } else {
    return It->second;
  }
}

static void Parse_ODS_Device(ods_target &Target,
                             const std::string_view &DeviceStr) {
  // DeviceStr will be: 'gpu', '*', '0', '0.1', 'gpu.*', '0.*', or 'gpu.2', etc.
  std::vector<std::string_view> DeviceSubPair = tokenize(DeviceStr, ".");
  std::string_view TopDeviceStr = DeviceSubPair[0];

  // Handle explicit device type (e.g. 'gpu').
  auto DeviceTypeMap =
      getSyclDeviceTypeMap(); // <-- std::array<std::pair<std::string,
                              // info::device::type>>
  auto It = std::find_if(
      std::begin(DeviceTypeMap), std::end(DeviceTypeMap), [&](auto DtPair) {
        return std::string::npos != TopDeviceStr.find(DtPair.first);
      });
  if (It != DeviceTypeMap.end()) {
    Target.DeviceType = It->second;
    // Handle wildcard.
    if (TopDeviceStr[0] == '*') {
      Target.HasDeviceWildCard = true;
      Target.DeviceType = {};
    }
  } else { // Only thing left is a number.
    std::string TDS(TopDeviceStr);
    try {
      Target.DeviceNum = std::stoi(TDS);
    } catch (...) {
      std::stringstream ss;
      ss << "error parsing device number: " << TDS;
      throw sycl::exception(sycl::make_error_code(errc::invalid), ss.str());
    }
  }

  if (DeviceSubPair.size() >= 2) {
    // We have a subdevice.
    std::string_view SubDeviceStr = DeviceSubPair[1];
    // SubDeviceStr is wildcard or number.
    if (SubDeviceStr[0] == '*') {
      Target.HasSubDeviceWildCard = true;
    } else {
      std::string SDS(SubDeviceStr);
      try {
        Target.SubDeviceNum = std::stoi(SDS);
      } catch (...) {
        std::stringstream ss;
        ss << "error parsing sub-device index: " << SDS;
        throw sycl::exception(sycl::make_error_code(errc::invalid), ss.str());
      }
    }
  }
  if (DeviceSubPair.size() == 3) {
    // We have a sub-sub-device.
    std::string_view SubSubDeviceStr = DeviceSubPair[2];
    if (SubSubDeviceStr[0] == '*') {
      Target.HasSubSubDeviceWildCard = true;
    } else {
      std::string SSDS(SubSubDeviceStr);
      try {
        Target.SubSubDeviceNum = std::stoi(SSDS);
      } catch (...) {
        std::stringstream ss;
        ss << "error parsing sub-sub-device index: " << SSDS;
        throw sycl::exception(sycl::make_error_code(errc::invalid), ss.str());
      }
    }
  } else if (DeviceSubPair.size() > 3) {
    std::stringstream ss;
    ss << "error parsing " << DeviceStr
       << "  Only two levels of sub-devices supported at this time";
    throw sycl::exception(sycl::make_error_code(errc::invalid), ss.str());
  }
}

std::vector<ods_target>
Parse_ONEAPI_DEVICE_SELECTOR(const std::string &envStr) {
  std::vector<ods_target> Result;
  if (envStr.empty()) {
    ods_target acceptAnything;
    Result.push_back(acceptAnything);
    return Result;
  }

  std::vector<std::string_view> Entries = tokenize(envStr, ";");
  // Each entry: "level_zero:gpu" or "opencl:0.0,0.1" or "opencl:*" but NOT just
  // "opencl".
  for (const auto Entry : Entries) {
    std::vector<std::string_view> Pair = tokenize(Entry, ":");
    backend be = Parse_ODS_Backend(Pair[0], Entry); // Pair[0] is backend.

    if (Pair.size() == 1) {
      std::stringstream ss;
      ss << "Incomplete selector!  Try '" << Pair[0]
         << ":*' if all devices under the backend was original intention.";
      throw sycl::exception(sycl::make_error_code(errc::invalid), ss.str());
    } else if (Pair.size() == 2) {
      std::vector<std::string_view> Targets = tokenize(Pair[1], ",");
      for (auto TargetStr : Targets) {
        ods_target DeviceTarget(be);
        Parse_ODS_Device(DeviceTarget, TargetStr);
        Result.push_back(DeviceTarget);
      }
    } else if (Pair.size() > 2) {
      std::stringstream ss;
      ss << "Error parsing selector string \"" << Entry
         << "\"  Too many colons (:)";
      throw sycl::exception(sycl::make_error_code(errc::invalid), ss.str());
    }
  }

  return Result;
}

std::ostream &operator<<(std::ostream &Out, const ods_target &Target) {
  Out << Target.Backend;
  if (Target.DeviceType) {
    auto DeviceTypeMap = getSyclDeviceTypeMap();
    auto Match = std::find_if(
        DeviceTypeMap.begin(), DeviceTypeMap.end(),
        [&](auto Pair) { return (Pair.second == Target.DeviceType); });
    if (Match != DeviceTypeMap.end()) {
      Out << ":" << Match->first;
    } else {
      Out << ":???";
    }
  }
  if (Target.HasDeviceWildCard)
    Out << ":*";
  if (Target.DeviceNum)
    Out << ":" << Target.DeviceNum.value();
  if (Target.HasSubDeviceWildCard)
    Out << ".*";
  if (Target.SubDeviceNum)
    Out << "." << Target.SubDeviceNum.value();

  return Out;
}

ods_target_list::ods_target_list(const std::string &envStr) {
  TargetList = Parse_ONEAPI_DEVICE_SELECTOR(envStr);
}

bool ods_target_list::containsHost() {
  for (const ods_target &Target : TargetList) {
    if (Target.Backend == backend::host || Target.Backend == backend::all)
      if (Target.DeviceType == info::device_type::host ||
          Target.DeviceType == info::device_type::all)
        // SYCL RT never creates more than one HOST device.
        // All device numbers other than 0 are rejected.
        if (!Target.DeviceNum || Target.DeviceNum.value() == 0)
          return true;
  }
  return false;
}

// Backend is compatible with the SYCL_DEVICE_FILTER in the following cases.
// 1. Filter backend is '*' which means ANY backend.
// 2. Filter backend match exactly with the given 'Backend'
bool ods_target_list::backendCompatible(backend Backend) {
  for (const ods_target &Target : TargetList) {
    backend TargetBackend = Target.Backend.value_or(backend::all);
    if (TargetBackend == Backend || TargetBackend == backend::all)
      return true;
  }
  return false;
}

// ---------------------------------------
// SYCL_DEVICE_FILTER support

device_filter::device_filter(const std::string &FilterString) {
  std::vector<std::string_view> Tokens = tokenize(FilterString, ":");
  size_t TripleValueID = 0;

  auto FindElement = [&](auto Element) {
    return std::string::npos != Tokens[TripleValueID].find(Element.first);
  };

  // Handle the optional 1st field of the filter, backend
  // Check if the first entry matches with a known backend type
  auto It = std::find_if(std::begin(getSyclBeMap()), std::end(getSyclBeMap()),
                         FindElement);
  // If no match is found, set the backend type backend::all
  // which actually means 'any backend' will be a match.
  if (It == getSyclBeMap().end())
    Backend = backend::all;
  else {
    Backend = It->second;
    TripleValueID++;
  }

  // Handle the optional 2nd field of the filter - device type.
  // Check if the 2nd entry matches with any known device type.
  if (TripleValueID >= Tokens.size()) {
    DeviceType = info::device_type::all;
  } else {
    auto Iter = std::find_if(std::begin(getSyclDeviceTypeMap()),
                             std::end(getSyclDeviceTypeMap()), FindElement);
    // If no match is found, set device_type 'all',
    // which actually means 'any device_type' will be a match.
    if (Iter == getSyclDeviceTypeMap().end())
      DeviceType = info::device_type::all;
    else {
      DeviceType = Iter->second;
      TripleValueID++;
    }
  }

  // Handle the optional 3rd field of the filter, device number
  // Try to convert the remaining string to an integer.
  // If succeessful, the converted integer is the desired device num.
  if (TripleValueID < Tokens.size()) {
    try {
      DeviceNum = std::stoi(Tokens[TripleValueID].data());
    } catch (...) {
      std::string Message =
          std::string("Invalid device filter: ") + FilterString +
          "\nPossible backend values are "
          "{host,opencl,level_zero,cuda,hip,esimd_emulator,*}.\n"
          "Possible device types are {host,cpu,gpu,acc,*}.\n"
          "Device number should be an non-negative integer.\n";
      throw sycl::invalid_parameter_error(Message, PI_ERROR_INVALID_VALUE);
    }
  }
}

device_filter_list::device_filter_list(const std::string &FilterStr) {
  // First, change the string in all lowercase.
  // This means we allow the user to use both uppercase and lowercase strings.
  std::string FilterString = FilterStr;
  std::transform(FilterString.begin(), FilterString.end(), FilterString.begin(),
                 ::tolower);
  // SYCL_DEVICE_FILTER can set multiple filters separated by commas.
  // convert each filter triple string into an istance of device_filter class.
  size_t Pos = 0;
  while (Pos < FilterString.size()) {
    size_t CommaPos = FilterString.find(",", Pos);
    if (CommaPos == std::string::npos) {
      CommaPos = FilterString.size();
    }
    std::string SubString = FilterString.substr(Pos, CommaPos - Pos);
    FilterList.push_back(device_filter(SubString));
    Pos = CommaPos + 1;
  }
}

device_filter_list::device_filter_list(device_filter &Filter) {
  FilterList.push_back(Filter);
}

void device_filter_list::addFilter(device_filter &Filter) {
  FilterList.push_back(Filter);
}

// Backend is compatible with the SYCL_DEVICE_FILTER in the following cases.
// 1. Filter backend is '*' which means ANY backend.
// 2. Filter backend match exactly with the given 'Backend'
bool device_filter_list::backendCompatible(backend Backend) {
  for (const device_filter &Filter : FilterList) {
    backend FilterBackend =
        Filter.Backend ? Filter.Backend.value() : backend::all;
    if (FilterBackend == Backend || FilterBackend == backend::all)
      return true;
  }
  return false;
}

bool device_filter_list::deviceTypeCompatible(info::device_type DeviceType) {
  for (const device_filter &Filter : FilterList) {
    info::device_type FilterDevType =
        Filter.DeviceType ? Filter.DeviceType.value() : info::device_type::all;
    if (FilterDevType == DeviceType || FilterDevType == info::device_type::all)
      return true;
  }
  return false;
}

bool device_filter_list::deviceNumberCompatible(int DeviceNum) {
  for (const device_filter &Filter : FilterList) {
    if (!Filter.DeviceNum || Filter.DeviceNum.value() == DeviceNum)
      return true;
  }
  return false;
}

bool device_filter_list::containsHost() {
  for (const device_filter &Filter : FilterList) {
    if (Filter.Backend == backend::host || Filter.Backend == backend::all)
      if (Filter.DeviceType == info::device_type::host ||
          Filter.DeviceType == info::device_type::all)
        // SYCL RT never creates more than one HOST device.
        // All device numbers other than 0 are rejected.
        if (!Filter.DeviceNum || Filter.DeviceNum.value() == 0)
          return true;
  }
  return false;
}

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
