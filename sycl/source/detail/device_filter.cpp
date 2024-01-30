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
inline namespace _V1 {
namespace detail {

std::vector<std::string_view> tokenize(const std::string_view &Filter,
                                       const std::string &Delim,
                                       bool ProhibitEmptyTokens = false) {
  std::vector<std::string_view> Tokens;
  size_t Pos = 0;
  size_t LastPos = 0;

  while ((Pos = Filter.find(Delim, LastPos)) != std::string::npos) {
    std::string_view Tok(Filter.data() + LastPos, (Pos - LastPos));

    if (!Tok.empty()) {
      Tokens.push_back(Tok);
    } else if (ProhibitEmptyTokens) {
      throw sycl::exception(
          sycl::make_error_code(errc::invalid),
          "ONEAPI_DEVICE_SELECTOR parsing error. Empty input before '" + Delim +
              "' delimiter is not allowed.");
    }
    // move the search starting index
    LastPos = Pos + 1;
  }

  // Add remainder if any
  if (LastPos < Filter.size()) {
    std::string_view Tok(Filter.data() + LastPos, Filter.size() - LastPos);
    Tokens.push_back(Tok);
  } else if ((LastPos != 0) && ProhibitEmptyTokens) {
    // if delimiter is the last sybmol in the string.
    throw sycl::exception(
        sycl::make_error_code(errc::invalid),
        "ONEAPI_DEVICE_SELECTOR parsing error. Empty input after '" + Delim +
            "' delimiter is not allowed.");
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
  auto It =
      std::find_if(std::begin(SyclBeMap), std::end(SyclBeMap),
                   [&](auto BePair) { return BackendStr == BePair.first; });

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
  std::vector<std::string_view> DeviceSubTuple =
      tokenize(DeviceStr, ".", true /* ProhibitEmptyTokens */);
  if (DeviceSubTuple.empty())
    throw sycl::exception(
        sycl::make_error_code(errc::invalid),
        "ONEAPI_DEVICE_SELECTOR parsing error. Device must be specified.");

  std::string_view TopDeviceStr = DeviceSubTuple[0];

  // Handle explicit device type (e.g. 'gpu').
  auto DeviceTypeMap =
      getODSDeviceTypeMap(); // <-- std::array<std::pair<std::string,
                              // info::device::type>>
  auto It =
      std::find_if(std::begin(DeviceTypeMap), std::end(DeviceTypeMap),
                   [&](auto DtPair) { return TopDeviceStr == DtPair.first; });
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
      throw sycl::exception(sycl::make_error_code(errc::invalid),
                            "error parsing device number: " + TDS);
    }
  }

  if (DeviceSubTuple.size() >= 2) {
    // We have a subdevice.
    // The grammar for sub-devices is ... restrictive. Neither 'gpu.0' nor
    // 'gpu.*' are allowed. If wanting a sub-device, then the device itself must
    // be specified by a number or a wildcard, and if by wildcard, the only
    // allowable sub-device is another wildcard.

    if (Target.DeviceType)
      throw sycl::exception(
          sycl::make_error_code(errc::invalid),
          "sub-devices can only be requested when parent device is specified "
          "by number or wildcard, not a device type like 'gpu'");

    std::string_view SubDeviceStr = DeviceSubTuple[1];
    // SubDeviceStr is wildcard or number.
    if (SubDeviceStr[0] == '*') {
      Target.HasSubDeviceWildCard = true;
    } else {
      // sub-device requested by number. So parent device must be a number too
      // or it's a parsing error.
      if (Target.HasDeviceWildCard)
        throw sycl::exception(sycl::make_error_code(errc::invalid),
                              "sub-device can't be requested by number if "
                              "parent device is specified by a wildcard.");

      std::string SDS(SubDeviceStr);
      try {
        Target.SubDeviceNum = std::stoi(SDS);
      } catch (...) {
        throw sycl::exception(sycl::make_error_code(errc::invalid),
                              "error parsing sub-device index: " + SDS);
      }
    }
  }
  if (DeviceSubTuple.size() == 3) {
    // We have a sub-sub-device.
    // Similar rules for sub-sub-devices as for sub-devices above.

    std::string_view SubSubDeviceStr = DeviceSubTuple[2];
    if (SubSubDeviceStr[0] == '*') {
      Target.HasSubSubDeviceWildCard = true;
    } else {
      // sub-sub-device requested by number. So partition above must be a number
      // too or it's a parsing error.
      if (Target.HasSubDeviceWildCard)
        throw sycl::exception(sycl::make_error_code(errc::invalid),
                              "sub-sub-device can't be requested by number if "
                              "sub-device before is specified by a wildcard.");

      std::string SSDS(SubSubDeviceStr);
      try {
        Target.SubSubDeviceNum = std::stoi(SSDS);
      } catch (...) {
        throw sycl::exception(sycl::make_error_code(errc::invalid),
                              "error parsing sub-sub-device index: " + SSDS);
      }
    }
  } else if (DeviceSubTuple.size() > 3) {
    std::stringstream ss;
    ss << "error parsing " << DeviceStr
       << "  Only two levels of sub-devices supported at this time";
    throw sycl::exception(sycl::make_error_code(errc::invalid), ss.str());
  }
}

std::vector<ods_target>
Parse_ONEAPI_DEVICE_SELECTOR(const std::string &envString) {
  // lowercase
  std::string envStr = envString;
  std::transform(envStr.begin(), envStr.end(), envStr.begin(), ::tolower);

  std::vector<ods_target> Result;
  if (envStr.empty()) {
    ods_target acceptAnything;
    Result.push_back(acceptAnything);
    return Result;
  }

  std::vector<std::string_view> Entries = tokenize(envStr, ";");
  unsigned int negative_filters = 0;
  // Each entry: "level_zero:gpu" or "opencl:0.0,0.1" or "opencl:*" but NOT just
  // "opencl".
  for (const auto Entry : Entries) {
    std::vector<std::string_view> Pair =
        tokenize(Entry, ":", true /* ProhibitEmptyTokens */);

    if (Pair.empty()) {
      std::stringstream ss;
      ss << "Incomplete selector! Backend and device must be specified.";
      throw sycl::exception(sycl::make_error_code(errc::invalid), ss.str());
    } else if (Pair.size() == 1) {
      std::stringstream ss;
      ss << "Incomplete selector!  Try '" << Pair[0]
         << ":*' if all devices under the backend was original intention.";
      throw sycl::exception(sycl::make_error_code(errc::invalid), ss.str());
    } else if (Pair.size() == 2) {
      backend be = Parse_ODS_Backend(Pair[0], Entry); // Pair[0] is backend.
      std::vector<std::string_view> Targets = tokenize(Pair[1], ",");
      for (auto TargetStr : Targets) {
        ods_target DeviceTarget(be);
        if (Entry[0] == '!') { // negative filter
          DeviceTarget.IsNegativeTarget = true;
          ++negative_filters;
        } else { // positive filter
          // no need to set IsNegativeTarget=false because it is so by default.
          // ensure that no negative filter has been seen because all
          // negative filters must come after all positive filters
          if (negative_filters > 0) {
            std::stringstream ss;
            ss << "All negative(discarding) filters must appear after all "
                  "positive(accepting) filters!";
            throw sycl::exception(sycl::make_error_code(errc::invalid),
                                  ss.str());
          }
        }
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

  // This if statement handles the special case when the filter list
  // contains at least one negative filter but no positive filters.
  // This means that no devices will be available at all and so its as if
  // the filter list was empty because the negative filters do not have any
  // any effect. Hoewever, it is desirable to be able to set the
  // ONEAPI_DEVICE_SELECTOR=!*:gpu to consider all devices except gpu
  // devices so that we must implicitly add an acceptall target to the
  // list of targets to make this work. So the result will be as if
  // the filter string had the *:* string in it.
  if (!Result.empty() && negative_filters == Result.size()) {
    ods_target acceptAll{backend::all};
    acceptAll.DeviceType = info::device_type::all;
    Result.push_back(acceptAll);
  }
  return Result;
}

std::ostream &operator<<(std::ostream &Out, const ods_target &Target) {
  Out << Target.Backend;
  if (Target.DeviceType) {
    auto DeviceTypeMap = getODSDeviceTypeMap();
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

// Backend is compatible with the ONEAPI_DEVICE_SELECTOR in the following cases.
// 1. Filter backend is '*' which means ANY backend.
// 2. Filter backend match exactly with the given 'Backend'
bool ods_target_list::backendCompatible(backend Backend) {

  return std::any_of(
      TargetList.begin(), TargetList.end(), [&](ods_target &Target) {
        backend TargetBackend = Target.Backend.value_or(backend::all);
        return (TargetBackend == Backend) || (TargetBackend == backend::all);
      });
}

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
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

    if (Backend == backend::host)
      std::cerr << "WARNING: The 'host' backend type is no longer supported in "
                   "device filter."
                << std::endl;
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

      if (DeviceType == info::device_type::host)
        std::cerr << "WARNING: The 'host' device type is no longer supported "
                     "in device filter."
                  << std::endl;
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
          "{opencl,level_zero,cuda,hip,*}.\n"
          "Possible device types are {cpu,gpu,acc,*}.\n"
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
  return std::any_of(
      FilterList.begin(), FilterList.end(), [&](device_filter &Filter) {
        backend FilterBackend = Filter.Backend.value_or(backend::all);
        return (FilterBackend == Backend) || (FilterBackend == backend::all);
      });
}

bool device_filter_list::deviceTypeCompatible(info::device_type DeviceType) {
  return std::any_of(FilterList.begin(), FilterList.end(),
                     [&](device_filter &Filter) {
                       info::device_type FilterDevType =
                           Filter.DeviceType.value_or(info::device_type::all);
                       return (FilterDevType == DeviceType) ||
                              (FilterDevType == info::device_type::all);
                     });
}

bool device_filter_list::deviceNumberCompatible(int DeviceNum) {
  return std::any_of(
      FilterList.begin(), FilterList.end(), [&](device_filter &Filter) {
        return (!Filter.DeviceNum) || (Filter.DeviceNum.value() == DeviceNum);
      });
}
#endif // __INTEL_PREVIEW_BREAKING_CHANGES

} // namespace detail
} // namespace _V1
} // namespace sycl
