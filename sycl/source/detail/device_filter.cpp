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
  auto DeviceTypeMap = getSyclDeviceTypeMap();

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

    // Error handling. ONEAPI_DEVICE_SELECTOR terms should be in the
    // format: <backend>:<devices>.
    if (Pair.empty()) {
      std::stringstream ss;
      ss << "Incomplete selector! Backend and device must be specified.";
      throw sycl::exception(sycl::make_error_code(errc::invalid), ss.str());
    } else if (Pair.size() == 1) {
      std::stringstream ss;
      ss << "Incomplete selector!  Try '" << Pair[0]
         << ":*' if all devices under the backend was original intention.";
      throw sycl::exception(sycl::make_error_code(errc::invalid), ss.str());
    } else if (Pair.size() > 2) {
      std::stringstream ss;
      ss << "Error parsing selector string \"" << Entry
         << "\"  Too many colons (:)";
      throw sycl::exception(sycl::make_error_code(errc::invalid), ss.str());
    }

    // Parse ONEAPI_DEVICE_SELECTOR terms for Pair.size() == 2.
    else {

      // Remove `!` from input backend string if it is present.
      std::string_view input_be = Pair[0];
      if (Pair[0][0] == '!')
        input_be = Pair[0].substr(1);

      backend be = Parse_ODS_Backend(input_be, Entry);

      // For each backend, we can have multiple targets, seperated by ','.
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
} // namespace detail
} // namespace _V1
} // namespace sycl
