//==------ filter_selector.cpp - oneapi filter selector --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/device_impl.hpp>
#include <detail/filter_selector_impl.hpp>
#include <sycl/backend_types.hpp>
#include <sycl/device.hpp>
#include <sycl/device_selector.hpp>
#include <sycl/exception.hpp>
#include <sycl/stl.hpp>

#include <cctype>
#include <regex>
#include <string>
#include <vector>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::detail {

std::vector<std::string> tokenize(const std::string &Filter,
                                  const std::string &Delim) {
  std::vector<std::string> Tokens;
  size_t Pos = 0;
  std::string Input = Filter;
  std::string Tok;

  while ((Pos = Input.find(Delim)) != std::string::npos) {
    Tok = Input.substr(0, Pos);
    Input.erase(0, Pos + Delim.length());

    if (!Tok.empty()) {
      Tokens.push_back(std::move(Tok));
    }
  }

  // Add remainder
  if (!Input.empty())
    Tokens.push_back(std::move(Input));

  return Tokens;
}

filter create_filter(const std::string &Input) {
  filter Result;
  constexpr auto Error = "Invalid filter string! Valid strings conform to "
                         "BE:DeviceType:DeviceNum, where any are optional";

  std::vector<std::string> Tokens = tokenize(Input, ":");
  std::regex IntegerExpr("[[:digit:]]+");

  // There should only be up to 3 tokens.
  // BE:Device Type:Device Num
  if (Tokens.size() > 3)
    throw sycl::runtime_error(Error, PI_ERROR_INVALID_VALUE);

  for (const std::string &Token : Tokens) {
    if (Token == "cpu" && !Result.DeviceType) {
      Result.DeviceType = sycl::info::device_type::cpu;
    } else if (Token == "gpu" && !Result.DeviceType) {
      Result.DeviceType = sycl::info::device_type::gpu;
    } else if (Token == "accelerator" && !Result.DeviceType) {
      Result.DeviceType = sycl::info::device_type::accelerator;
    } else if (Token == "opencl" && !Result.Backend) {
      Result.Backend = backend::opencl;
    } else if (Token == "level_zero" && !Result.Backend) {
      Result.Backend = backend::ext_oneapi_level_zero;
    } else if (Token == "cuda" && !Result.Backend) {
      Result.Backend = backend::ext_oneapi_cuda;
    } else if (Token == "hip" && !Result.Backend) {
      Result.Backend = backend::ext_oneapi_hip;
    } else if (std::regex_match(Token, IntegerExpr) && !Result.DeviceNum) {
      try {
        Result.DeviceNum = std::stoi(Token);
      } catch (std::logic_error &) {
        throw sycl::runtime_error(Error, PI_ERROR_INVALID_VALUE);
      }
    } else {
      throw sycl::runtime_error(Error, PI_ERROR_INVALID_VALUE);
    }
  }

  return Result;
}

filter_selector_impl::filter_selector_impl(const std::string &Input)
    : mFilters(), mNumDevicesSeen(0), mMatchFound(false) {
  std::vector<std::string> Filters = detail::tokenize(Input, ",");
  mNumTotalDevices = device::get_devices().size();

  for (const std::string &Filter : Filters) {
    detail::filter F = detail::create_filter(Filter);
    mFilters.push_back(std::move(F));
  }
}

int filter_selector_impl::operator()(const device &Dev) const {
  assert(!sycl::detail::getSyclObjImpl(Dev)->is_host() &&
         "filter_selector_impl should not be used with host.");

  int Score = REJECT_DEVICE_SCORE;

  for (auto &Filter : mFilters) {
    bool BackendOK = true;
    bool DeviceTypeOK = true;
    bool DeviceNumOK = true;

    if (Filter.Backend) {
      backend BE = sycl::detail::getSyclObjImpl(Dev)->getBackend();
      // Backend is okay if the filter BE is set 'all'.
      if (Filter.Backend.value() == backend::all)
        BackendOK = true;
      else
        BackendOK = (BE == Filter.Backend.value());
    }
    if (Filter.DeviceType) {
      sycl::info::device_type DT =
          Dev.get_info<sycl::info::device::device_type>();
      // DeviceType is okay if the filter is set 'all'.
      if (Filter.DeviceType == sycl::info::device_type::all)
        DeviceTypeOK = true;
      else
        DeviceTypeOK = (DT == Filter.DeviceType);
    }
    if (Filter.DeviceNum) {
      // Only check device number if we're good on the previous matches
      if (BackendOK && DeviceTypeOK) {
        // Do we match?
        DeviceNumOK = (Filter.MatchesSeen == Filter.DeviceNum.value());
        // Safe to increment matches even if we find it
        Filter.MatchesSeen++;
      }
    }
    if (BackendOK && DeviceTypeOK && DeviceNumOK) {
      Score = default_selector_v(Dev);
      mMatchFound = true;
      break;
    }
  }

  mNumDevicesSeen++;
  if ((mNumDevicesSeen == mNumTotalDevices) && !mMatchFound) {
    throw sycl::runtime_error(
        "Could not find a device that matches the specified filter(s)!",
        PI_ERROR_DEVICE_NOT_FOUND);
  }

  return Score;
}

void filter_selector_impl::reset() const {
  // This is a bit of an abuse of "const" method...
  // Reset state if you want to reuse this selector.
  for (auto &Filter : mFilters) {
    Filter.MatchesSeen = 0;
  }
  mMatchFound = false;
  mNumDevicesSeen = 0;
}

} // namespace ext::oneapi::detail

namespace __SYCL2020_DEPRECATED("use 'ext::oneapi' instead") ONEAPI {
using namespace ext::oneapi;
}
} // namespace _V1
} // namespace sycl
