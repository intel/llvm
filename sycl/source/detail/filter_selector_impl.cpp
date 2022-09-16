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
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext {
namespace oneapi {
namespace detail {

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
    if (Token == "cpu" && !Result.HasDeviceType) {
      Result.DeviceType = sycl::info::device_type::cpu;
      Result.HasDeviceType = true;
    } else if (Token == "gpu" && !Result.HasDeviceType) {
      Result.DeviceType = sycl::info::device_type::gpu;
      Result.HasDeviceType = true;
    } else if (Token == "accelerator" && !Result.HasDeviceType) {
      Result.DeviceType = sycl::info::device_type::accelerator;
      Result.HasDeviceType = true;
    } else if (Token == "opencl" && !Result.HasBackend) {
      Result.Backend = backend::opencl;
      Result.HasBackend = true;
    } else if (Token == "level_zero" && !Result.HasBackend) {
      Result.Backend = backend::ext_oneapi_level_zero;
      Result.HasBackend = true;
    } else if (Token == "cuda" && !Result.HasBackend) {
      Result.Backend = backend::ext_oneapi_cuda;
      Result.HasBackend = true;
    } else if (Token == "hip" && !Result.HasBackend) {
      Result.Backend = backend::ext_oneapi_hip;
      Result.HasBackend = true;
    } else if (Token == "host") {
      if (!Result.HasBackend) {
        Result.Backend = backend::host;
        Result.HasBackend = true;
      } else if (!Result.HasDeviceType && Result.Backend != backend::host) {
        // We already set everything earlier or it's an error.
        throw sycl::runtime_error(
            "Cannot specify host device with non-host backend.",
            PI_ERROR_INVALID_VALUE);
      }
    } else if (std::regex_match(Token, IntegerExpr) && !Result.HasDeviceNum) {
      try {
        Result.DeviceNum = std::stoi(Token);
      } catch (std::logic_error &) {
        throw sycl::runtime_error(Error, PI_ERROR_INVALID_VALUE);
      }
      Result.HasDeviceNum = true;
    } else {
      throw sycl::runtime_error(Error, PI_ERROR_INVALID_VALUE);
    }
  }

  return Result;
}

filter_selector_impl::filter_selector_impl(const std::string &Input)
    : mFilters(), mRanker(), mNumDevicesSeen(0), mMatchFound(false) {
  std::vector<std::string> Filters = detail::tokenize(Input, ",");
  mNumTotalDevices = device::get_devices().size();

  for (const std::string &Filter : Filters) {
    detail::filter F = detail::create_filter(Filter);
    mFilters.push_back(std::move(F));
  }
}

int filter_selector_impl::operator()(const device &Dev) const {
  int Score = REJECT_DEVICE_SCORE;

  for (auto &Filter : mFilters) {
    bool BackendOK = true;
    bool DeviceTypeOK = true;
    bool DeviceNumOK = true;

    // handle host device specially
    if (Filter.HasBackend) {
      backend BE;
      if (Dev.is_host()) {
        BE = backend::host;
      } else {
        BE = sycl::detail::getSyclObjImpl(Dev)->getPlugin().getBackend();
      }
      // Backend is okay if the filter BE is set 'all'.
      if (Filter.Backend == backend::all)
        BackendOK = true;
      else
        BackendOK = (BE == Filter.Backend);
    }
    if (Filter.HasDeviceType) {
      sycl::info::device_type DT =
          Dev.get_info<sycl::info::device::device_type>();
      // DeviceType is okay if the filter is set 'all'.
      if (Filter.DeviceType == sycl::info::device_type::all)
        DeviceTypeOK = true;
      else
        DeviceTypeOK = (DT == Filter.DeviceType);
    }
    if (Filter.HasDeviceNum) {
      // Only check device number if we're good on the previous matches
      if (BackendOK && DeviceTypeOK) {
        // Do we match?
        DeviceNumOK = (Filter.MatchesSeen == Filter.DeviceNum);
        // Safe to increment matches even if we find it
        Filter.MatchesSeen++;
      }
    }
    if (BackendOK && DeviceTypeOK && DeviceNumOK) {
      Score = mRanker(Dev);
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

} // namespace detail
} // namespace oneapi
} // namespace ext

namespace __SYCL2020_DEPRECATED("use 'ext::oneapi' instead") ONEAPI {
using namespace ext::oneapi;
}
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
