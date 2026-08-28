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
#include <sycl/ext/oneapi/filter_selector.hpp>

#include <algorithm>
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
    throw exception(make_error_code(errc::invalid), Error);

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
        throw exception(make_error_code(errc::invalid), Error);
      }
    } else {
      throw exception(make_error_code(errc::invalid), Error);
    }
  }

  return Result;
}

filter_selector_impl::filter_selector_impl(const std::string &Input) {
  std::vector<filter> Filters;
  for (const std::string &Filter : detail::tokenize(Input, ","))
    Filters.push_back(detail::create_filter(Filter));

  // Matching the filters requires state to be kept between the devices (to
  // track the relative device number), so do it once here instead of doing it
  // in operator().
  for (const device &Dev : device::get_devices()) {
    for (filter &Filter : Filters) {
      bool BackendOK = true;
      bool DeviceTypeOK = true;
      bool DeviceNumOK = true;

      if (Filter.Backend) {
        // Backend is okay if the filter BE is set 'all'.
        BackendOK = Filter.Backend.value() == backend::all ||
                    sycl::detail::getSyclObjImpl(Dev)->getBackend() ==
                        Filter.Backend.value();
      }
      if (Filter.DeviceType) {
        // DeviceType is okay if the filter is set 'all'.
        DeviceTypeOK =
            Filter.DeviceType.value() == sycl::info::device_type::all ||
            Dev.get_info<sycl::info::device::device_type>() ==
                Filter.DeviceType.value();
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
        mMatchingDevices.push_back(Dev);
        break;
      }
    }
  }
}

int filter_selector_impl::operator()(const device &Dev) const {
  if (mMatchingDevices.empty())
    throw exception(
        make_error_code(errc::runtime),
        "Could not find a device that matches the specified filter(s)!");

  if (std::find(mMatchingDevices.begin(), mMatchingDevices.end(), Dev) ==
      mMatchingDevices.end())
    return REJECT_DEVICE_SCORE;

  // Let the default selector rank the devices that passed the filters.
  return default_selector_v(Dev);
}

} // namespace ext::oneapi::detail

namespace ext::oneapi {

filter_selector::filter_selector(sycl::detail::string_view Input)
    : impl(std::make_shared<detail::filter_selector_impl>(
          std::string(std::string_view(Input)))) {}

int filter_selector::operator()(const device &Dev) const {
  return impl->operator()(Dev);
}

void filter_selector::reset() const {
  // The selector keeps no state between the invocations of operator(), so
  // there is nothing to reset. Kept for backwards compatibility.
}

device filter_selector::select_device() const {
  return sycl::detail::select_device(*this);
}

} // namespace ext::oneapi

namespace __SYCL2020_DEPRECATED("use 'ext::oneapi' instead") ONEAPI {
using namespace ext::oneapi;

filter_selector::filter_selector(sycl::detail::string_view Input)
    : ext::oneapi::filter_selector(Input) {}

int filter_selector::operator()(const device &Dev) const {
  return ext::oneapi::filter_selector::operator()(Dev);
}

void filter_selector::reset() const { ext::oneapi::filter_selector::reset(); }

device filter_selector::select_device() const {
  return ext::oneapi::filter_selector::select_device();
}
} // namespace __SYCL2020_DEPRECATED("use 'ext::oneapi' instead")ONEAPI
} // namespace _V1
} // namespace sycl
