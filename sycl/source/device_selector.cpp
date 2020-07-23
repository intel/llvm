//==------ device_selector.cpp - SYCL device selector ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/backend_types.hpp>
#include <CL/sycl/device.hpp>
#include <CL/sycl/device_selector.hpp>
#include <CL/sycl/exception.hpp>
#include <CL/sycl/stl.hpp>
#include <detail/device_impl.hpp>
#include <detail/force_device.hpp>

#include <algorithm>
// 4.6.1 Device selection class

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

// Utility function to check if device is of the preferred backend.
// Currently preference is given to the level0 backend.
static bool isDeviceOfPreferredSyclBe(const device &Device) {
  if (Device.is_host())
    return false;

  return detail::getSyclObjImpl(Device)->getPlugin().getBackend() ==
         backend::level0;
}

device device_selector::select_device() const {
  vector_class<device> devices = device::get_devices();
  int score = REJECT_DEVICE_SCORE;
  const device *res = nullptr;

  for (const auto &dev : devices) {
    int dev_score = (*this)(dev);

    if (detail::pi::trace(detail::pi::TraceLevel::PI_TRACE_ALL)) {
      string_class PlatformVersion = dev.get_info<info::device::platform>()
                                         .get_info<info::platform::version>();
      string_class DeviceName = dev.get_info<info::device::name>();
      std::cout << "SYCL_PI_TRACE[all]: "
                << "select_device(): -> score = " << score
                << ((score < 0) ? "(REJECTED)" : " ") << std::endl
                << "SYCL_PI_TRACE[all]: "
                << "  platform: " << PlatformVersion << std::endl
                << "SYCL_PI_TRACE[all]: "
                << "  device: " << DeviceName << std::endl;
    }

    // A negative score means that a device must not be selected.
    if (dev_score < 0)
      continue;

    // SYCL spec says: "If more than one device receives the high score then
    // one of those tied devices will be returned, but which of the devices
    // from the tied set is to be returned is not defined". Here we give a
    // preference to the device of the preferred BE.
    //
    if ((score < dev_score) ||
        (score == dev_score && isDeviceOfPreferredSyclBe(dev))) {
      res = &dev;
      score = dev_score;
    }
  }

  if (res != nullptr) {
    if (detail::pi::trace(detail::pi::TraceLevel::PI_TRACE_BASIC)) {
      string_class PlatformVersion = res->get_info<info::device::platform>()
                                         .get_info<info::platform::version>();
      string_class DeviceName = res->get_info<info::device::name>();
      std::cout << "SYCL_PI_TRACE[all]: "
                << "Selected device ->" << std::endl
                << "SYCL_PI_TRACE[all]: "
                << "  platform: " << PlatformVersion << std::endl
                << "SYCL_PI_TRACE[all]: "
                << "  device: " << DeviceName << std::endl;
    }
    return *res;
  }

  throw cl::sycl::runtime_error("No device of requested type available.",
                                PI_DEVICE_NOT_FOUND);
}

/// Devices of different kinds are prioritized in the following order:
/// 1. GPU
/// 2. CPU
/// 3. Host
int default_selector::operator()(const device &dev) const {

  int Score = REJECT_DEVICE_SCORE;

  // Give preference to device of SYCL BE.
  if (isDeviceOfPreferredSyclBe(dev))
    Score = 50;

  // override always wins
  if (dev.get_info<info::device::device_type>() == detail::get_forced_type())
    Score += 1000;

  if (dev.is_gpu())
    Score += 500;

  if (dev.is_cpu())
    Score += 300;

  if (dev.is_host())
    Score += 100;

  return Score;
}

int gpu_selector::operator()(const device &dev) const {
  int Score = REJECT_DEVICE_SCORE;

  if (dev.is_gpu()) {
    Score = 1000;
    // Give preference to device of SYCL BE.
    if (isDeviceOfPreferredSyclBe(dev))
      Score += 50;
  }
  return Score;
}

int cpu_selector::operator()(const device &dev) const {
  int Score = REJECT_DEVICE_SCORE;
  if (dev.is_cpu()) {
    Score = 1000;
    // Give preference to device of SYCL BE.
    if (isDeviceOfPreferredSyclBe(dev))
      Score += 50;
  }
  return Score;
}

int accelerator_selector::operator()(const device &dev) const {
  int Score = REJECT_DEVICE_SCORE;
  if (dev.is_accelerator()) {
    Score = 1000;
    // Give preference to device of SYCL BE.
    if (isDeviceOfPreferredSyclBe(dev))
      Score += 50;
  }
  return Score;
}

int host_selector::operator()(const device &dev) const {
  int Score = REJECT_DEVICE_SCORE;
  if (dev.is_host()) {
    Score = 1000;
    // Give preference to device of SYCL BE.
    if (isDeviceOfPreferredSyclBe(dev))
      Score += 50;
  }
  return Score;
}

namespace detail {
std::string trim_spaces(std::string input) {
  size_t LStart = input.find_first_not_of(" ");
  std::string LTrimmed =
      (LStart == std::string::npos) ? "" : input.substr(LStart);

  size_t REnd = LTrimmed.find_last_not_of(" ");
  return (REnd == std::string::npos) ? "" : LTrimmed.substr(0, REnd + 1);
}

std::vector<std::string> tokenize(std::string filter, std::string delim) {
  std::vector<std::string> Tokens;
  size_t Pos = 0;
  std::string Input = filter;
  std::string Tok;

  while ((Pos = Input.find(delim)) != std::string::npos) {
    Tok = Input.substr(0, Pos);
    Input.erase(0, Pos + delim.length());
    // Erase leading and trailing WS
    Tok = trim_spaces(Tok);

    if (Tok.size() > 0)
      Tokens.push_back(Tok);
  }

  if (Input.size() > 0)
    Input = trim_spaces(Input);

  // Add remainder
  if (Input.size() > 0)
    Tokens.push_back(Input);

  return Tokens;
}

enum TokenType { TokPlatform, TokDeviceType, TokUnknown };

TokenType parse_kind(std::string token) {
  TokenType Result = TokUnknown;

  if (token.find("platform") != std::string::npos)
    Result = TokPlatform;
  if (token.find("type") != std::string::npos)
    Result = TokDeviceType;

  return Result;
}

std::string strip_kind(std::string token) {
  std::string Prefix = "=";
  size_t Loc = token.find(Prefix);

  if (Loc == std::string::npos)
    return token;

  // move past the '='
  Loc = Loc + 1;

  return token.substr(Loc);
}

bool match(std::string input, std::string pattern) {
  return (input.find(pattern) != std::string::npos);
}
} // namespace detail

namespace ext {
namespace oneapi {
string_selector::string_selector(std::string filter) {
  std::transform(filter.begin(), filter.end(), filter.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  std::vector<std::string> Tokens = detail::tokenize(filter, ";");

  for (auto Tok : Tokens) {
    if (Tok.find("=") == std::string::npos)
      continue;

    detail::TokenType TTy = detail::parse_kind(Tok);
    std::string SubTok = detail::strip_kind(Tok);
    std::vector<std::string> SubTokens = detail::tokenize(SubTok, ",");

    if (TTy == detail::TokPlatform) {
      mPlatforms.insert(mPlatforms.end(), SubTokens.begin(), SubTokens.end());
    } else if (TTy == detail::TokDeviceType) {
      mDeviceTypes.insert(mDeviceTypes.end(), SubTokens.begin(),
                          SubTokens.end());
    } else {
      throw runtime_error("Invalid string_selector input! Please specify at "
                          "least one platform or device type filter.",
                          PI_INVALID_VALUE);
    }
  }
}

int string_selector::operator()(const device &dev) const {
  int Score = REJECT_DEVICE_SCORE;

  std::string CPU = "cpu";
  std::string GPU = "gpu";
  std::string PlatformName =
      dev.get_platform().get_info<info::platform::name>();
  std::transform(PlatformName.begin(), PlatformName.end(), PlatformName.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  if (mPlatforms.empty() && mDeviceTypes.empty()) {
    Score = mRanker(dev);
  } else if (!mPlatforms.empty() && mDeviceTypes.empty()) {
    for (auto Plat : mPlatforms) {
      if (detail::match(PlatformName, Plat))
        Score = mRanker(dev);
    }
  } else if (mPlatforms.empty() && !mDeviceTypes.empty()) {
    for (auto DT : mDeviceTypes) {
      if ((detail::match(DT, CPU) && dev.is_cpu()) ||
          (detail::match(DT, GPU) && dev.is_gpu()))
        Score = mRanker(dev);
    }
  } else {
    for (auto Plat : mPlatforms) {
      for (auto DT : mDeviceTypes) {
        if (detail::match(PlatformName, Plat) &&
            ((detail::match(DT, CPU) && dev.is_cpu()) ||
             (detail::match(DT, GPU) && dev.is_gpu())))
          Score = mRanker(dev);
      }
    }
  }

  return Score;
}
} // namespace oneapi
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
