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
std::vector<std::string> tokenize(std::string filter, std::string delim) {
  std::vector<std::string> Tokens;
  size_t Pos = 0;
  std::string Input = filter;
  std::string Tok;

  while ((Pos = Input.find(delim)) != std::string::npos) {
    Tok = Input.substr(0, Pos);
    Input.erase(0, Pos + delim.length());
    Tokens.push_back(Tok);
  }

  return Tokens;
}

enum TokenType { TokPlatform, TokDeviceType, TokUnknown };

TokenType parse_kind(std::string token) {
  TokenType Result = TokUnknown;

  if (token.find("platform=") != std::string::npos)
    Result = TokPlatform;
  if (token.find("type=") != std::string::npos)
    Result = TokDeviceType;

  return Result;
}

std::string strip_kind(std::string token, TokenType tty) {
  std::string Prefix = "";

  if (tty == TokPlatform) {
    Prefix = "platform=";
  } else if (tty == TokDeviceType) {
    Prefix = "type=";
  }

  return token.substr(Prefix.size());
}
} // namespace detail

namespace ext {
namespace oneapi {
string_selector::string_selector(std::string filter) {
  std::vector<std::string> Tokens = detail::tokenize(filter, ";");

  for (auto Tok : Tokens) {
    detail::TokenType TTy = detail::parse_kind(Tok);

    if (TTy == detail::TokPlatform) {
      std::string SubTok = detail::strip_kind(Tok, detail::TokPlatform);
      std::vector<std::string> PlatTokens = detail::tokenize(SubTok, ",");

      mPlatforms.insert(mPlatforms.end(), PlatTokens.begin(), PlatTokens.end());
    } else if (TTy == detail::TokDeviceType) {
      std::string SubTok = detail::strip_kind(Tok, detail::TokDeviceType);
      std::vector<std::string> TypeTokens = detail::tokenize(SubTok, ",");

      mDeviceTypes.insert(mDeviceTypes.end(), TypeTokens.begin(),
                          TypeTokens.end());
    } else {
      throw runtime_error("Invalid string_selector input!", PI_INVALID_VALUE);
    }
  }
}

int string_selector::operator()(const device &dev) const {
  int Score = REJECT_DEVICE_SCORE;
  std::string PlatformName =
      dev.get_platform().get_info<info::platform::name>();

  if (mPlatforms.empty() && mDeviceTypes.empty()) {
    Score = mRanker(dev);
  } else if (!mPlatforms.empty() && mDeviceTypes.empty()) {
    for (auto Plat : mPlatforms) {
      Score = mRanker(dev);
    }
  } else if (mPlatforms.empty() && !mDeviceTypes.empty()) {
    for (auto DT : mDeviceTypes) {
      if ((DT == "cpu" && dev.is_cpu()) || (DT == "gpu" && dev.is_gpu()))
        Score = mRanker(dev);
    }
  } else {
    for (auto Plat : mPlatforms) {
      for (auto DT : mDeviceTypes) {
        if ((PlatformName.find(Plat) != std::string::npos) &&
            ((DT == "cpu" && dev.is_cpu()) || (DT == "gpu" && dev.is_gpu())))
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
