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

#include <algorithm>
#include <cctype>
#include <regex>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

// Utility function to check if device is of the preferred backend.
// Currently preference is given to the level_zero backend.
static bool isDeviceOfPreferredSyclBe(const device &Device) {
  if (Device.is_host())
    return false;

  return detail::getSyclObjImpl(Device)->getPlugin().getBackend() ==
         backend::level_zero;
}

device device_selector::select_device() const {
  vector_class<device> devices = device::get_devices();
  int score = REJECT_DEVICE_SCORE;
  const device *res = nullptr;

  for (const auto &dev : devices) {
    int dev_score = (*this)(dev);

    if (detail::pi::trace(detail::pi::TraceLevel::PI_TRACE_ALL)) {
      string_class PlatformName = dev.get_info<info::device::platform>()
                                      .get_info<info::platform::name>();
      string_class DeviceName = dev.get_info<info::device::name>();
      std::cout << "SYCL_PI_TRACE[all]: "
                << "select_device(): -> score = " << dev_score
                << ((dev_score < 0) ? " (REJECTED)" : "") << std::endl
                << "SYCL_PI_TRACE[all]: "
                << "  platform: " << PlatformName << std::endl
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
      string_class PlatformName = res->get_info<info::device::platform>()
                                      .get_info<info::platform::name>();
      string_class DeviceName = res->get_info<info::device::name>();
      std::cout << "SYCL_PI_TRACE[all]: "
                << "Selected device ->" << std::endl
                << "SYCL_PI_TRACE[all]: "
                << "  platform: " << PlatformName << std::endl
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

namespace ONEAPI {
namespace detail {
struct filter {
  backend Backend;
  RT::PiDeviceType DeviceType;
  int DeviceNum;
  bool hasBackend = false;
  bool hasDeviceType = false;
  bool hasDeviceNum = false;
  int MatchesSeen = 0;
};

std::vector<std::string> tokenize(std::string filter, std::string delim) {
  std::vector<std::string> Tokens;
  size_t Pos = 0;
  std::string Input = filter;
  std::string Tok;

  while ((Pos = Input.find(delim)) != std::string::npos) {
    Tok = Input.substr(0, Pos);
    Input.erase(0, Pos + delim.length());

    if (!Tok.empty()) {
      Tokens.push_back(Tok);
    }
  }

  // Add remainder
  if (!Input.empty())
    Tokens.push_back(Input);

  return Tokens;
}

filter create_filter(std::string Input) {
  filter Result;
  std::string Error = "Invalid filter string! Valid strings conform to "
                      "BE:DeviceType:DeviceNum, where any are optional";

  std::vector<std::string> Tokens = tokenize(Input, ":");
  std::regex IntegerExpr("[[:digit:]]+");

  std::cout << "Tokens = { ";
  for (auto S : Tokens) {
    std::cout << S << "  ";
  }
  std::cout << "}" << std::endl;

  // There should only be up to 3 tokens.
  // BE:Device Type:Device Num
  if (Tokens.size() > 3)
    throw runtime_error(Error, PI_INVALID_VALUE);

  for (const std::string &Token : Tokens) {
    if (Token == "cpu" && !Result.hasDeviceType) {
      Result.DeviceType = PI_DEVICE_TYPE_CPU;
      Result.hasDeviceType = true;
    } else if (Token == "gpu" && !Result.hasDeviceType) {
      Result.DeviceType = PI_DEVICE_TYPE_GPU;
      Result.hasDeviceType = true;
    } else if (Token == "accelerator" && !Result.hasDeviceType) {
      Result.DeviceType = PI_DEVICE_TYPE_ACC;
      Result.hasDeviceType = true;
    } else if (Token == "opencl" && !Result.hasBackend) {
      Result.Backend = backend::opencl;
      Result.hasBackend = true;
    } else if (Token == "level-zero" && !Result.hasBackend) {
      Result.Backend = backend::level_zero;
      Result.hasBackend = true;
    } else if (Token == "cuda" && !Result.hasBackend) {
      Result.Backend = backend::cuda;
      Result.hasBackend = true;
    } else if (Token == "host") {
      if (!Result.hasBackend) {
        Result.Backend = backend::host;
        Result.hasBackend = true;
      } else if (!Result.hasDeviceType && Result.Backend != backend::host) {
        // We already set everything earlier or it's an error.
        throw runtime_error("Cannot specify host device with non-host backend.",
                            PI_INVALID_VALUE);
      }
    } else if (std::regex_match(Token, IntegerExpr) && !Result.hasDeviceNum) {
      Result.DeviceNum = std::stoi(Token);
      Result.hasDeviceNum = true;
    } else {
      throw runtime_error(Error, PI_INVALID_VALUE);
    }
  }

  return Result;
}
} // namespace detail

filter_selector::filter_selector(std::string filter)
    : mFilters(), mRanker(), mNumDevicesSeen(0), mMatchFound(false) {
  std::vector<std::string> Filters = detail::tokenize(filter, ",");
  mNumTotalDevices = device::get_devices().size();

  std::cout << "Input Filters = { ";
  for (auto S : Filters) {
    std::cout << S << "  ";
  }
  std::cout << "}" << std::endl;

  for (const std::string &Filter : Filters) {
    detail::filter F = detail::create_filter(Filter);
    mFilters.push_back(std::make_shared<detail::filter>(F));
  }
}

int filter_selector::operator()(const device &dev) const {
  int Score = REJECT_DEVICE_SCORE;

  for (auto &Filter : mFilters) {
    bool BackendOK = true;
    bool DeviceTypeOK = true;
    bool DeviceNumOK = true;

    std::cout << "Filter: " << std::endl;
    std::cout << "  hasBackend = " << Filter->hasBackend << std::endl;
    std::cout << "  hasDeviceType = " << Filter->hasDeviceType << std::endl;
    std::cout << "  hasDeviceNum = " << Filter->hasDeviceNum << std::endl;
    std::cout << "  DeviceNum = " << Filter->DeviceNum << std::endl;
    std::cout << "  MatchesSeen = " << Filter->MatchesSeen << std::endl;
    std::cout << "  mNumDevicesSeen  = " << mNumDevicesSeen << std::endl;
    std::cout << "  mNumTotalDevices = " << mNumTotalDevices << std::endl;

    // handle host device specially
    if (Filter->hasBackend) {
      backend BE;
      if (dev.is_host()) {
        BE = backend::host;
      } else {
        BE = sycl::detail::getSyclObjImpl(dev)->getPlugin().getBackend();
      }
      BackendOK = (BE == Filter->Backend);
    }
    if (Filter->hasDeviceType) {
      RT::PiDeviceType DT =
          sycl::detail::getSyclObjImpl(dev)->get_device_type();
      DeviceTypeOK = (DT == Filter->DeviceType);
    }
    if (Filter->hasDeviceNum) {
      // Only check device number if we're good on the previous matches
      if (BackendOK && DeviceTypeOK) {
        // Do we match?
        DeviceNumOK = (Filter->MatchesSeen == Filter->DeviceNum);
        // Safe to increment matches even if we find it
        Filter->MatchesSeen++;
      }
    }
    if (BackendOK && DeviceTypeOK && DeviceNumOK) {
      Score = mRanker(dev);
      mMatchFound = true;
      break;
    }
  }

  mNumDevicesSeen++;
  std::cout << "mNumDevicesSeen = " << mNumDevicesSeen << std::endl;
  std::cout << "mNumTotalDevices = " << mNumTotalDevices << std::endl;
  std::cout << "mMatchFound = " << mMatchFound << std::endl;
  if ((mNumDevicesSeen == mNumTotalDevices) && !mMatchFound) {
    std::cout << "ERROR" << std::endl;
    throw runtime_error(
        "Could not find a device that matches the specified filter(s)!",
        PI_DEVICE_NOT_FOUND);
  }

  return Score;
}
} // namespace ONEAPI
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
