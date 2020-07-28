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
#include <detail/plugin.hpp>
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

// return a device with the requested deviceType (and requested backend)
// if no such device is found, heuristic is used to select a device.
// 'deviceType' is the desired device type
//   info::device_type::all means it relies on the heuristic to select a device
// 'be' is a specific desired GPU backend choice when multiple backends are found.
device device_selector::select_device(info::device_type deviceType, backend be) const {
  // return if a requested deviceType is found
  if (deviceType != info::device_type::all) {
    if (deviceType == info::device_type::host) {
      vector_class<device> devices;
      devices.resize(1);
      return devices[0];
    }
    
    const vector_class<detail::plugin> &plugins = RT::initialize();
    for (unsigned int i = 0; i < plugins.size(); i++) {
      pi_uint32 numPlatforms = 0;
      plugins[i].call<detail::PiApiKind::piPlatformsGet>(0, nullptr, &numPlatforms);
      if (numPlatforms) {
	vector_class<RT::PiPlatform> piPlatforms(numPlatforms);
	plugins[i].call<detail::PiApiKind::piPlatformsGet>(numPlatforms,
						   piPlatforms.data(), nullptr);
	for (const auto &piPlatform : piPlatforms) {
	  platform pltf = detail::createSyclObjFromImpl<platform>(
		std::make_shared<detail::platform_impl>(piPlatform, plugins[i]));
	  if (!pltf.is_host()) {
	    vector_class<device> devices = pltf.get_devices(deviceType);
	    for (uint32_t i=0; i<devices.size(); i++) {
	      if (deviceType != info::device_type::gpu) {
		return devices[i];
	      } else if (devices[i].is_gpu() && be == pltf.get_backend()) {
		return devices[i];
	      }
	    }
	  }
	}
      }
    }
  }
  
  // return a device that has the highest score according to heuristic
  vector_class<device> devices = device::get_devices(deviceType);
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
    string_class PlatformName = res->get_info<info::device::platform>()
      .get_info<info::platform::name>();
    if (detail::pi::trace(detail::pi::TraceLevel::PI_TRACE_BASIC)) {
      string_class DeviceName = res->get_info<info::device::name>();
      std::cout << "SYCL_PI_TRACE[all]: "
                << "Selected device ->" << std::endl
                << "SYCL_PI_TRACE[all]: "
                << "  platform: " << PlatformName << std::endl
                << "SYCL_PI_TRACE[all]: "
                << "  device: " << DeviceName << std::endl;
    }
    if (deviceType != info::device_type::all) {
      std::cout << "WARNING: the requested device and/or backend is not found.\n";
      std::cout << PlatformName << " is chosen based on a heuristic.\n";
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

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
