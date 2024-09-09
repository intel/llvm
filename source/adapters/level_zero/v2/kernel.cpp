//===--------- kernel.cpp - Level Zero Adapter ---------------------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <ur_api.h>

#include "context.hpp"
#include "kernel.hpp"

#include "../device.hpp"
#include "../platform.hpp"
#include "../program.hpp"
#include "../ur_interface_loader.hpp"

ur_single_device_kernel_t::ur_single_device_kernel_t(ze_device_handle_t hDevice,
                                                     ze_kernel_handle_t hKernel,
                                                     bool ownZeHandle)
    : hDevice(hDevice), hKernel(hKernel, ownZeHandle) {
  zeKernelProperties.Compute =
      [hKernel = hKernel](ze_kernel_properties_t &properties) {
        ZE_CALL_NOCHECK(zeKernelGetProperties, (hKernel, &properties));
      };
}

ur_result_t ur_single_device_kernel_t::release() {
  hKernel.reset();
  return UR_RESULT_SUCCESS;
}

ur_kernel_handle_t_::ur_kernel_handle_t_(ur_program_handle_t hProgram,
                                         const char *kernelName)
    : hProgram(hProgram),
      deviceKernels(hProgram->Context->getPlatform()->getNumDevices()) {
  ur::level_zero::urProgramRetain(hProgram);

  for (auto [zeDevice, zeModule] : hProgram->ZeModuleMap) {
    ZeStruct<ze_kernel_desc_t> zeKernelDesc;
    zeKernelDesc.pKernelName = kernelName;

    ze_kernel_handle_t zeKernel;
    ZE2UR_CALL_THROWS(zeKernelCreate, (zeModule, &zeKernelDesc, &zeKernel));

    auto urDevice = std::find_if(hProgram->Context->getDevices().begin(),
                                 hProgram->Context->getDevices().end(),
                                 [zeDevice = zeDevice](const auto &urDevice) {
                                   return urDevice->ZeDevice == zeDevice;
                                 });
    assert(urDevice != hProgram->Context->getDevices().end());
    auto deviceId = (*urDevice)->Id.value();

    deviceKernels[deviceId].emplace(zeDevice, zeKernel, true);
  }
  completeInitialization();
}

ur_kernel_handle_t_::ur_kernel_handle_t_(
    ur_native_handle_t hNativeKernel, ur_program_handle_t hProgram,
    const ur_kernel_native_properties_t *pProperties)
    : hProgram(hProgram), deviceKernels(1) {
  ze_kernel_handle_t zeKernel = ur_cast<ze_kernel_handle_t>(hNativeKernel);

  if (!zeKernel) {
    throw UR_RESULT_ERROR_INVALID_KERNEL;
  }

  deviceKernels.back().emplace(nullptr, zeKernel,
                               pProperties->isNativeHandleOwned);
  completeInitialization();
}

ur_result_t ur_kernel_handle_t_::release() {
  // manually release kernels to allow errors to be propagated
  for (auto &singleDeviceKernelOpt : deviceKernels) {
    if (singleDeviceKernelOpt.has_value()) {
      singleDeviceKernelOpt.value().hKernel.reset();
    }
  }

  UR_CALL_THROWS(ur::level_zero::urProgramRelease(hProgram));

  return UR_RESULT_SUCCESS;
}

void ur_kernel_handle_t_::completeInitialization() {
  // Cache kernel name. Should be the same for all devices
  assert(deviceKernels.size() > 0);
  auto nonEmptyKernel =
      std::find_if(deviceKernels.begin(), deviceKernels.end(),
                   [](const auto &kernel) { return kernel.has_value(); });

  zeKernelName.Compute = [kernel =
                              &nonEmptyKernel->value()](std::string &name) {
    size_t size = 0;
    ZE_CALL_NOCHECK(zeKernelGetName, (kernel->hKernel.get(), &size, nullptr));
    name.resize(size);
    ZE_CALL_NOCHECK(zeKernelGetName,
                    (kernel->hKernel.get(), &size, name.data()));
  };
}

ze_kernel_handle_t
ur_kernel_handle_t_::getZeHandle(ur_device_handle_t hDevice) {
  // root-device's kernel can be submitted to a sub-device's queue
  if (hDevice->isSubDevice()) {
    hDevice = hDevice->RootDevice;
  }

  if (deviceKernels.size() == 1) {
    assert(deviceKernels[0].has_value());
    assert(deviceKernels[0].value().hKernel.get());

    auto &kernel = deviceKernels[0].value();

    // hDevice is nullptr for native handle
    if ((kernel.hDevice != nullptr && kernel.hDevice != hDevice->ZeDevice)) {
      throw UR_RESULT_ERROR_INVALID_DEVICE;
    }

    return kernel.hKernel.get();
  }

  if (!deviceKernels[hDevice->Id.value()].has_value()) {
    throw UR_RESULT_ERROR_INVALID_DEVICE;
  }

  assert(deviceKernels[hDevice->Id.value()].value().hKernel.get());

  return deviceKernels[hDevice->Id.value()].value().hKernel.get();
}

const std::string &ur_kernel_handle_t_::getName() const {
  return *zeKernelName.operator->();
}

const ze_kernel_properties_t &
ur_kernel_handle_t_::getProperties(ur_device_handle_t hDevice) const {
  if (!deviceKernels[hDevice->Id.value()].has_value()) {
    throw UR_RESULT_ERROR_INVALID_DEVICE;
  }

  assert(deviceKernels[hDevice->Id.value()].value().hKernel.get());

  return *deviceKernels[hDevice->Id.value()]
              .value()
              .zeKernelProperties.
              operator->();
}

ur_result_t ur_kernel_handle_t_::setArgValue(
    uint32_t argIndex, size_t argSize,
    const ur_kernel_arg_value_properties_t *pProperties,
    const void *pArgValue) {
  std::ignore = pProperties;

  // OpenCL: "the arg_value pointer can be NULL or point to a NULL value
  // in which case a NULL value will be used as the value for the argument
  // declared as a pointer to global or constant memory in the kernel"
  //
  // We don't know the type of the argument but it seems that the only time
  // SYCL RT would send a pointer to NULL in 'arg_value' is when the argument
  // is a NULL pointer. Treat a pointer to NULL in 'arg_value' as a NULL.
  if (argSize == sizeof(void *) && pArgValue &&
      *(void **)(const_cast<void *>(pArgValue)) == nullptr) {
    pArgValue = nullptr;
  }

  std::scoped_lock<ur_shared_mutex> guard(Mutex);

  for (auto &singleDeviceKernel : deviceKernels) {
    if (!singleDeviceKernel.has_value()) {
      continue;
    }

    ZE2UR_CALL(zeKernelSetArgumentValue,
               (singleDeviceKernel.value().hKernel.get(), argIndex, argSize,
                pArgValue));
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t ur_kernel_handle_t_::setArgPointer(
    uint32_t argIndex, const ur_kernel_arg_pointer_properties_t *pProperties,
    const void *pArgValue) {
  std::ignore = pProperties;

  // KernelSetArgValue is expecting a pointer to the argument
  return setArgValue(argIndex, sizeof(const void *), nullptr, &pArgValue);
}

ur_program_handle_t ur_kernel_handle_t_::getProgramHandle() const {
  return hProgram;
}

namespace ur::level_zero {
ur_result_t urKernelCreate(ur_program_handle_t hProgram,
                           const char *pKernelName,
                           ur_kernel_handle_t *phKernel) {
  *phKernel = new ur_kernel_handle_t_(hProgram, pKernelName);
  return UR_RESULT_SUCCESS;
}

ur_result_t urKernelRetain(
    ur_kernel_handle_t hKernel ///< [in] handle for the Kernel to retain
) {
  hKernel->RefCount.increment();
  return UR_RESULT_SUCCESS;
}

ur_result_t urKernelRelease(
    ur_kernel_handle_t hKernel ///< [in] handle for the Kernel to release
) {
  if (!hKernel->RefCount.decrementAndTest())
    return UR_RESULT_SUCCESS;

  hKernel->release();
  delete hKernel;

  return UR_RESULT_SUCCESS;
}

ur_result_t urKernelSetArgValue(
    ur_kernel_handle_t hKernel, ///< [in] handle of the kernel object
    uint32_t argIndex, ///< [in] argument index in range [0, num args - 1]
    size_t argSize,    ///< [in] size of argument type
    const ur_kernel_arg_value_properties_t
        *pProperties, ///< [in][optional] argument properties
    const void
        *pArgValue ///< [in] argument value represented as matching arg type.
) {
  TRACK_SCOPE_LATENCY("ur_kernel_handle_t_::setArgValue");
  return hKernel->setArgValue(argIndex, argSize, pProperties, pArgValue);
}

ur_result_t urKernelSetArgPointer(
    ur_kernel_handle_t hKernel, ///< [in] handle of the kernel object
    uint32_t argIndex, ///< [in] argument index in range [0, num args - 1]
    const ur_kernel_arg_pointer_properties_t
        *pProperties, ///< [in][optional] argument properties
    const void
        *pArgValue ///< [in] argument value represented as matching arg type.
) {
  TRACK_SCOPE_LATENCY("ur_kernel_handle_t_::setArgPointer");
  return hKernel->setArgPointer(argIndex, pProperties, pArgValue);
}
} // namespace ur::level_zero
