//===--------- program.hpp - Level Zero Adapter ---------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "common.hpp"
#include "device.hpp"

struct ur_program_handle_t_ : _ur_object {
  // ur_program_handle_t_() {}

  typedef enum {
    // The program has been created from intermediate language (SPIR-V), but it
    // is not yet compiled.
    IL,

    // The program has been created by loading native code, but it has not yet
    // been built.  This is equivalent to an OpenCL "program executable" that
    // is loaded via clCreateProgramWithBinary().
    Native,

    // The program was notionally compiled from SPIR-V form.  However, since we
    // postpone compilation until the module is linked, the internal state
    // still represents the module as SPIR-V.
    Object,

    // The program has been built or linked, and it is represented as a Level
    // Zero module.
    Exe,

    // An error occurred during urProgramLink, but we created a
    // ur_program_handle_t
    // object anyways in order to hold the ZeBuildLog.  Note that the ZeModule
    // may or may not be nullptr in this state, depending on the error.
    Invalid
  } state;

  // A utility class that converts specialization constants into the form
  // required by the Level Zero driver.
  class SpecConstantShim {
  public:
    SpecConstantShim(ur_program_handle_t_ *Program) {
      ZeSpecConstants.numConstants = Program->SpecConstants.size();
      ZeSpecContantsIds.reserve(ZeSpecConstants.numConstants);
      ZeSpecContantsValues.reserve(ZeSpecConstants.numConstants);

      for (auto &SpecConstant : Program->SpecConstants) {
        ZeSpecContantsIds.push_back(SpecConstant.first);
        ZeSpecContantsValues.push_back(SpecConstant.second);
      }
      ZeSpecConstants.pConstantIds = ZeSpecContantsIds.data();
      ZeSpecConstants.pConstantValues = ZeSpecContantsValues.data();
    }

    const ze_module_constants_t *ze() { return &ZeSpecConstants; }

  private:
    std::vector<uint32_t> ZeSpecContantsIds;
    std::vector<const void *> ZeSpecContantsValues;
    ze_module_constants_t ZeSpecConstants;
  };

  // Construct a program in IL.
  ur_program_handle_t_(state St, ur_context_handle_t Context, const void *Input,
                       size_t Length);

  // Construct a program in NATIVE for multiple devices.
  ur_program_handle_t_(state St, ur_context_handle_t Context,
                       const uint32_t NumDevices,
                       const ur_device_handle_t *Devices,
                       const ur_program_properties_t *Properties,
                       const uint8_t **Inputs, const size_t *Lengths);

  ur_program_handle_t_(ur_context_handle_t Context);

  // Construct a program in Exe or Invalid state.
  ur_program_handle_t_(state, ur_context_handle_t Context,
                       ze_module_handle_t InteropZeModule);

  // Construct a program in Exe state (interop).
  // TODO: Currently it is not possible to get the device associated with the
  // interop module, API must be changed to either get that info from the user
  // or new API need to be added to L0 to fetch that info. Consider it
  // associated with the first device in the context.
  ur_program_handle_t_(state, ur_context_handle_t Context,
                       ze_module_handle_t InteropZeModule, bool OwnZeModule);

  // Construct a program in Invalid state with a custom error message.
  ur_program_handle_t_(state St, ur_context_handle_t Context,
                       const std::string &ErrorMessage);

  ~ur_program_handle_t_();
  void ur_release_program_resources(bool deletion);

  state getState(ze_device_handle_t ZeDevice) {
    if ((DeviceDataMap.find(ZeDevice) == DeviceDataMap.end()) &&
        InteropZeModule)
      return state::Exe;

    return DeviceDataMap[ZeDevice].State;
  }

  ze_module_handle_t getZeModuleHandle(ze_device_handle_t ZeDevice) {
    if (DeviceDataMap.find(ZeDevice) == DeviceDataMap.end())
      return InteropZeModule;

    return DeviceDataMap[ZeDevice].ZeModule;
  }

  uint8_t *getCode(ze_device_handle_t ZeDevice = nullptr) {
    if (!ZeDevice)
      return SpirvCode.get();

    if (DeviceDataMap.find(ZeDevice) == DeviceDataMap.end())
      return nullptr;

    if (DeviceDataMap[ZeDevice].State == state::IL)
      return SpirvCode.get();
    else
      return DeviceDataMap[ZeDevice].Binary.first.get();
  }

  size_t getCodeSize(ze_device_handle_t ZeDevice = nullptr) {
    if (ZeDevice == nullptr)
      return SpirvCodeLength;

    if (DeviceDataMap.find(ZeDevice) == DeviceDataMap.end())
      return 0;

    if (DeviceDataMap[ZeDevice].State == state::IL)
      return SpirvCodeLength;
    else
      return DeviceDataMap[ZeDevice].Binary.second;
  }

  ze_module_build_log_handle_t getBuildLog(ze_device_handle_t ZeDevice) {
    if (DeviceDataMap.find(ZeDevice) == DeviceDataMap.end())
      return nullptr;

    return DeviceDataMap[ZeDevice].ZeBuildLog;
  }

  void setState(ze_device_handle_t ZeDevice, state NewState) {
    DeviceDataMap[ZeDevice].State = NewState;
  }

  void setZeModule(ze_device_handle_t ZeDevice, ze_module_handle_t ZeModule) {
    DeviceDataMap[ZeDevice].ZeModule = ZeModule;
  }

  void setBuildLog(ze_device_handle_t ZeDevice,
                   ze_module_build_log_handle_t ZeBuildLog) {
    DeviceDataMap[ZeDevice].ZeBuildLog = ZeBuildLog;
  }

  void setBuildOptions(ze_device_handle_t ZeDevice,
                       const std::string &Options) {
    DeviceDataMap[ZeDevice].BuildFlags = Options;
  }

  void appendBuildOptions(ze_device_handle_t ZeDevice,
                          const std::string &Options) {
    DeviceDataMap[ZeDevice].BuildFlags += Options;
  }

  std::string &getBuildOptions(ze_device_handle_t ZeDevice) {
    return DeviceDataMap[ZeDevice].BuildFlags;
  }

  // Tracks the release state of the program handle to determine if the
  // internal handle needs to be released.
  bool resourcesReleased = false;

  const ur_context_handle_t Context; // Context of the program.

  // Properties used for the Native Build
  const ur_program_properties_t *NativeProperties;

  // Indicates if we own the ZeModule or it came from interop that
  // asked to not transfer the ownership to SYCL RT.
  const bool OwnZeModule;

  // This error message is used only in Invalid state to hold a custom error
  // message from a call to urProgramLink.
  const std::string ErrorMessage;

  // Used only in IL and Object states.  Contains the SPIR-V specialization
  // constants as a map from the SPIR-V "SpecID" to a buffer that contains the
  // associated value.  The caller of the PI layer is responsible for
  // maintaining the storage of this buffer.
  std::unordered_map<uint32_t, const void *> SpecConstants;

  // Keep the vector of devices associated with the program.
  // It is populated at program creation and used to provide information for the
  // descriptors like UR_PROGRAM_INFO_DEVICES, UR_PROGRAM_INFO_BINARY_SIZES,
  // UR_PROGRAM_INFO_BINARIES as they are supposed to return information in the
  // same order. I.e. the first binary in the array returned by
  // UR_PROGRAM_INFO_BINARIES is supposed to be associated with the first device
  // in the returned array of devices for UR_PROGRAM_INFO_DEVICES. Same for
  // UR_PROGRAM_INFO_BINARY_SIZES.
  const std::vector<ur_device_handle_t> AssociatedDevices;

private:
  struct DeviceData {
    // Log from the result of building the program for the device using
    // zeModuleCreate().
    ze_module_build_log_handle_t ZeBuildLog = nullptr;

    // The Level Zero module handle for the device. Used primarily in Exe state.
    ze_module_handle_t ZeModule = nullptr;

    // In Native state, contains the pair of the binary code for the device and
    // its length in bytes.
    std::pair<std::unique_ptr<uint8_t[]>, size_t> Binary{nullptr, 0};

    // Build flags used for building the program for the device.
    // May be different for different devices, for example, if
    // urProgramCompileExp was called multiple times with different build flags
    // for different devices.
    std::string BuildFlags{};

    // State of the program for the device.
    state State{};
  };

  std::unordered_map<ze_device_handle_t, DeviceData> DeviceDataMap;

  // In IL and Object states, this contains the SPIR-V representation of the
  // module.
  std::unique_ptr<uint8_t[]> SpirvCode; // Array containing raw IL code.
  size_t SpirvCodeLength = 0;           // Size (bytes) of the array.

  // The Level Zero module handle for interoperability.
  // This module handle is either initialized with the handle provided to
  // interoperability UR API, or with one of the handles after building the
  // program. This handle is returned by UR API which allows to get the native
  // handle from the program.
  // TODO: Currently interoparability UR API does not support multiple devices.
  ze_module_handle_t InteropZeModule = nullptr;
};
