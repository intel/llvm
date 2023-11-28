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

  // Construct a program in IL or Native state.
  ur_program_handle_t_(state St, ur_context_handle_t Context, const void *Input,
                       size_t Length)
      : Context{Context},
        OwnZeModule{true}, State{St}, Code{new uint8_t[Length]},
        CodeLength{Length}, ZeModule{nullptr}, ZeBuildLog{nullptr} {
    std::memcpy(Code.get(), Input, Length);
  }

  // Construct a program in Exe or Invalid state.
  ur_program_handle_t_(state St, ur_context_handle_t Context,
                       ze_module_handle_t ZeModule,
                       ze_module_build_log_handle_t ZeBuildLog)
      : Context{Context}, OwnZeModule{true}, State{St}, ZeModule{ZeModule},
        ZeBuildLog{ZeBuildLog} {}

  // Construct a program in Exe state (interop).
  ur_program_handle_t_(state St, ur_context_handle_t Context,
                       ze_module_handle_t ZeModule, bool OwnZeModule)
      : Context{Context}, OwnZeModule{OwnZeModule}, State{St},
        ZeModule{ZeModule}, ZeBuildLog{nullptr} {}

  // Construct a program from native handle
  ur_program_handle_t_(state St, ur_context_handle_t Context,
                       ze_module_handle_t ZeModule)
      : Context{Context}, OwnZeModule{true}, State{St}, ZeModule{ZeModule},
        ZeBuildLog{nullptr} {}

  // Construct a program in Invalid state with a custom error message.
  ur_program_handle_t_(state St, ur_context_handle_t Context,
                       const std::string &ErrorMessage)
      : Context{Context}, OwnZeModule{true}, ErrorMessage{ErrorMessage},
        State{St}, ZeModule{nullptr}, ZeBuildLog{nullptr} {}

  ~ur_program_handle_t_();

  const ur_context_handle_t Context; // Context of the program.

  // Indicates if we own the ZeModule or it came from interop that
  // asked to not transfer the ownership to SYCL RT.
  const bool OwnZeModule;

  // This error message is used only in Invalid state to hold a custom error
  // message from a call to urProgramLink.
  const std::string ErrorMessage;

  state State;

  // In IL and Object states, this contains the SPIR-V representation of the
  // module.  In Native state, it contains the native code.
  std::unique_ptr<uint8_t[]> Code; // Array containing raw IL / native code.
  size_t CodeLength{0};            // Size (bytes) of the array.

  // Used only in IL and Object states.  Contains the SPIR-V specialization
  // constants as a map from the SPIR-V "SpecID" to a buffer that contains the
  // associated value.  The caller of the PI layer is responsible for
  // maintaining the storage of this buffer.
  std::unordered_map<uint32_t, const void *> SpecConstants;

  // Used only in Object state.  Contains the build flags from the last call to
  // urProgramCompile().
  std::string BuildFlags;

  // The Level Zero module handle.  Used primarily in Exe state.
  ze_module_handle_t ZeModule{};

  // Map of L0 Modules created for all the devices for which a UR Program
  // has been built.
  std::unordered_map<ze_device_handle_t, ze_module_handle_t> ZeModuleMap;

  // The Level Zero build log from the last call to zeModuleCreate().
  ze_module_build_log_handle_t ZeBuildLog{};

  // Map of L0 Module Build logs created for all the devices for which a UR
  // Program has been built.
  std::unordered_map<ze_device_handle_t, ze_module_build_log_handle_t>
      ZeBuildLogMap;
};
