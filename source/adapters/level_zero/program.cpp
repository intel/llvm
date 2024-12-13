//===--------- program.cpp - Level Zero Adapter ---------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "program.hpp"
#include "device.hpp"
#include "logger/ur_logger.hpp"
#include "ur_interface_loader.hpp"

#ifdef UR_ADAPTER_LEVEL_ZERO_V2
#include "v2/context.hpp"
#else
#include "context.hpp"
#endif

extern "C" {
// Check to see if a Level Zero module has any unresolved symbols.
//
// @param ZeModule    The module handle to check.
// @param ZeBuildLog  If there are unresolved symbols, this build log handle is
//                     modified to receive information telling which symbols
//                     are unresolved.
//
// @return ZE_RESULT_ERROR_MODULE_LINK_FAILURE indicates there are unresolved
//  symbols.  ZE_RESULT_SUCCESS indicates all symbols are resolved.  Any other
//  value indicates there was an error and we cannot tell if symbols are
//  resolved.
static ze_result_t
checkUnresolvedSymbols(ze_module_handle_t ZeModule,
                       ze_module_build_log_handle_t *ZeBuildLog) {

  // First check to see if the module has any imported symbols.  If there are
  // no imported symbols, it's not possible to have any unresolved symbols.  We
  // do this check first because we assume it's faster than the call to
  // zeModuleDynamicLink below.
  ZeStruct<ze_module_properties_t> ZeModuleProps;
  ze_result_t ZeResult =
      ZE_CALL_NOCHECK(zeModuleGetProperties, (ZeModule, &ZeModuleProps));
  if (ZeResult != ZE_RESULT_SUCCESS)
    return ZeResult;

  // If there are imported symbols, attempt to "link" the module with itself.
  // As a side effect, this will return the error
  // ZE_RESULT_ERROR_MODULE_LINK_FAILURE if there are any unresolved symbols.
  if (ZeModuleProps.flags & ZE_MODULE_PROPERTY_FLAG_IMPORTS) {
    return ZE_CALL_NOCHECK(zeModuleDynamicLink, (1, &ZeModule, ZeBuildLog));
  }
  return ZE_RESULT_SUCCESS;
}
} // extern "C"

namespace ur::level_zero {

ur_result_t urProgramCreateWithIL(
    ur_context_handle_t Context, ///< [in] handle of the context instance
    const void *IL,              ///< [in] pointer to IL binary.
    size_t Length,               ///< [in] length of `pIL` in bytes.
    const ur_program_properties_t
        *Properties, ///< [in][optional] pointer to program creation properties.
    ur_program_handle_t
        *Program ///< [out] pointer to handle of program object created.
) {
  std::ignore = Properties;
  UR_ASSERT(Context, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(IL && Program, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  try {
    ur_program_handle_t_ *UrProgram =
        new ur_program_handle_t_(ur_program_handle_t_::IL, Context, IL, Length);
    *Program = reinterpret_cast<ur_program_handle_t>(UrProgram);
  } catch (const std::bad_alloc &) {
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t urProgramCreateWithBinary(
    ur_context_handle_t hContext, ///< [in] handle of the context instance
    uint32_t numDevices,          ///< [in] number of devices
    ur_device_handle_t
        *phDevices,   ///< [in][range(0, numDevices)] a pointer to a list of
                      ///< device handles. The binaries are loaded for devices
                      ///< specified in this list.
    size_t *pLengths, ///< [in][range(0, numDevices)] array of sizes of program
                      ///< binaries specified by `pBinaries` (in bytes).
    const uint8_t *
        *ppBinaries, ///< [in][range(0, numDevices)] pointer to program binaries
                     ///< to be loaded for devices specified by `phDevices`.
    const ur_program_properties_t *
        pProperties, ///< [in][optional] pointer to program creation properties.
    ur_program_handle_t
        *phProgram ///< [out] pointer to handle of Program object created.
) {
  // In OpenCL, clCreateProgramWithBinary() can be used to load any of the
  // following: "program executable", "compiled program", or "library of
  // compiled programs".  In addition, the loaded program can be either
  // IL (SPIR-v) or native device code.  For now, we assume that
  // urProgramCreateWithBinary() is only used to load a "program executable"
  // as native device code.
  // If we wanted to support all the same cases as OpenCL, we would need to
  // somehow examine the binary image to distinguish the cases.  Alternatively,
  // we could change the PI interface and have the caller pass additional
  // information to distinguish the cases.
  try {
    for (uint32_t i = 0; i < numDevices; i++) {
      UR_ASSERT(ppBinaries[i] || !pLengths[0], UR_RESULT_ERROR_INVALID_VALUE);
      UR_ASSERT(hContext->isValidDevice(phDevices[i]),
                UR_RESULT_ERROR_INVALID_DEVICE);
    }
    ur_program_handle_t_ *UrProgram = new ur_program_handle_t_(
        ur_program_handle_t_::Native, hContext, numDevices, phDevices,
        pProperties, ppBinaries, pLengths);
    *phProgram = reinterpret_cast<ur_program_handle_t>(UrProgram);
    return UR_RESULT_SUCCESS;
  } catch (const std::bad_alloc &) {
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
}

ur_result_t urProgramBuild(
    ur_context_handle_t Context, ///< [in] handle of the context instance.
    ur_program_handle_t Program, ///< [in] Handle of the program to build.
    const char *Options          ///< [in][optional] pointer to build options
                                 ///< null-terminated string.
) {
  std::vector<ur_device_handle_t> Devices = Context->getDevices();
  return ur::level_zero::urProgramBuildExp(Program, Devices.size(),
                                           Devices.data(), Options);
}

ur_result_t urProgramBuildExp(
    ur_program_handle_t hProgram,  ///< [in] Handle of the program to build.
    uint32_t numDevices,           ///< [in] number of devices
    ur_device_handle_t *phDevices, ///< [in][range(0, numDevices)] pointer to
                                   ///< array of device handles
    const char *pOptions           ///< [in][optional] pointer to build options
                                   ///< null-terminated string.
) {
  // TODO
  // Check if device belongs to associated context.
  // UR_ASSERT(Program->Context, UR_RESULT_ERROR_INVALID_PROGRAM);
  // UR_ASSERT(Program->Context->isValidDevice(Devices[0]),
  // UR_RESULT_ERROR_INVALID_VALUE);

  std::scoped_lock<ur_shared_mutex> Guard(hProgram->Mutex);

  ur_program_handle_t_::SpecConstantShim Shim(hProgram);

  // if large allocations are selected, then pass
  // ze-opt-greater-than-4GB-buffer-required to disable
  // stateful optimizations and be able to use larger than
  // 4GB allocations on these kernels.
  std::string ZeBuildOptions{};
  if (pOptions) {
    ZeBuildOptions += pOptions;
  }

  if (phDevices[0]->useRelaxedAllocationLimits()) {
    ZeBuildOptions += " -ze-opt-greater-than-4GB-buffer-required";
  }

  ur_result_t Result = UR_RESULT_SUCCESS;
  for (uint32_t i = 0; i < numDevices; i++) {
    ZeStruct<ze_module_desc_t> ZeModuleDesc;
    ZeModuleDesc.pBuildFlags = ZeBuildOptions.c_str();
    ZeModuleDesc.pConstants = Shim.ze();
    ze_device_handle_t ZeDevice = phDevices[i]->ZeDevice;
    auto State = hProgram->getState(ZeDevice);

    // We don't want to rebuild the module if it was already built.
    if (State == ur_program_handle_t_::Exe)
      continue;

    // It is legal to build a program created from either IL or from native
    // device code.
    if (State != ur_program_handle_t_::IL &&
        State != ur_program_handle_t_::Native)
      return UR_RESULT_ERROR_INVALID_OPERATION;

    // We should have either IL or native device code.
    auto Code = hProgram->getCode(ZeDevice);
    UR_ASSERT(Code, UR_RESULT_ERROR_INVALID_PROGRAM);

    ZeModuleDesc.format = (State == ur_program_handle_t_::IL)
                              ? ZE_MODULE_FORMAT_IL_SPIRV
                              : ZE_MODULE_FORMAT_NATIVE;
    ZeModuleDesc.inputSize = hProgram->getCodeSize(ZeDevice);
    ZeModuleDesc.pInputModule = Code;
    ze_context_handle_t ZeContext = hProgram->Context->getZeHandle();
    ze_module_handle_t ZeModuleHandle = nullptr;
    ze_module_build_log_handle_t ZeBuildLog{};

    ze_result_t ZeResult =
        ZE_CALL_NOCHECK(zeModuleCreate, (ZeContext, ZeDevice, &ZeModuleDesc,
                                         &ZeModuleHandle, &ZeBuildLog));
    hProgram->setState(ZeDevice, ur_program_handle_t_::Exe);
    if (ZeResult != ZE_RESULT_SUCCESS) {
      // We adjust ur_program below to avoid attempting to release zeModule when
      // RT calls urProgramRelease().
      hProgram->setState(ZeDevice, ur_program_handle_t_::Invalid);
      Result = ze2urResult(ZeResult);
      if (ZeModuleHandle) {
        ZE_CALL_NOCHECK(zeModuleDestroy, (ZeModuleHandle));
        ZeModuleHandle = nullptr;
      }
    } else {
      // The call to zeModuleCreate does not report an error if there are
      // unresolved symbols because it thinks these could be resolved later via
      // a call to zeModuleDynamicLink.  However, modules created with
      // urProgramBuild are supposed to be fully linked and ready to use.
      // Therefore, do an extra check now for unresolved symbols.
      ZeResult = checkUnresolvedSymbols(ZeModuleHandle, &ZeBuildLog);
      if (ZeResult != ZE_RESULT_SUCCESS) {
        hProgram->setState(ZeDevice, ur_program_handle_t_::Invalid);
        Result = (ZeResult == ZE_RESULT_ERROR_MODULE_LINK_FAILURE)
                     ? UR_RESULT_ERROR_PROGRAM_BUILD_FAILURE
                     : ze2urResult(ZeResult);
        if (ZeModuleHandle) {
          ZE_CALL_NOCHECK(zeModuleDestroy, (ZeModuleHandle));
          ZeModuleHandle = nullptr;
        }
      }
      hProgram->setZeModule(ZeDevice, ZeModuleHandle);
    }
    hProgram->setBuildLog(ZeDevice, ZeBuildLog);
  }

  return Result;
}

ur_result_t urProgramCompileExp(
    ur_program_handle_t
        hProgram,        ///< [in][out] handle of the program to compile.
    uint32_t numDevices, ///< [in] number of devices
    ur_device_handle_t *phDevices, ///< [in][range(0, numDevices)] pointer to
                                   ///< array of device handles
    const char *pOptions           ///< [in][optional] pointer to build options
                                   ///< null-terminated string.
) {
  std::scoped_lock<ur_shared_mutex> Guard(hProgram->Mutex);
  // Check that state is IL for all devices in the context and set the state to
  // Object.
  for (uint32_t I = 0; I < numDevices; I++) {
    auto ZeDevice = phDevices[I]->ZeDevice;
    // It's only valid to compile a program created from IL (we don't support
    // programs created from source code).
    //
    // The OpenCL spec says that the header parameters are ignored when
    // compiling IL programs, so we don't validate them.
    if (hProgram->getState(ZeDevice) != ur_program_handle_t_::IL)
      return UR_RESULT_ERROR_INVALID_OPERATION;
    hProgram->setState(ZeDevice, ur_program_handle_t_::Object);
    // We don't compile anything now.  Instead, we delay compilation until
    // urProgramLink, where we do both compilation and linking as a single step.
    // This produces better code because the driver can do cross-module
    // optimizations.  Therefore, we just remember the compilation flags, so we
    // can use them later.
    if (pOptions) {
      hProgram->setBuildOptions(ZeDevice, pOptions);
      // if large allocations are selected, then pass
      // ze-opt-greater-than-4GB-buffer-required to disable
      // stateful optimizations and be able to use larger than
      // 4GB allocations on these kernels.
      if (phDevices[I]->useRelaxedAllocationLimits()) {
        hProgram->appendBuildOptions(
            ZeDevice, " -ze-opt-greater-than-4GB-buffer-required");
      }
    }
    hProgram->setState(ZeDevice, ur_program_handle_t_::Object);
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t urProgramCompile(
    ur_context_handle_t Context, ///< [in] handle of the context instance.
    ur_program_handle_t
        Program,        ///< [in][out] handle of the program to compile.
    const char *Options ///< [in][optional] pointer to build options
                        ///< null-terminated string.
) {
  auto devices = Context->getDevices();
  return ur::level_zero::urProgramCompileExp(Program, devices.size(),
                                             devices.data(), Options);
}

ur_result_t urProgramLink(
    ur_context_handle_t Context, ///< [in] handle of the context instance.
    uint32_t Count, ///< [in] number of program handles in `phPrograms`.
    const ur_program_handle_t *Programs, ///< [in][range(0, count)] pointer to
                                         ///< array of program handles.
    const char *Options, ///< [in][optional] pointer to linker options
                         ///< null-terminated string.
    ur_program_handle_t
        *Program ///< [out] pointer to handle of program object created.
) {
  std::vector<ur_device_handle_t> Devices = Context->getDevices();
  return ur::level_zero::urProgramLinkExp(Context, Devices.size(),
                                          Devices.data(), Count, Programs,
                                          Options, Program);
}

ur_result_t urProgramLinkExp(
    ur_context_handle_t hContext,  ///< [in] handle of the context instance.
    uint32_t numDevices,           ///< [in] number of devices
    ur_device_handle_t *phDevices, ///< [in][range(0, numDevices)] pointer to
                                   ///< array of device handles
    uint32_t count, ///< [in] number of program handles in `phPrograms`.
    const ur_program_handle_t *phPrograms, ///< [in][range(0, count)] pointer to
                                           ///< array of program handles.
    const char *pOptions, ///< [in][optional] pointer to linker options
                          ///< null-terminated string.
    ur_program_handle_t
        *phProgram ///< [out] pointer to handle of program object created.
) {
  if (nullptr != phProgram) {
    *phProgram = nullptr;
  }
  for (uint32_t i = 0; i < numDevices; i++) {
    UR_ASSERT(hContext->isValidDevice(phDevices[i]),
              UR_RESULT_ERROR_INVALID_DEVICE);
  }

  // We do not support any link flags at this time because the Level Zero API
  // does not have any way to pass flags that are specific to linking.
  if (pOptions && *pOptions != '\0') {
    std::string ErrorMessage(
        "Level Zero does not support kernel link flags: \"");
    ErrorMessage.append(pOptions);
    ErrorMessage.push_back('\"');
    ur_program_handle_t_ *UrProgram = new ur_program_handle_t_(
        ur_program_handle_t_::Invalid, hContext, ErrorMessage);
    *phProgram = reinterpret_cast<ur_program_handle_t>(UrProgram);
    return UR_RESULT_ERROR_PROGRAM_LINK_FAILURE;
  }

  ur_result_t UrResult = UR_RESULT_SUCCESS;
  try {
    // Acquire a "shared" lock on each of the input programs, and also validate
    // that they are all in Object state for each device in the input list.
    //
    // There is no danger of deadlock here even if two threads call
    // urProgramLink simultaneously with the same input programs in a different
    // order.  If we were acquiring these with "exclusive" access, this could
    // lead to a classic lock ordering deadlock.  However, there is no such
    // deadlock potential with "shared" access.  There could also be a deadlock
    // potential if there was some other code that holds more than one of these
    // locks simultaneously with "exclusive" access.  However, there is no such
    // code like that, so this is also not a danger.
    std::vector<std::shared_lock<ur_shared_mutex>> Guards(count);
    for (uint32_t I = 0; I < count; I++) {
      std::shared_lock<ur_shared_mutex> Guard(phPrograms[I]->Mutex);
      Guards[I].swap(Guard);
      for (uint32_t DeviceIndex = 0; DeviceIndex < numDevices; DeviceIndex++) {
        auto Device = phDevices[DeviceIndex];
        if (phPrograms[I]->getState(Device->ZeDevice) !=
            ur_program_handle_t_::Object) {
          return UR_RESULT_ERROR_INVALID_OPERATION;
        }
      }
    }

    // Previous calls to urProgramCompile did not actually compile the SPIR-V.
    // Instead, we postpone compilation until this point, when all the modules
    // are linked together.  By doing compilation and linking together, the JIT
    // compiler is able see all modules and do cross-module optimizations.
    //
    // Construct a ze_module_program_exp_desc_t which contains information about
    // all of the modules that will be linked together.
    ZeStruct<ze_module_program_exp_desc_t> ZeExtModuleDesc;
    std::vector<size_t> CodeSizes(count);
    std::vector<const uint8_t *> CodeBufs(count);
    std::vector<const char *> BuildFlagPtrs(count);
    std::vector<const ze_module_constants_t *> SpecConstPtrs(count);
    std::vector<ur_program_handle_t_::SpecConstantShim> SpecConstShims;
    SpecConstShims.reserve(count);

    for (uint32_t I = 0; I < count; I++) {
      ur_program_handle_t Program = phPrograms[I];
      CodeSizes[I] = Program->getCodeSize();
      CodeBufs[I] = Program->getCode();
      SpecConstShims.emplace_back(Program);
      SpecConstPtrs[I] = SpecConstShims[I].ze();
    }

    ZeExtModuleDesc.count = count;
    ZeExtModuleDesc.inputSizes = CodeSizes.data();
    ZeExtModuleDesc.pInputModules = CodeBufs.data();
    ZeExtModuleDesc.pConstants = SpecConstPtrs.data();

    ZeStruct<ze_module_desc_t> ZeModuleDesc;
    ZeModuleDesc.pNext = &ZeExtModuleDesc;
    ZeModuleDesc.format = ZE_MODULE_FORMAT_IL_SPIRV;

    // This works around a bug in the Level Zero driver.  When "ZE_DEBUG=-1",
    // the driver does validation of the API calls, and it expects
    // "pInputModule" to be non-NULL and "inputSize" to be non-zero.  This
    // validation is wrong when using the "ze_module_program_exp_desc_t"
    // extension because those fields are supposed to be ignored.  As a
    // workaround, set both fields to 1.
    //
    // TODO: Remove this workaround when the driver is fixed.
    ZeModuleDesc.pInputModule = reinterpret_cast<const uint8_t *>(1);
    ZeModuleDesc.inputSize = 1;

    // We need a Level Zero extension to compile multiple programs together into
    // a single Level Zero module.  However, we don't need that extension if
    // there happens to be only one input program.
    //
    // The "|| (NumInputPrograms == 1)" term is a workaround for a bug in the
    // Level Zero driver.  The driver's "ze_module_program_exp_desc_t"
    // extension should work even in the case when there is just one input
    // module.  However, there is currently a bug in the driver that leads to a
    // crash.  As a workaround, do not use the extension when there is one
    // input module.
    //
    // TODO: Remove this workaround when the driver is fixed.
    if (!phDevices[0]->Platform->ZeDriverModuleProgramExtensionFound ||
        (count == 1)) {
      if (count == 1) {
        ZeModuleDesc.pNext = nullptr;
        ZeModuleDesc.inputSize = ZeExtModuleDesc.inputSizes[0];
        ZeModuleDesc.pInputModule = ZeExtModuleDesc.pInputModules[0];
        ZeModuleDesc.pConstants = ZeExtModuleDesc.pConstants[0];
      } else {
        logger::error(
            "urProgramLink: level_zero driver does not have static linking "
            "support.");
        return UR_RESULT_ERROR_INVALID_VALUE;
      }
    }

    ur_program_handle_t_ *UrProgram = new ur_program_handle_t_(hContext);
    *phProgram = reinterpret_cast<ur_program_handle_t>(UrProgram);
    for (uint32_t i = 0; i < numDevices; i++) {

      // Call the Level Zero API to compile, link, and create the module.
      ze_device_handle_t ZeDevice = phDevices[i]->ZeDevice;
      ze_context_handle_t ZeContext = hContext->getZeHandle();
      ze_module_handle_t ZeModule = nullptr;
      ze_module_build_log_handle_t ZeBuildLog = nullptr;

      // Build flags may be different for different devices, so handle them
      // here. Clear values of the previous device first.
      BuildFlagPtrs.clear();
      std::vector<std::string> TemporaryOptionsStrings;
      for (uint32_t I = 0; I < count; I++) {
        TemporaryOptionsStrings.push_back(
            phPrograms[I]->getBuildOptions(ZeDevice));
        BuildFlagPtrs.push_back(TemporaryOptionsStrings.back().c_str());
      }
      ZeExtModuleDesc.pBuildFlags = BuildFlagPtrs.data();
      if (count == 1)
        ZeModuleDesc.pBuildFlags = ZeExtModuleDesc.pBuildFlags[0];

      ze_result_t ZeResult =
          ZE_CALL_NOCHECK(zeModuleCreate, (ZeContext, ZeDevice, &ZeModuleDesc,
                                           &ZeModule, &ZeBuildLog));

      // We still create a ur_program_handle_t_ object even if there is a
      // BUILD_FAILURE because we need the object to hold the ZeBuildLog.  There
      // is no build log created for other errors, so we don't create an object.
      UrResult = ze2urResult(ZeResult);
      if (ZeResult != ZE_RESULT_SUCCESS &&
          ZeResult != ZE_RESULT_ERROR_MODULE_BUILD_FAILURE) {
        return ze2urResult(ZeResult);
      }

      // The call to zeModuleCreate does not report an error if there are
      // unresolved symbols because it thinks these could be resolved later via
      // a call to zeModuleDynamicLink.  However, modules created with
      // piProgramLink are supposed to be fully linked and ready to use.
      // Therefore, do an extra check now for unresolved symbols.  Note that we
      // still create a ur_program_handle_t_ if there are unresolved symbols
      // because the ZeBuildLog tells which symbols are unresolved.
      if (ZeResult == ZE_RESULT_SUCCESS) {
        ZeResult = checkUnresolvedSymbols(ZeModule, &ZeBuildLog);
        if (ZeResult != ZE_RESULT_SUCCESS) {
          return ze2urResult(ZeResult);
        }
      }
      UrProgram->setZeModule(ZeDevice, ZeModule);
      UrProgram->setBuildLog(ZeDevice, ZeBuildLog);
      UrProgram->setState(ZeDevice, (UrResult == UR_RESULT_SUCCESS)
                                        ? ur_program_handle_t_::Exe
                                        : ur_program_handle_t_::Invalid);
    }
  } catch (const std::bad_alloc &) {
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
  return UrResult;
}

ur_result_t urProgramRetain(
    ur_program_handle_t Program ///< [in] handle for the Program to retain
) {
  Program->RefCount.increment();
  return UR_RESULT_SUCCESS;
}

ur_result_t urProgramRelease(
    ur_program_handle_t Program ///< [in] handle for the Program to release
) {
  if (!Program->RefCount.decrementAndTest())
    return UR_RESULT_SUCCESS;

  delete Program;

  return UR_RESULT_SUCCESS;
}

// Function gets characters between delimeter's in str
// then checks if they are equal to the sub_str.
// returns true if there is at least one instance
// returns false if there are no instances of the name
static bool is_in_separated_string(const std::string &str, char delimiter,
                                   const std::string &sub_str) {
  size_t beg = 0;
  size_t length = 0;
  for (const auto &x : str) {
    if (x == delimiter) {
      if (str.substr(beg, length) == sub_str)
        return true;

      beg += length + 1;
      length = 0;
      continue;
    }
    length++;
  }
  if (length != 0)
    if (str.substr(beg, length) == sub_str)
      return true;

  return false;
}

ur_result_t urProgramGetFunctionPointer(
    ur_device_handle_t
        Device, ///< [in] handle of the device to retrieve pointer for.
    ur_program_handle_t
        Program, ///< [in] handle of the program to search for function in.
                 ///< The program must already be built to the specified
                 ///< device, or otherwise
                 ///< ::UR_RESULT_ERROR_INVALID_PROGRAM_EXECUTABLE is returned.
    const char *FunctionName, ///< [in] A null-terminates string denoting the
                              ///< mangled function name.
    void **FunctionPointerRet ///< [out] Returns the pointer to the function if
                              ///< it is found in the program.
) {
  std::shared_lock<ur_shared_mutex> Guard(Program->Mutex);
  if (Program->getState(Device->ZeDevice) != ur_program_handle_t_::Exe) {
    return UR_RESULT_ERROR_INVALID_PROGRAM_EXECUTABLE;
  }

  ze_module_handle_t ZeModule = Program->getZeModuleHandle(Device->ZeDevice);
  ze_result_t ZeResult = ZE_CALL_NOCHECK(
      zeModuleGetFunctionPointer, (ZeModule, FunctionName, FunctionPointerRet));

  // zeModuleGetFunctionPointer currently fails for all
  // kernels regardless of if the kernel exist or not
  // with ZE_RESULT_ERROR_INVALID_ARGUMENT
  // TODO: remove when this is no longer the case
  // If zeModuleGetFunctionPointer returns invalid argument,
  // fallback to searching through kernel list and return
  // PI_ERROR_FUNCTION_ADDRESS_IS_NOT_AVAILABLE if the function exists
  // or PI_ERROR_INVALID_KERNEL_NAME if the function does not exist.
  // FunctionPointerRet should always be 0
  if (ZeResult == ZE_RESULT_ERROR_INVALID_ARGUMENT) {
    size_t Size;
    *FunctionPointerRet = 0;
    UR_CALL(ur::level_zero::urProgramGetInfo(
        Program, UR_PROGRAM_INFO_KERNEL_NAMES, 0, nullptr, &Size));

    std::string ClResult(Size, ' ');
    UR_CALL(ur::level_zero::urProgramGetInfo(
        Program, UR_PROGRAM_INFO_KERNEL_NAMES, ClResult.size(), &ClResult[0],
        nullptr));

    // Get rid of the null terminator and search for kernel_name
    // If function can be found return error code to indicate it
    // exists
    ClResult.pop_back();
    if (is_in_separated_string(ClResult, ';', std::string(FunctionName)))
      return UR_RESULT_ERROR_FUNCTION_ADDRESS_NOT_AVAILABLE;

    return UR_RESULT_ERROR_INVALID_KERNEL_NAME;
  }

  if (ZeResult == ZE_RESULT_ERROR_INVALID_FUNCTION_NAME) {
    *FunctionPointerRet = 0;
    return UR_RESULT_ERROR_INVALID_KERNEL_NAME;
  }

  return ze2urResult(ZeResult);
}

ur_result_t urProgramGetGlobalVariablePointer(
    ur_device_handle_t
        Device, ///< [in] handle of the device to retrieve the pointer for.
    ur_program_handle_t
        Program, ///< [in] handle of the program where the global variable is.
    const char *GlobalVariableName, ///< [in] mangled name of the global
                                    ///< variable to retrieve the pointer for.
    size_t *GlobalVariableSizeRet,  ///< [out][optional] Returns the size of the
                                    ///< global variable if it is found in the
                                    ///< program.
    void **GlobalVariablePointerRet ///< [out] Returns the pointer to the global
                                    ///< variable if it is found in the program.
) {
  std::scoped_lock<ur_shared_mutex> lock(Program->Mutex);
  if (Program->getState(Device->ZeDevice) != ur_program_handle_t_::Exe) {
    return UR_RESULT_ERROR_INVALID_PROGRAM_EXECUTABLE;
  }

  ze_module_handle_t ZeModuleEntry{};
  ZeModuleEntry = Program->getZeModuleHandle(Device->ZeDevice);

  ze_result_t ZeResult =
      zeModuleGetGlobalPointer(ZeModuleEntry, GlobalVariableName,
                               GlobalVariableSizeRet, GlobalVariablePointerRet);

  if (ZeResult == ZE_RESULT_ERROR_UNSUPPORTED_FEATURE) {
    return UR_RESULT_ERROR_INVALID_VALUE;
  }

  return ze2urResult(ZeResult);
}

ur_result_t urProgramGetInfo(
    ur_program_handle_t Program, ///< [in] handle of the Program object
    ur_program_info_t PropName,  ///< [in] name of the Program property to query
    size_t PropSize,             ///< [in] the size of the Program property.
    void *ProgramInfo,  ///< [in,out][optional] array of bytes of holding the
                        ///< program info property. If propSize is not equal to
                        ///< or greater than the real number of bytes needed to
                        ///< return the info then the
                        ///< ::UR_RESULT_ERROR_INVALID_SIZE error is returned
                        ///< and pProgramInfo is not used.
    size_t *PropSizeRet ///< [out][optional] pointer to the actual size in
                        ///< bytes of data copied to propName.
) {
  UrReturnHelper ReturnValue(PropSize, ProgramInfo, PropSizeRet);

  switch (PropName) {
  case UR_PROGRAM_INFO_REFERENCE_COUNT:
    return ReturnValue(uint32_t{Program->RefCount.load()});
  case UR_PROGRAM_INFO_CONTEXT:
    return ReturnValue(Program->Context);
  case UR_PROGRAM_INFO_NUM_DEVICES:
    return ReturnValue(
        uint32_t{ur_cast<uint32_t>(Program->AssociatedDevices.size())});
  case UR_PROGRAM_INFO_DEVICES:
    return ReturnValue(Program->AssociatedDevices.data(),
                       Program->AssociatedDevices.size());
  case UR_PROGRAM_INFO_BINARY_SIZES: {
    std::shared_lock<ur_shared_mutex> Guard(Program->Mutex);
    std::vector<size_t> binarySizes;
    for (auto Device : Program->AssociatedDevices) {
      auto State = Program->getState(Device->ZeDevice);
      if (State == ur_program_handle_t_::Native) {
        binarySizes.push_back(Program->getCodeSize(Device->ZeDevice));
        continue;
      }
      if (State == ur_program_handle_t_::IL ||
          State == ur_program_handle_t_::Object) {
        // We don't have a binary for this device, so return 0.
        binarySizes.push_back(0);
        continue;
      } else if (State == ur_program_handle_t_::Exe) {
        auto ZeModule = Program->getZeModuleHandle(Device->ZeDevice);
        if (!ZeModule)
          return UR_RESULT_ERROR_INVALID_PROGRAM;

        size_t binarySize = 0;
        ZE2UR_CALL(zeModuleGetNativeBinary, (ZeModule, &binarySize, nullptr));
        binarySizes.push_back(binarySize);
      } else {
        return UR_RESULT_ERROR_INVALID_PROGRAM;
      }
    }
    return ReturnValue(binarySizes.data(), binarySizes.size());
  }
  case UR_PROGRAM_INFO_BINARIES: {
    // The caller sets "ParamValue" to an array of pointers, one for each
    // device.
    uint8_t **PBinary = nullptr;
    if (ProgramInfo) {
      PBinary = ur_cast<uint8_t **>(ProgramInfo);
      if (!PBinary[0]) {
        break;
      }
    }
    std::shared_lock<ur_shared_mutex> Guard(Program->Mutex);
    uint8_t *NativeBinaryPtr = nullptr;
    if (PBinary) {
      NativeBinaryPtr = PBinary[0];
    }

    size_t SzBinary = 0;
    for (uint32_t deviceIndex = 0;
         deviceIndex < Program->AssociatedDevices.size(); deviceIndex++) {
      auto ZeDevice = Program->AssociatedDevices[deviceIndex]->ZeDevice;
      auto State = Program->getState(ZeDevice);
      if (State == ur_program_handle_t_::Native) {
        // If Program was created from Native code then return that code.
        if (PBinary) {
          std::memcpy(PBinary[deviceIndex], Program->getCode(ZeDevice),
                      Program->getCodeSize(ZeDevice));
        }
        SzBinary += Program->getCodeSize(ZeDevice);
        continue;
      }
      if (State == ur_program_handle_t_::IL ||
          State == ur_program_handle_t_::Object) {
        // We don't have a binary for this device, so don't update the output
        // pointer to the binary, only set return size to 0.
        if (PropSizeRet)
          *PropSizeRet = 0;
      } else if (State == ur_program_handle_t_::Exe) {
        auto ZeModule = Program->getZeModuleHandle(ZeDevice);
        if (!ZeModule) {
          return UR_RESULT_ERROR_INVALID_PROGRAM;
        }
        size_t binarySize = 0;
        if (PBinary) {
          NativeBinaryPtr = PBinary[deviceIndex];
        }
        // If the caller is using a Program which is a built binary, then
        // the program returned will either be a single module if this is a
        // native binary or the native binary for each device will be returned.
        ZE2UR_CALL(zeModuleGetNativeBinary,
                   (ZeModule, &binarySize, NativeBinaryPtr));
        SzBinary += binarySize;
      } else {
        return UR_RESULT_ERROR_INVALID_PROGRAM;
      }
    }
    if (PropSizeRet)
      *PropSizeRet = SzBinary;
    break;
  }
  case UR_PROGRAM_INFO_NUM_KERNELS: {
    std::shared_lock<ur_shared_mutex> Guard(Program->Mutex);
    uint32_t NumKernels = 0;
    ze_module_handle_t ZeModule = nullptr;
    // Find the first module in exe state.
    for (const auto &Device : Program->AssociatedDevices) {
      if (Program->getState(Device->ZeDevice) == ur_program_handle_t_::Exe) {
        ZeModule = Program->getZeModuleHandle(Device->ZeDevice);
        break;
      }
    }

    // If none of the modules are in exe state, return error.
    if (!ZeModule)
      return UR_RESULT_ERROR_INVALID_PROGRAM;

    ZE2UR_CALL(zeModuleGetKernelNames, (ZeModule, &NumKernels, nullptr));
    return ReturnValue(size_t{NumKernels});
  }
  case UR_PROGRAM_INFO_KERNEL_NAMES:
    try {
      std::shared_lock<ur_shared_mutex> Guard(Program->Mutex);
      ze_module_handle_t ZeModule = nullptr;
      // Find the first module in exe state.
      for (const auto &Device : Program->AssociatedDevices) {
        if (Program->getState(Device->ZeDevice) == ur_program_handle_t_::Exe) {
          ZeModule = Program->getZeModuleHandle(Device->ZeDevice);
          break;
        }
      }

      // If none of the modules are in exe state, return error.
      if (!ZeModule)
        return UR_RESULT_ERROR_INVALID_PROGRAM;

      std::string PINames{""};
      uint32_t Count = 0;
      std::unique_ptr<const char *[]> PNames;
      ZE2UR_CALL(zeModuleGetKernelNames, (ZeModule, &Count, nullptr));
      PNames = std::make_unique<const char *[]>(Count);
      ZE2UR_CALL(zeModuleGetKernelNames, (ZeModule, &Count, PNames.get()));
      for (uint32_t I = 0; I < Count; ++I) {
        PINames += (I > 0 ? ";" : "");
        PINames += PNames[I];
      }
      return ReturnValue(PINames.c_str());
    } catch (const std::bad_alloc &) {
      return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    } catch (...) {
      return UR_RESULT_ERROR_UNKNOWN;
    }
  case UR_PROGRAM_INFO_IL:
    return ReturnValue(Program->getCode(), Program->getCodeSize());
  default:
    return UR_RESULT_ERROR_INVALID_ENUMERATION;
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t urProgramGetBuildInfo(
    ur_program_handle_t Program, ///< [in] handle of the Program object
    ur_device_handle_t Device,   ///< [in] handle of the Device object
    ur_program_build_info_t
        PropName,    ///< [in] name of the Program build info to query
    size_t PropSize, ///< [in] size of the Program build info property.
    void *PropValue, ///< [in,out][optional] value of the Program build
                     ///< property. If propSize is not equal to or greater than
                     ///< the real number of bytes needed to return the info
                     ///< then the ::UR_RESULT_ERROR_INVALID_SIZE error is
                     ///< returned and pKernelInfo is not used.
    size_t *PropSizeRet ///< [out][optional] pointer to the actual size in
                        ///< bytes of data being queried by propName.
) {
  std::ignore = Device;

  std::shared_lock<ur_shared_mutex> Guard(Program->Mutex);
  UrReturnHelper ReturnValue(PropSize, PropValue, PropSizeRet);
  if (PropName == UR_PROGRAM_BUILD_INFO_BINARY_TYPE) {
    ur_program_binary_type_t Type = UR_PROGRAM_BINARY_TYPE_NONE;
    auto State = Program->getState(Device->ZeDevice);
    if (State == ur_program_handle_t_::Object) {
      Type = UR_PROGRAM_BINARY_TYPE_COMPILED_OBJECT;
    } else if (State == ur_program_handle_t_::Exe) {
      Type = UR_PROGRAM_BINARY_TYPE_EXECUTABLE;
    }

    return ReturnValue(ur_program_binary_type_t{Type});
  }
  if (PropName == UR_PROGRAM_BUILD_INFO_OPTIONS) {
    // TODO: how to get module build options out of Level Zero?
    // For the programs that we compiled we can remember the options
    // passed with urProgramCompile/urProgramBuild, but what can we
    // return for programs that were built outside and registered
    // with urProgramRegister?
    return ReturnValue("");
  } else if (PropName == UR_PROGRAM_BUILD_INFO_STATUS) {
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
  } else if (PropName == UR_PROGRAM_BUILD_INFO_LOG) {
    // Check first to see if the plugin code recorded an error message.
    if (!Program->ErrorMessage.empty()) {
      return ReturnValue(Program->ErrorMessage.c_str());
    }

    // Next check if there is a Level Zero build log.
    auto ZeBuildLog = Program->getBuildLog(Device->ZeDevice);
    if (ZeBuildLog) {
      size_t LogSize = PropSize;
      ZE2UR_CALL(zeModuleBuildLogGetString,
                 (ZeBuildLog, &LogSize, ur_cast<char *>(PropValue)));
      if (PropSizeRet) {
        *PropSizeRet = LogSize;
      }
      if (PropValue) {
        // When the program build fails in urProgramBuild(), we delayed
        // cleaning up the build log because RT later calls this routine to
        // get the failed build log. To avoid memory leaks, we should clean up
        // the failed build log here because RT does not create sycl::program
        // when urProgramBuild() fails, thus it won't call urProgramRelease()
        // to clean up the build log.
        if (Program->getState(Device->ZeDevice) ==
            ur_program_handle_t_::Invalid) {
          ZE_CALL_NOCHECK(zeModuleBuildLogDestroy, (ZeBuildLog));
          Program->setBuildLog(Device->ZeDevice, nullptr);
        }
      }
      return UR_RESULT_SUCCESS;
    }

    // Otherwise, there is no error.  The OpenCL spec says to return an empty
    // string if there ws no previous attempt to compile, build, or link the
    // program.
    return ReturnValue("");
  } else {
    logger::error("urProgramGetBuildInfo: unsupported ParamName");
    return UR_RESULT_ERROR_INVALID_VALUE;
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t urProgramSetSpecializationConstant(
    ur_program_handle_t Program, ///< [in] handle of the Program object
    uint32_t SpecId,             ///< [in] specification constant Id
    size_t SpecSize,      ///< [in] size of the specialization constant value
    const void *SpecValue ///< [in] pointer to the specialization value bytes
) {
  std::ignore = Program;
  std::ignore = SpecId;
  std::ignore = SpecSize;
  std::ignore = SpecValue;
  logger::error(logger::LegacyMessage("[UR][L0] {} function not implemented!"),
                "{} function not implemented!");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urProgramGetNativeHandle(
    ur_program_handle_t Program,      ///< [in] handle of the program.
    ur_native_handle_t *NativeProgram ///< [out] a pointer to the native
                                      ///< handle of the program.
) {
  auto ZeModule = ur_cast<ze_module_handle_t *>(NativeProgram);

  std::shared_lock<ur_shared_mutex> Guard(Program->Mutex);
  assert(Program->AssociatedDevices.size() > 0);
  // Current API doesn't allow to specify device for which we want to get the
  // native handle. So, find the first device with a valid module handle.
  ze_module_handle_t Module = nullptr;
  for (const auto &Device : Program->AssociatedDevices) {
    Module = Program->getZeModuleHandle(Device->ZeDevice);
    if (Module) {
      break;
    }
  }
  if (!Module)
    return UR_RESULT_ERROR_INVALID_OPERATION;

  *ZeModule = Module;
  return UR_RESULT_SUCCESS;
}

ur_result_t urProgramCreateWithNativeHandle(
    ur_native_handle_t
        NativeProgram,           ///< [in] the native handle of the program.
    ur_context_handle_t Context, ///< [in] handle of the context instance
    const ur_program_native_properties_t
        *Properties, ///< [in][optional] pointer to native program properties
                     ///< struct.
    ur_program_handle_t *Program ///< [out] pointer to the handle of the
                                 ///< program object created.
) {
  UR_ASSERT(Context && NativeProgram, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(Program, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  auto ZeModule = ur_cast<ze_module_handle_t>(NativeProgram);

  // We assume here that programs created from a native handle always
  // represent a fully linked executable (state Exe) and not an unlinked
  // executable (state Object).

  try {
    ur_program_handle_t_ *UrProgram = new ur_program_handle_t_(
        ur_program_handle_t_::Exe, Context, ZeModule,
        Properties ? Properties->isNativeHandleOwned : false);
    *Program = reinterpret_cast<ur_program_handle_t>(UrProgram);
  } catch (const std::bad_alloc &) {
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t urProgramSetSpecializationConstants(
    ur_program_handle_t Program, ///< [in] handle of the Program object
    uint32_t Count, ///< [in] the number of elements in the pSpecConstants array
    const ur_specialization_constant_info_t
        *SpecConstants ///< [in][range(0, count)] array of specialization
                       ///< constant value descriptions
) {
  std::scoped_lock<ur_shared_mutex> Guard(Program->Mutex);

  // Remember the value of this specialization constant until the program is
  // built.  Note that we only save the pointer to the buffer that contains the
  // value.  The caller is responsible for maintaining storage for this buffer.
  //
  // NOTE: SpecSize is unused in Level Zero, the size is known from SPIR-V by
  // SpecID.
  for (uint32_t SpecIt = 0; SpecIt < Count; SpecIt++) {
    uint32_t SpecId = SpecConstants[SpecIt].id;
    Program->SpecConstants[SpecId] = SpecConstants[SpecIt].pValue;
  }
  return UR_RESULT_SUCCESS;
}

} // namespace ur::level_zero

ur_program_handle_t_::ur_program_handle_t_(state St,
                                           ur_context_handle_t Context,
                                           const void *Input, size_t Length)
    : Context{Context}, NativeProperties{nullptr}, OwnZeModule{true},
      AssociatedDevices(Context->getDevices()), SpirvCode{new uint8_t[Length]},
      SpirvCodeLength{Length} {
  std::memcpy(SpirvCode.get(), Input, Length);
  // All devices have the program in IL state.
  for (auto &Device : Context->getDevices()) {
    DeviceData &PerDevData = DeviceDataMap[Device->ZeDevice];
    PerDevData.State = St;
  }
}

ur_program_handle_t_::ur_program_handle_t_(
    state St, ur_context_handle_t Context, const uint32_t NumDevices,
    const ur_device_handle_t *Devices,
    const ur_program_properties_t *Properties, const uint8_t **Inputs,
    const size_t *Lengths)
    : Context{Context}, NativeProperties(Properties), OwnZeModule{true},
      AssociatedDevices(Devices, Devices + NumDevices) {
  for (uint32_t I = 0; I < NumDevices; ++I) {
    DeviceData &PerDevData = DeviceDataMap[Devices[I]->ZeDevice];
    PerDevData.State = St;
    PerDevData.Binary = std::make_pair(
        std::unique_ptr<uint8_t[]>(new uint8_t[Lengths[I]]), Lengths[I]);
    std::memcpy(PerDevData.Binary.first.get(), Inputs[I], Lengths[I]);
  }
}

ur_program_handle_t_::ur_program_handle_t_(ur_context_handle_t Context)
    : Context{Context}, NativeProperties{nullptr}, OwnZeModule{true},
      AssociatedDevices(Context->getDevices()) {}

ur_program_handle_t_::ur_program_handle_t_(state, ur_context_handle_t Context,
                                           ze_module_handle_t InteropZeModule)
    : Context{Context}, NativeProperties{nullptr}, OwnZeModule{true},
      AssociatedDevices({Context->getDevices()[0]}), InteropZeModule{
                                                         InteropZeModule} {}

ur_program_handle_t_::ur_program_handle_t_(state, ur_context_handle_t Context,
                                           ze_module_handle_t InteropZeModule,
                                           bool OwnZeModule)
    : Context{Context}, NativeProperties{nullptr}, OwnZeModule{OwnZeModule},
      AssociatedDevices({Context->getDevices()[0]}), InteropZeModule{
                                                         InteropZeModule} {
  // TODO: Currently it is not possible to understand the device associated
  // with provided ZeModule. So we can't set the state on that device to Exe.
}

ur_program_handle_t_::ur_program_handle_t_(state St,
                                           ur_context_handle_t Context,
                                           const std::string &ErrorMessage)
    : Context{Context}, NativeProperties{nullptr}, OwnZeModule{true},
      ErrorMessage{ErrorMessage}, AssociatedDevices(Context->getDevices()) {
  for (auto &Device : Context->getDevices()) {
    DeviceData &PerDevData = DeviceDataMap[Device->ZeDevice];
    PerDevData.State = St;
  }
}

ur_program_handle_t_::~ur_program_handle_t_() {
  if (!resourcesReleased) {
    ur_release_program_resources(true);
  }
}

void ur_program_handle_t_::ur_release_program_resources(bool deletion) {
  // According to Level Zero Specification, all kernels and build logs
  // must be destroyed before the Module can be destroyed.  So, be sure
  // to destroy build log before destroying the module.
  if (!deletion) {
    if (!RefCount.decrementAndTest()) {
      return;
    }
  }
  if (!resourcesReleased) {
    for (auto &[ZeDevice, DeviceData] : this->DeviceDataMap) {
      if (DeviceData.ZeBuildLog)
        ZE_CALL_NOCHECK(zeModuleBuildLogDestroy, (DeviceData.ZeBuildLog));
    }

    // interop api
    if (InteropZeModule && OwnZeModule)
      ZE_CALL_NOCHECK(zeModuleDestroy, (InteropZeModule));

    for (auto &[ZeDevice, DeviceData] : this->DeviceDataMap)
      if (DeviceData.ZeModule)
        ZE_CALL_NOCHECK(zeModuleDestroy, (DeviceData.ZeModule));

    this->DeviceDataMap.clear();

    resourcesReleased = true;
  }
}
