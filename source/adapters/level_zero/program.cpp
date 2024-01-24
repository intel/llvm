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
#include "ur_level_zero.hpp"

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

UR_APIEXPORT ur_result_t UR_APICALL urProgramCreateWithIL(
    ur_context_handle_t Context, ///< [in] handle of the context instance
    const void *IL,              ///< [in] pointer to IL binary.
    size_t Length,               ///< [in] length of `pIL` in bytes.
    const ur_program_properties_t
        *Properties, ///< [in][optional] pointer to program creation properties.
    ur_program_handle_t
        *Program ///< [out] pointer to handle of program object created.
) {
  std::ignore = Properties;
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

UR_APIEXPORT ur_result_t UR_APICALL urProgramCreateWithBinary(
    ur_context_handle_t Context, ///< [in] handle of the context instance
    ur_device_handle_t
        Device,            ///< [in] handle to device associated with binary.
    size_t Size,           ///< [in] size in bytes.
    const uint8_t *Binary, ///< [in] pointer to binary.
    const ur_program_properties_t
        *Properties, ///< [in][optional] pointer to program creation properties.
    ur_program_handle_t
        *Program ///< [out] pointer to handle of Program object created.
) {
  std::ignore = Device;
  std::ignore = Properties;
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
    ur_program_handle_t_ *UrProgram = new ur_program_handle_t_(
        ur_program_handle_t_::Native, Context, Binary, Size);
    *Program = reinterpret_cast<ur_program_handle_t>(UrProgram);
  } catch (const std::bad_alloc &) {
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramBuild(
    ur_context_handle_t Context, ///< [in] handle of the context instance.
    ur_program_handle_t Program, ///< [in] Handle of the program to build.
    const char *Options          ///< [in][optional] pointer to build options
                                 ///< null-terminated string.
) {
  return urProgramBuildExp(Program, 1, Context->Devices.data(), Options);
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramBuildExp(
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

  // We should have either IL or native device code.
  UR_ASSERT(hProgram->Code, UR_RESULT_ERROR_INVALID_PROGRAM);

  // It is legal to build a program created from either IL or from native
  // device code.
  if (hProgram->State != ur_program_handle_t_::IL &&
      hProgram->State != ur_program_handle_t_::Native) {
    return UR_RESULT_ERROR_INVALID_OPERATION;
  }

  std::scoped_lock<ur_shared_mutex> Guard(hProgram->Mutex);

  // Ask Level Zero to build and load the native code onto the device.
  ZeStruct<ze_module_desc_t> ZeModuleDesc;
  ur_program_handle_t_::SpecConstantShim Shim(hProgram);
  ZeModuleDesc.format = (hProgram->State == ur_program_handle_t_::IL)
                            ? ZE_MODULE_FORMAT_IL_SPIRV
                            : ZE_MODULE_FORMAT_NATIVE;

  ZeModuleDesc.inputSize = hProgram->CodeLength;
  ZeModuleDesc.pInputModule = hProgram->Code.get();

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

  ZeModuleDesc.pBuildFlags = ZeBuildOptions.c_str();
  ZeModuleDesc.pConstants = Shim.ze();
  ur_result_t Result = UR_RESULT_SUCCESS;

  for (uint32_t i = 0; i < numDevices; i++) {
    ze_device_handle_t ZeDevice = phDevices[i]->ZeDevice;
    ze_context_handle_t ZeContext = hProgram->Context->ZeContext;
    ze_module_handle_t ZeModuleHandle = nullptr;
    ze_module_build_log_handle_t ZeBuildLog{};

    hProgram->State = ur_program_handle_t_::Exe;
    ze_result_t ZeResult =
        ZE_CALL_NOCHECK(zeModuleCreate, (ZeContext, ZeDevice, &ZeModuleDesc,
                                         &ZeModuleHandle, &ZeBuildLog));
    if (ZeResult != ZE_RESULT_SUCCESS) {
      // We adjust ur_program below to avoid attempting to release zeModule when
      // RT calls urProgramRelease().
      hProgram->State = ur_program_handle_t_::Invalid;
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
        hProgram->State = ur_program_handle_t_::Invalid;
        Result = (ZeResult == ZE_RESULT_ERROR_MODULE_LINK_FAILURE)
                     ? UR_RESULT_ERROR_PROGRAM_BUILD_FAILURE
                     : ze2urResult(ZeResult);
        if (ZeModuleHandle) {
          ZE_CALL_NOCHECK(zeModuleDestroy, (ZeModuleHandle));
          ZeModuleHandle = nullptr;
        }
      }
      hProgram->ZeModuleMap.insert(std::make_pair(ZeDevice, ZeModuleHandle));
      hProgram->ZeBuildLogMap.insert(std::make_pair(ZeDevice, ZeBuildLog));
    }
  }

  // We no longer need the IL / native code.
  hProgram->Code.reset();
  if (!hProgram->ZeModuleMap.empty())
    hProgram->ZeModule = hProgram->ZeModuleMap.begin()->second;
  if (!hProgram->ZeBuildLogMap.empty())
    hProgram->ZeBuildLog = hProgram->ZeBuildLogMap.begin()->second;
  return Result;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramCompileExp(
    ur_program_handle_t
        hProgram,        ///< [in][out] handle of the program to compile.
    uint32_t numDevices, ///< [in] number of devices
    ur_device_handle_t *phDevices, ///< [in][range(0, numDevices)] pointer to
                                   ///< array of device handles
    const char *pOptions           ///< [in][optional] pointer to build options
                                   ///< null-terminated string.
) {
  std::ignore = numDevices;
  std::ignore = phDevices;
  return urProgramCompile(hProgram->Context, hProgram, pOptions);
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramCompile(
    ur_context_handle_t Context, ///< [in] handle of the context instance.
    ur_program_handle_t
        Program,        ///< [in][out] handle of the program to compile.
    const char *Options ///< [in][optional] pointer to build options
                        ///< null-terminated string.
) {
  std::ignore = Context;
  std::scoped_lock<ur_shared_mutex> Guard(Program->Mutex);

  // It's only valid to compile a program created from IL (we don't support
  // programs created from source code).
  //
  // The OpenCL spec says that the header parameters are ignored when compiling
  // IL programs, so we don't validate them.
  if (Program->State != ur_program_handle_t_::IL)
    return UR_RESULT_ERROR_INVALID_OPERATION;

  // We don't compile anything now.  Instead, we delay compilation until
  // urProgramLink, where we do both compilation and linking as a single step.
  // This produces better code because the driver can do cross-module
  // optimizations.  Therefore, we just remember the compilation flags, so we
  // can use them later.
  if (Options) {
    Program->BuildFlags = Options;

    // if large allocations are selected, then pass
    // ze-opt-greater-than-4GB-buffer-required to disable
    // stateful optimizations and be able to use larger than
    // 4GB allocations on these kernels.
    if (Context->Devices[0]->useRelaxedAllocationLimits()) {
      Program->BuildFlags += " -ze-opt-greater-than-4GB-buffer-required";
    }
  }
  Program->State = ur_program_handle_t_::Object;

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramLink(
    ur_context_handle_t Context, ///< [in] handle of the context instance.
    uint32_t Count, ///< [in] number of program handles in `phPrograms`.
    const ur_program_handle_t *Programs, ///< [in][range(0, count)] pointer to
                                         ///< array of program handles.
    const char *Options, ///< [in][optional] pointer to linker options
                         ///< null-terminated string.
    ur_program_handle_t
        *Program ///< [out] pointer to handle of program object created.
) {
  return urProgramLinkExp(Context, Count, Context->Devices.data(), 1, Programs,
                          Options, Program);
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramLinkExp(
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
    // that they are all in Object state.
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
      if (phPrograms[I]->State != ur_program_handle_t_::Object) {
        return UR_RESULT_ERROR_INVALID_OPERATION;
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
      CodeSizes[I] = Program->CodeLength;
      CodeBufs[I] = Program->Code.get();
      BuildFlagPtrs[I] = Program->BuildFlags.c_str();
      SpecConstShims.emplace_back(Program);
      SpecConstPtrs[I] = SpecConstShims[I].ze();
    }

    ZeExtModuleDesc.count = count;
    ZeExtModuleDesc.inputSizes = CodeSizes.data();
    ZeExtModuleDesc.pInputModules = CodeBufs.data();
    ZeExtModuleDesc.pBuildFlags = BuildFlagPtrs.data();
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
        ZeModuleDesc.pBuildFlags = ZeExtModuleDesc.pBuildFlags[0];
        ZeModuleDesc.pConstants = ZeExtModuleDesc.pConstants[0];
      } else {
        urPrint("urProgramLink: level_zero driver does not have static linking "
                "support.");
        return UR_RESULT_ERROR_INVALID_VALUE;
      }
    }
    std::unordered_map<ze_device_handle_t, ze_module_handle_t> ZeModuleMap;
    std::unordered_map<ze_device_handle_t, ze_module_build_log_handle_t>
        ZeBuildLogMap;

    for (uint32_t i = 0; i < numDevices; i++) {

      // Call the Level Zero API to compile, link, and create the module.
      ze_device_handle_t ZeDevice = phDevices[i]->ZeDevice;
      ze_context_handle_t ZeContext = hContext->ZeContext;
      ze_module_handle_t ZeModule = nullptr;
      ze_module_build_log_handle_t ZeBuildLog = nullptr;
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
        if (ZeResult == ZE_RESULT_ERROR_MODULE_LINK_FAILURE) {
          UrResult =
              UR_RESULT_ERROR_UNKNOWN; // TODO:
                                       // UR_RESULT_ERROR_PROGRAM_LINK_FAILURE;
        } else if (ZeResult != ZE_RESULT_SUCCESS) {
          return ze2urResult(ZeResult);
        }
      }
      ZeModuleMap.insert(std::make_pair(ZeDevice, ZeModule));
      ZeBuildLogMap.insert(std::make_pair(ZeDevice, ZeBuildLog));
    }

    ur_program_handle_t_::state State = (UrResult == UR_RESULT_SUCCESS)
                                            ? ur_program_handle_t_::Exe
                                            : ur_program_handle_t_::Invalid;
    ur_program_handle_t_ *UrProgram =
        new ur_program_handle_t_(State, hContext, ZeModuleMap.begin()->second,
                                 ZeBuildLogMap.begin()->second);
    *phProgram = reinterpret_cast<ur_program_handle_t>(UrProgram);
    (*phProgram)->ZeModuleMap = std::move(ZeModuleMap);
    (*phProgram)->ZeBuildLogMap = std::move(ZeBuildLogMap);
  } catch (const std::bad_alloc &) {
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
  return UrResult;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramRetain(
    ur_program_handle_t Program ///< [in] handle for the Program to retain
) {
  Program->RefCount.increment();
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramRelease(
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

UR_APIEXPORT ur_result_t UR_APICALL urProgramGetFunctionPointer(
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
  std::ignore = Device;

  std::shared_lock<ur_shared_mutex> Guard(Program->Mutex);
  if (Program->State != ur_program_handle_t_::Exe) {
    return UR_RESULT_ERROR_INVALID_PROGRAM_EXECUTABLE;
  }

  ze_result_t ZeResult =
      ZE_CALL_NOCHECK(zeModuleGetFunctionPointer,
                      (Program->ZeModule, FunctionName, FunctionPointerRet));

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
    UR_CALL(urProgramGetInfo(Program, UR_PROGRAM_INFO_KERNEL_NAMES, 0, nullptr,
                             &Size));

    std::string ClResult(Size, ' ');
    UR_CALL(urProgramGetInfo(Program, UR_PROGRAM_INFO_KERNEL_NAMES,
                             ClResult.size(), &ClResult[0], nullptr));

    // Get rid of the null terminator and search for kernel_name
    // If function can be found return error code to indicate it
    // exists
    ClResult.pop_back();
    if (is_in_separated_string(ClResult, ';', std::string(FunctionName)))
      return UR_RESULT_ERROR_INVALID_FUNCTION_NAME;

    return UR_RESULT_ERROR_INVALID_KERNEL_NAME;
  }

  if (ZeResult == ZE_RESULT_ERROR_INVALID_FUNCTION_NAME) {
    *FunctionPointerRet = 0;
    return UR_RESULT_ERROR_INVALID_KERNEL_NAME;
  }

  return ze2urResult(ZeResult);
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramGetInfo(
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
    // TODO: return true number of devices this program exists for.
    return ReturnValue(uint32_t{1});
  case UR_PROGRAM_INFO_DEVICES:
    // TODO: return all devices this program exists for.
    return ReturnValue(Program->Context->Devices[0]);
  case UR_PROGRAM_INFO_BINARY_SIZES: {
    std::shared_lock<ur_shared_mutex> Guard(Program->Mutex);
    size_t SzBinary;
    if (Program->State == ur_program_handle_t_::IL ||
        Program->State == ur_program_handle_t_::Native ||
        Program->State == ur_program_handle_t_::Object) {
      SzBinary = Program->CodeLength;
    } else if (Program->State == ur_program_handle_t_::Exe) {
      ZE2UR_CALL(zeModuleGetNativeBinary,
                 (Program->ZeModule, &SzBinary, nullptr));
    } else {
      return UR_RESULT_ERROR_INVALID_PROGRAM;
    }
    // This is an array of 1 element, initialized as if it were scalar.
    return ReturnValue(size_t{SzBinary});
  }
  case UR_PROGRAM_INFO_BINARIES: {
    // The caller sets "ParamValue" to an array of pointers, one for each
    // device.  Since Level Zero supports only one device, there is only one
    // pointer.  If the pointer is NULL, we don't do anything.  Otherwise, we
    // copy the program's binary image to the buffer at that pointer.
    uint8_t **PBinary = ur_cast<uint8_t **>(ProgramInfo);
    if (!PBinary[0])
      break;

    std::shared_lock<ur_shared_mutex> Guard(Program->Mutex);
    if (Program->State == ur_program_handle_t_::IL ||
        Program->State == ur_program_handle_t_::Native ||
        Program->State == ur_program_handle_t_::Object) {
      std::memcpy(PBinary[0], Program->Code.get(), Program->CodeLength);
    } else if (Program->State == ur_program_handle_t_::Exe) {
      size_t SzBinary = 0;
      ZE2UR_CALL(zeModuleGetNativeBinary,
                 (Program->ZeModule, &SzBinary, PBinary[0]));
    } else {
      return UR_RESULT_ERROR_INVALID_PROGRAM;
    }
    break;
  }
  case UR_PROGRAM_INFO_NUM_KERNELS: {
    std::shared_lock<ur_shared_mutex> Guard(Program->Mutex);
    uint32_t NumKernels;
    if (Program->State == ur_program_handle_t_::IL ||
        Program->State == ur_program_handle_t_::Native ||
        Program->State == ur_program_handle_t_::Object) {
      return UR_RESULT_ERROR_INVALID_PROGRAM_EXECUTABLE;
    } else if (Program->State == ur_program_handle_t_::Exe) {
      NumKernels = 0;
      ZE2UR_CALL(zeModuleGetKernelNames,
                 (Program->ZeModule, &NumKernels, nullptr));
    } else {
      return UR_RESULT_ERROR_INVALID_PROGRAM;
    }
    return ReturnValue(size_t{NumKernels});
  }
  case UR_PROGRAM_INFO_KERNEL_NAMES:
    try {
      std::shared_lock<ur_shared_mutex> Guard(Program->Mutex);
      std::string PINames{""};
      if (Program->State == ur_program_handle_t_::IL ||
          Program->State == ur_program_handle_t_::Native ||
          Program->State == ur_program_handle_t_::Object) {
        return UR_RESULT_ERROR_INVALID_PROGRAM_EXECUTABLE;
      } else if (Program->State == ur_program_handle_t_::Exe) {
        uint32_t Count = 0;
        ZE2UR_CALL(zeModuleGetKernelNames,
                   (Program->ZeModule, &Count, nullptr));
        std::unique_ptr<const char *[]> PNames(new const char *[Count]);
        ZE2UR_CALL(zeModuleGetKernelNames,
                   (Program->ZeModule, &Count, PNames.get()));
        for (uint32_t I = 0; I < Count; ++I) {
          PINames += (I > 0 ? ";" : "");
          PINames += PNames[I];
        }
      } else {
        return UR_RESULT_ERROR_INVALID_PROGRAM;
      }
      return ReturnValue(PINames.c_str());
    } catch (const std::bad_alloc &) {
      return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    } catch (...) {
      return UR_RESULT_ERROR_UNKNOWN;
    }
  default:
    die("urProgramGetInfo: not implemented");
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramGetBuildInfo(
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
    if (Program->State == ur_program_handle_t_::Object) {
      Type = UR_PROGRAM_BINARY_TYPE_COMPILED_OBJECT;
    } else if (Program->State == ur_program_handle_t_::Exe) {
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
  } else if (PropName == UR_PROGRAM_BUILD_INFO_LOG) {
    // Check first to see if the plugin code recorded an error message.
    if (!Program->ErrorMessage.empty()) {
      return ReturnValue(Program->ErrorMessage.c_str());
    }

    // Next check if there is a Level Zero build log.
    if (Program->ZeBuildLogMap.find(Device->ZeDevice) !=
        Program->ZeBuildLogMap.end()) {
      ze_module_build_log_handle_t ZeBuildLog =
          Program->ZeBuildLogMap.begin()->second;
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
        if (Program->State == ur_program_handle_t_::Invalid) {
          ZE_CALL_NOCHECK(zeModuleBuildLogDestroy, (ZeBuildLog));
          Program->ZeBuildLogMap.erase(Device->ZeDevice);
          ZeBuildLog = nullptr;
        }
      }
      return UR_RESULT_SUCCESS;
    }

    // Otherwise, there is no error.  The OpenCL spec says to return an empty
    // string if there ws no previous attempt to compile, build, or link the
    // program.
    return ReturnValue("");
  } else {
    urPrint("urProgramGetBuildInfo: unsupported ParamName\n");
    return UR_RESULT_ERROR_INVALID_VALUE;
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramSetSpecializationConstant(
    ur_program_handle_t Program, ///< [in] handle of the Program object
    uint32_t SpecId,             ///< [in] specification constant Id
    size_t SpecSize,      ///< [in] size of the specialization constant value
    const void *SpecValue ///< [in] pointer to the specialization value bytes
) {
  std::ignore = Program;
  std::ignore = SpecId;
  std::ignore = SpecSize;
  std::ignore = SpecValue;
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramGetNativeHandle(
    ur_program_handle_t Program,      ///< [in] handle of the program.
    ur_native_handle_t *NativeProgram ///< [out] a pointer to the native
                                      ///< handle of the program.
) {
  auto ZeModule = ur_cast<ze_module_handle_t *>(NativeProgram);

  std::shared_lock<ur_shared_mutex> Guard(Program->Mutex);
  switch (Program->State) {
  case ur_program_handle_t_::Exe: {
    *ZeModule = Program->ZeModule;
    break;
  }

  default:
    return UR_RESULT_ERROR_INVALID_OPERATION;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramCreateWithNativeHandle(
    ur_native_handle_t
        NativeProgram,           ///< [in] the native handle of the program.
    ur_context_handle_t Context, ///< [in] handle of the context instance
    const ur_program_native_properties_t
        *Properties, ///< [in][optional] pointer to native program properties
                     ///< struct.
    ur_program_handle_t *Program ///< [out] pointer to the handle of the
                                 ///< program object created.
) {
  std::ignore = Properties;
  auto ZeModule = ur_cast<ze_module_handle_t>(NativeProgram);

  // We assume here that programs created from a native handle always
  // represent a fully linked executable (state Exe) and not an unlinked
  // executable (state Object).

  try {
    ur_program_handle_t_ *UrProgram =
        new ur_program_handle_t_(ur_program_handle_t_::Exe, Context, ZeModule,
                                 Properties->isNativeHandleOwned);
    *Program = reinterpret_cast<ur_program_handle_t>(UrProgram);
  } catch (const std::bad_alloc &) {
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
  return UR_RESULT_SUCCESS;
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
    for (auto &ZeBuildLogPair : this->ZeBuildLogMap) {
      ZE_CALL_NOCHECK(zeModuleBuildLogDestroy, (ZeBuildLogPair.second));
    }

    if (ZeModule && OwnZeModule) {
      for (auto &ZeModulePair : this->ZeModuleMap) {
        ZE_CALL_NOCHECK(zeModuleDestroy, (ZeModulePair.second));
      }
      this->ZeModuleMap.clear();
    }
    resourcesReleased = true;
  }
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramSetSpecializationConstants(
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
