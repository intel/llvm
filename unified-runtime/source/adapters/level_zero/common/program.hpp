//===--------- program.hpp - Level Zero Adapter ---------------------------===//
//
// Copyright (C) 2025 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <ur_api.h>

namespace ur::level_zero::common {

ur_result_t urProgramCreateWithIL(ur_context_handle_t hContext, const void *pIL,
                                  size_t length,
                                  const ur_program_properties_t *pProperties,
                                  ur_program_handle_t *phProgram);
ur_result_t urProgramCreateWithBinary(
    ur_context_handle_t hContext, uint32_t numDevices,
    ur_device_handle_t *phDevices, size_t *pLengths, const uint8_t **ppBinaries,
    const ur_program_properties_t *pProperties, ur_program_handle_t *phProgram);
ur_result_t urProgramBuild(ur_context_handle_t hContext,
                           ur_program_handle_t hProgram, const char *pOptions);
ur_result_t urProgramCompile(ur_context_handle_t hContext,
                             ur_program_handle_t hProgram,
                             const char *pOptions);
ur_result_t urProgramLink(ur_context_handle_t hContext, uint32_t count,
                          const ur_program_handle_t *phPrograms,
                          const char *pOptions, ur_program_handle_t *phProgram);
ur_result_t urProgramRetain(ur_program_handle_t hProgram);
ur_result_t urProgramRelease(ur_program_handle_t hProgram);
ur_result_t urProgramGetFunctionPointer(ur_device_handle_t hDevice,
                                        ur_program_handle_t hProgram,
                                        const char *pFunctionName,
                                        void **ppFunctionPointer);
ur_result_t urProgramGetGlobalVariablePointer(
    ur_device_handle_t hDevice, ur_program_handle_t hProgram,
    const char *pGlobalVariableName, size_t *pGlobalVariableSizeRet,
    void **ppGlobalVariablePointerRet);
ur_result_t urProgramGetInfo(ur_program_handle_t hProgram,
                             ur_program_info_t propName, size_t propSize,
                             void *pPropValue, size_t *pPropSizeRet);
ur_result_t urProgramGetBuildInfo(ur_program_handle_t hProgram,
                                  ur_device_handle_t hDevice,
                                  ur_program_build_info_t propName,
                                  size_t propSize, void *pPropValue,
                                  size_t *pPropSizeRet);
ur_result_t urProgramSetSpecializationConstants(
    ur_program_handle_t hProgram, uint32_t count,
    const ur_specialization_constant_info_t *pSpecConstants);
ur_result_t urProgramGetNativeHandle(ur_program_handle_t hProgram,
                                     ur_native_handle_t *phNativeProgram);
ur_result_t urProgramCreateWithNativeHandle(
    ur_native_handle_t hNativeProgram, ur_context_handle_t hContext,
    const ur_program_native_properties_t *pProperties,
    ur_program_handle_t *phProgram);
ur_result_t urProgramDynamicLinkExp(ur_context_handle_t hContext,
                                    uint32_t count,
                                    const ur_program_handle_t *phPrograms);
ur_result_t urProgramBuildExp(ur_program_handle_t hProgram, uint32_t numDevices,
                              ur_device_handle_t *phDevices,
                              ur_exp_program_flags_t flags,
                              const char *pOptions);
ur_result_t urProgramCompileExp(ur_program_handle_t hProgram,
                                uint32_t numDevices,
                                ur_device_handle_t *phDevices,
                                ur_exp_program_flags_t flags,
                                const char *pOptions);
ur_result_t urProgramLinkExp(ur_context_handle_t hContext, uint32_t numDevices,
                             ur_device_handle_t *phDevices,
                             ur_exp_program_flags_t flags, uint32_t count,
                             const ur_program_handle_t *phPrograms,
                             const char *pOptions,
                             ur_program_handle_t *phProgram);

} // namespace ur::level_zero::common
