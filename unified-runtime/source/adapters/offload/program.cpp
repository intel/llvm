//===----------- program.cpp - LLVM Offload Adapter  ----------------------===//
//
// Copyright (C) 2025 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <OffloadAPI.h>
#include <ur/ur.hpp>
#include <ur_api.h>

#include "context.hpp"
#include "device.hpp"
#include "offload_bundle_parser.hpp"
#include "platform.hpp"
#include "program.hpp"
#include "ur2offload.hpp"

#ifdef UR_CUDA_ENABLED
#include <cuda.h>
#endif

namespace {
// Workaround for Offload not supporting PTX binaries. Force CUDA programs
// to be linked so they end up as CUBIN.
#ifdef UR_CUDA_ENABLED
ur_result_t ProgramCreateCudaWorkaround(ur_context_handle_t hContext,
                                        const uint8_t *Binary, size_t Length,
                                        ur_program_handle_t hProgram) {
  uint8_t *RealBinary;
  size_t RealLength;
  CUlinkState State;
  cuLinkCreate(0, nullptr, nullptr, &State);

  cuLinkAddData(State, CU_JIT_INPUT_PTX, (char *)(Binary), Length, nullptr, 0,
                nullptr, nullptr);

  void *CuBin = nullptr;
  size_t CuBinSize = 0;
  cuLinkComplete(State, &CuBin, &CuBinSize);
  RealBinary = (uint8_t *)CuBin;
  RealLength = CuBinSize;

#if 0
  fprintf(stderr, "Performed CUDA bin workaround (size = %lu)\n", RealLength);
#endif

  auto Res = olCreateProgram(hContext->Device->OffloadDevice, RealBinary,
                             RealLength, &hProgram->OffloadProgram);

  // Program owns the linked module now
  cuLinkDestroy(State);

  return offloadResultToUR(Res);
}
#else
ur_result_t ProgramCreateCudaWorkaround(ur_context_handle_t, const uint8_t *,
                                        size_t, ur_program_handle_t) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
#endif

} // namespace

UR_APIEXPORT ur_result_t UR_APICALL urProgramCreateWithBinary(
    ur_context_handle_t hContext, uint32_t numDevices,
    ur_device_handle_t *phDevices, size_t *pLengths, const uint8_t **ppBinaries,
    const ur_program_properties_t *pProperties,
    ur_program_handle_t *phProgram) {
  if (numDevices > 1) {
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  auto *RealBinary = ppBinaries[0];
  size_t RealLength = pLengths[0];

  if (auto Parser = HipOffloadBundleParser::load(RealBinary, RealLength)) {
    std::string DevName{};
    size_t DevNameLength;
    OL_RETURN_ON_ERR(olGetDeviceInfoSize(phDevices[0]->OffloadDevice,
                                         OL_DEVICE_INFO_NAME, &DevNameLength));
    DevName.resize(DevNameLength - 1);
    OL_RETURN_ON_ERR(olGetDeviceInfo(phDevices[0]->OffloadDevice,
                                     OL_DEVICE_INFO_NAME, DevNameLength,
                                     DevName.data()));

    auto Res = Parser->extract(DevName, RealBinary, RealLength);
    if (Res != UR_RESULT_SUCCESS) {
      return Res;
    }
  }

  ur_program_handle_t Program = new ur_program_handle_t_{};
  Program->URContext = hContext;
  Program->Binary = RealBinary;
  Program->BinarySizeInBytes = RealLength;

  // Parse properties
  if (pProperties) {
    if (pProperties->count > 0 && pProperties->pMetadatas == nullptr) {
      return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    } else if (pProperties->count == 0 && pProperties->pMetadatas != nullptr) {
      return UR_RESULT_ERROR_INVALID_SIZE;
    }

    auto Length = pProperties->count;
    auto Metadata = pProperties->pMetadatas;
    for (size_t i = 0; i < Length; ++i) {
      const ur_program_metadata_t MetadataElement = Metadata[i];
      std::string MetadataElementName{MetadataElement.pName};

      auto [Prefix, Tag] = splitMetadataName(MetadataElementName);

      if (Tag == __SYCL_UR_PROGRAM_METADATA_GLOBAL_ID_MAPPING) {
        const char *MetadataValPtr =
            reinterpret_cast<const char *>(MetadataElement.value.pData) +
            sizeof(std::uint64_t);
        const char *MetadataValPtrEnd =
            MetadataValPtr + MetadataElement.size - sizeof(std::uint64_t);
        Program->GlobalIDMD[Prefix] =
            std::string{MetadataValPtr, MetadataValPtrEnd};
      }
    }
  }

  ur_result_t Res;
  ol_platform_backend_t Backend;
  olGetPlatformInfo(phDevices[0]->Platform->OffloadPlatform,
                    OL_PLATFORM_INFO_BACKEND, sizeof(Backend), &Backend);
  if (Backend == OL_PLATFORM_BACKEND_CUDA) {
    Res =
        ProgramCreateCudaWorkaround(hContext, RealBinary, RealLength, Program);
  } else {
    Res = offloadResultToUR(olCreateProgram(hContext->Device->OffloadDevice,
                                            RealBinary, RealLength,
                                            &Program->OffloadProgram));
  }

  if (Res != UR_RESULT_SUCCESS) {
    delete Program;
    return Res;
  }

  *phProgram = Program;

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urProgramCreateWithIL(ur_context_handle_t hContext, const void *pIL,
                      size_t length, const ur_program_properties_t *pProperties,
                      ur_program_handle_t *phProgram) {
  // Liboffload consumes both IR and binaries through the same entrypoint
  return urProgramCreateWithBinary(hContext, 1, &hContext->Device, &length,
                                   reinterpret_cast<const uint8_t **>(&pIL),
                                   pProperties, phProgram);
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramBuild(ur_context_handle_t,
                                                   ur_program_handle_t,
                                                   const char *) {
  // Do nothing, program is built upon creation
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramBuildExp(ur_program_handle_t,
                                                      uint32_t,
                                                      ur_device_handle_t *,
                                                      const char *) {
  // Do nothing, program is built upon creation
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramCompile(ur_context_handle_t,
                                                     ur_program_handle_t,
                                                     const char *) {
  // Do nothing, program is built upon creation
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urProgramGetInfo(ur_program_handle_t hProgram, ur_program_info_t propName,
                 size_t propSize, void *pPropValue, size_t *pPropSizeRet) {
  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);

  switch (propName) {
  case UR_PROGRAM_INFO_REFERENCE_COUNT:
    return ReturnValue(hProgram->RefCount.load());
  case UR_PROGRAM_INFO_CONTEXT:
    return ReturnValue(hProgram->URContext);
  case UR_PROGRAM_INFO_NUM_DEVICES:
    return ReturnValue(1);
  case UR_PROGRAM_INFO_DEVICES:
    return ReturnValue(&hProgram->URContext->Device, 1);
  case UR_PROGRAM_INFO_IL:
    return ReturnValue(reinterpret_cast<const char *>(0), 0);
  case UR_PROGRAM_INFO_BINARY_SIZES:
    return ReturnValue(&hProgram->BinarySizeInBytes, 1);
  case UR_PROGRAM_INFO_BINARIES: {
    if (!pPropValue && !pPropSizeRet) {
      return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (pPropValue != nullptr) {
      if (propSize < sizeof(void *)) {
        return UR_RESULT_ERROR_INVALID_SIZE;
      }

      std::memcpy(*reinterpret_cast<void **>(pPropValue), hProgram->Binary,
                  hProgram->BinarySizeInBytes);
    }

    if (pPropSizeRet != nullptr) {
      *pPropSizeRet = sizeof(void *);
    }
    break;
  }
  case UR_PROGRAM_INFO_NUM_KERNELS:
  case UR_PROGRAM_INFO_KERNEL_NAMES:
    // Program introspection is not available for liboffload (or generally,
    // amdgpu/cuda)
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
  default:
    return UR_RESULT_ERROR_INVALID_ENUMERATION;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urProgramRetain(ur_program_handle_t hProgram) {
  hProgram->RefCount++;
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urProgramRelease(ur_program_handle_t hProgram) {
  if (--hProgram->RefCount == 0) {
    auto Res = olDestroyProgram(hProgram->OffloadProgram);
    if (Res) {
      return offloadResultToUR(Res);
    }
    delete hProgram;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urProgramGetNativeHandle(ur_program_handle_t, ur_native_handle_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramCreateWithNativeHandle(
    ur_native_handle_t, ur_context_handle_t,
    const ur_program_native_properties_t *, ur_program_handle_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramSetSpecializationConstants(
    ur_program_handle_t, uint32_t, const ur_specialization_constant_info_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramGetGlobalVariablePointer(
    ur_device_handle_t, ur_program_handle_t hProgram,
    const char *pGlobalVariableName, size_t *pGlobalVariableSizeRet,
    void **ppGlobalVariablePointerRet) {
  auto DeviceGlobalNameIt = hProgram->GlobalIDMD.find(pGlobalVariableName);
  if (DeviceGlobalNameIt == hProgram->GlobalIDMD.end())
    return UR_RESULT_ERROR_INVALID_VALUE;
  std::string DeviceGlobalName = DeviceGlobalNameIt->second;

  ol_symbol_handle_t Symbol;
  auto Err = olGetSymbol(hProgram->OffloadProgram, DeviceGlobalName.c_str(),
                         OL_SYMBOL_KIND_GLOBAL_VARIABLE, &Symbol);
  if (Err && Err->Code == OL_ERRC_NOT_FOUND) {
    return UR_RESULT_ERROR_INVALID_VALUE;
  }
  OL_RETURN_ON_ERR(Err);

  if (pGlobalVariableSizeRet) {
    OL_RETURN_ON_ERR(olGetSymbolInfo(Symbol,
                                     OL_SYMBOL_INFO_GLOBAL_VARIABLE_SIZE,
                                     sizeof(size_t), pGlobalVariableSizeRet));
  }
  OL_RETURN_ON_ERR(olGetSymbolInfo(Symbol,
                                   OL_SYMBOL_INFO_GLOBAL_VARIABLE_ADDRESS,
                                   sizeof(void *), ppGlobalVariablePointerRet));

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urProgramGetFunctionPointer([[maybe_unused]] ur_device_handle_t hDevice,
                            [[maybe_unused]] ur_program_handle_t hProgram,
                            [[maybe_unused]] const char *pFunctionName,
                            [[maybe_unused]] void **ppFunctionPointer) {
  // liboffload doesn't support a representation of function pointers (yet)
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
