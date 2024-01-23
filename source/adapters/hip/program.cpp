//===--------- program.cpp - HIP Adapter ----------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "program.hpp"

#ifdef SYCL_ENABLE_KERNEL_FUSION
#ifdef UR_COMGR_VERSION4_INCLUDE
#include <amd_comgr.h>
#else
#include <amd_comgr/amd_comgr.h>
#endif
namespace {
template <typename ReleaseType, ReleaseType Release, typename T>
struct COMgrObjCleanUp {
  COMgrObjCleanUp(T Obj) : Obj{Obj} {}
  ~COMgrObjCleanUp() { Release(Obj); }
  T Obj;
};

using COMgrDataTCleanUp =
    COMgrObjCleanUp<decltype(&amd_comgr_release_data), &amd_comgr_release_data,
                    amd_comgr_data_t>;
using COMgrDataSetTCleanUp =
    COMgrObjCleanUp<decltype(&amd_comgr_destroy_data_set),
                    &amd_comgr_destroy_data_set, amd_comgr_data_set_t>;
using COMgrActionInfoCleanUp =
    COMgrObjCleanUp<decltype(&amd_comgr_destroy_action_info),
                    &amd_comgr_destroy_action_info, amd_comgr_action_info_t>;

void getCoMgrBuildLog(const amd_comgr_data_set_t BuildDataSet, char *BuildLog,
                      size_t MaxLogSize) {
  size_t count = 0;
  amd_comgr_status_t status = amd_comgr_action_data_count(
      BuildDataSet, AMD_COMGR_DATA_KIND_LOG, &count);

  if (status != AMD_COMGR_STATUS_SUCCESS || count == 0) {
    std::strcpy(BuildLog, "extracting build log failed (no log).");
    return;
  }

  amd_comgr_data_t LogBinaryData;

  if (amd_comgr_action_data_get_data(BuildDataSet, AMD_COMGR_DATA_KIND_LOG, 0,
                                     &LogBinaryData) !=
      AMD_COMGR_STATUS_SUCCESS) {
    std::strcpy(BuildLog, "extracting build log failed (no data).");
    return;
  }
  COMgrDataTCleanUp LogDataCleanup{LogBinaryData};

  size_t binarySize = 0;
  if (amd_comgr_get_data(LogBinaryData, &binarySize, NULL) !=
      AMD_COMGR_STATUS_SUCCESS) {
    std::strcpy(BuildLog, "extracting build log failed (no log size).");
    return;
  }

  if (binarySize == 0) {
    std::strcpy(BuildLog, "no log.");
    return;
  }

  size_t bufSize = binarySize < MaxLogSize ? binarySize : MaxLogSize;

  if (amd_comgr_get_data(LogBinaryData, &bufSize, BuildLog) !=
      AMD_COMGR_STATUS_SUCCESS) {
    std::strcpy(BuildLog, "extracting build log failed (cannot copy log).");
    return;
  }
}
} // namespace
#endif

std::pair<std::string, std::string>
splitMetadataName(const std::string &metadataName) {
  size_t splitPos = metadataName.rfind('@');
  if (splitPos == std::string::npos)
    return std::make_pair(metadataName, std::string{});
  return std::make_pair(metadataName.substr(0, splitPos),
                        metadataName.substr(splitPos, metadataName.length()));
}

ur_result_t
ur_program_handle_t_::setMetadata(const ur_program_metadata_t *Metadata,
                                  size_t Length) {
  for (size_t i = 0; i < Length; ++i) {
    const ur_program_metadata_t MetadataElement = Metadata[i];
    std::string MetadataElementName{MetadataElement.pName};

    auto [Prefix, Tag] = splitMetadataName(MetadataElementName);

    if (MetadataElementName ==
        __SYCL_UR_PROGRAM_METADATA_TAG_NEED_FINALIZATION) {
      assert(MetadataElement.type == UR_PROGRAM_METADATA_TYPE_UINT32);
      IsRelocatable = MetadataElement.value.data32;
    } else if (Tag == __SYCL_UR_PROGRAM_METADATA_GLOBAL_ID_MAPPING) {
      const char *MetadataValPtr =
          reinterpret_cast<const char *>(MetadataElement.value.pData) +
          sizeof(std::uint64_t);
      const char *MetadataValPtrEnd =
          MetadataValPtr + MetadataElement.size - sizeof(std::uint64_t);
      GlobalIDMD[Prefix] = std::string{MetadataValPtr, MetadataValPtrEnd};
    }
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t ur_program_handle_t_::setBinary(const char *Source, size_t Length) {
  // Do not re-set program binary data which has already been set as that will
  // delete the old binary data.
  UR_ASSERT(Binary == nullptr && BinarySizeInBytes == 0,
            UR_RESULT_ERROR_INVALID_OPERATION);
  Binary = Source;
  BinarySizeInBytes = Length;
  return UR_RESULT_SUCCESS;
}

ur_result_t ur_program_handle_t_::finalizeRelocatable() {
#ifndef SYCL_ENABLE_KERNEL_FUSION
  assert(false && "Relocation only available with fusion");
  return UR_RESULT_ERROR_UNKNOWN;
#else
  assert(IsRelocatable && "Not a relocatable input");
  amd_comgr_data_t ComgrData;
  amd_comgr_data_set_t RelocatableData;
  UR_CHECK_ERROR(amd_comgr_create_data_set(&RelocatableData));
  COMgrDataSetTCleanUp RelocatableDataCleanup{RelocatableData};

  UR_CHECK_ERROR(
      amd_comgr_create_data(AMD_COMGR_DATA_KIND_RELOCATABLE, &ComgrData));
  // RAII for auto clean-up
  COMgrDataTCleanUp DataCleanup{ComgrData};
  UR_CHECK_ERROR(amd_comgr_set_data(ComgrData, BinarySizeInBytes, Binary));
  UR_CHECK_ERROR(amd_comgr_set_data_name(ComgrData, "jit_obj.o"));

  UR_CHECK_ERROR(amd_comgr_data_set_add(RelocatableData, ComgrData));

  amd_comgr_action_info_t Action;

  UR_CHECK_ERROR(amd_comgr_create_action_info(&Action));
  COMgrActionInfoCleanUp ActionCleanUp{Action};

  std::string ISA = "amdgcn-amd-amdhsa--";
  hipDeviceProp_t Props;
  detail::ur::assertion(hipGetDeviceProperties(&Props, getDevice()->get()) ==
                        hipSuccess);
  ISA += Props.gcnArchName;
  UR_CHECK_ERROR(amd_comgr_action_info_set_isa_name(Action, ISA.data()));

  UR_CHECK_ERROR(amd_comgr_action_info_set_logging(Action, true));

  amd_comgr_data_set_t Output;
  UR_CHECK_ERROR(amd_comgr_create_data_set(&Output));
  COMgrDataSetTCleanUp OutputDataCleanup{Output};

  if (amd_comgr_do_action(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE,
                          Action, RelocatableData,
                          Output) != AMD_COMGR_STATUS_SUCCESS) {
    getCoMgrBuildLog(Output, ErrorLog, MAX_LOG_SIZE);
    return UR_RESULT_ERROR_PROGRAM_BUILD_FAILURE;
  }
  amd_comgr_data_t binaryData;

  UR_CHECK_ERROR(amd_comgr_action_data_get_data(
      Output, AMD_COMGR_DATA_KIND_EXECUTABLE, 0, &binaryData));
  {
    COMgrDataTCleanUp binaryDataCleanUp{binaryData};

    size_t binarySize = 0;
    UR_CHECK_ERROR(amd_comgr_get_data(binaryData, &binarySize, NULL));

    ExecutableCache.resize(binarySize);

    UR_CHECK_ERROR(
        amd_comgr_get_data(binaryData, &binarySize, ExecutableCache.data()));
  }
  Binary = ExecutableCache.data();
  BinarySizeInBytes = ExecutableCache.size();
  return UR_RESULT_SUCCESS;
#endif
}

ur_result_t ur_program_handle_t_::buildProgram(const char *BuildOptions) {
  if (IsRelocatable) {
    if (finalizeRelocatable() != UR_RESULT_SUCCESS) {
      BuildStatus = UR_PROGRAM_BUILD_STATUS_ERROR;
      return UR_RESULT_ERROR_PROGRAM_BUILD_FAILURE;
    }
    IsRelocatable = false;
  }

  if (BuildOptions) {
    this->BuildOptions = BuildOptions;
  }

  constexpr const unsigned int NumberOfOptions = 4u;

  hipJitOption Options[NumberOfOptions];
  void *OptionVals[NumberOfOptions];

  // Pass a buffer for info messages
  Options[0] = hipJitOptionInfoLogBuffer;
  OptionVals[0] = (void *)InfoLog;
  // Pass the size of the info buffer
  Options[1] = hipJitOptionInfoLogBufferSizeBytes;
  OptionVals[1] = (void *)(long)MAX_LOG_SIZE;
  // Pass a buffer for error message
  Options[2] = hipJitOptionErrorLogBuffer;
  OptionVals[2] = (void *)ErrorLog;
  // Pass the size of the error buffer
  Options[3] = hipJitOptionErrorLogBufferSizeBytes;
  OptionVals[3] = (void *)(long)MAX_LOG_SIZE;

  UR_CHECK_ERROR(hipModuleLoadDataEx(&Module, static_cast<const void *>(Binary),
                                     NumberOfOptions, Options, OptionVals));

  BuildStatus = UR_PROGRAM_BUILD_STATUS_SUCCESS;

  // If no exception, result is correct
  return UR_RESULT_SUCCESS;
}

/// Finds kernel names by searching for entry points in the PTX source, as the
/// HIP driver API doesn't expose an operation for this.
/// Note: This is currently only being used by the SYCL program class for the
///       has_kernel method, so an alternative would be to move the has_kernel
///       query to UR and use hipModuleGetFunction to check for a kernel.
ur_result_t getKernelNames(ur_program_handle_t) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

/// A program must be specific to a device so this entry point is UNSUPPORTED
UR_APIEXPORT ur_result_t UR_APICALL
urProgramCreateWithIL(ur_context_handle_t, const void *, size_t,
                      const ur_program_properties_t *, ur_program_handle_t *) {
  detail::ur::die("urProgramCreateWithIL not implemented for HIP adapter"
                  " please use urProgramCreateWithBinary instead");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

/// HIP will handle the PTX/HIPBIN binaries internally through a call to
/// hipModuleLoadDataEx. So, urProgramCompile and urProgramBuild are equivalent
/// in terms of HIP adapter. \TODO Implement asynchronous compilation
UR_APIEXPORT ur_result_t UR_APICALL
urProgramCompile(ur_context_handle_t hContext, ur_program_handle_t hProgram,
                 const char *pOptions) {
  return urProgramBuild(hContext, hProgram, pOptions);
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramCompileExp(ur_program_handle_t,
                                                        uint32_t,
                                                        ur_device_handle_t *,
                                                        const char *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramBuildExp(ur_program_handle_t,
                                                      uint32_t,
                                                      ur_device_handle_t *,
                                                      const char *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

/// Loads the images from a UR program into a hipModule_t that can be
/// used later on to extract functions (kernels).
/// See \ref ur_program_handle_t for implementation details.
UR_APIEXPORT ur_result_t UR_APICALL urProgramBuild(ur_context_handle_t,
                                                   ur_program_handle_t hProgram,
                                                   const char *pOptions) {
  ur_result_t Result = UR_RESULT_SUCCESS;

  try {
    ScopedContext Active(hProgram->getDevice());

    hProgram->buildProgram(pOptions);

  } catch (ur_result_t Err) {
    Result = Err;
  }
  return Result;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramLinkExp(
    ur_context_handle_t, uint32_t, ur_device_handle_t *, uint32_t,
    const ur_program_handle_t *, const char *, ur_program_handle_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramLink(ur_context_handle_t, uint32_t,
                                                  const ur_program_handle_t *,
                                                  const char *,
                                                  ur_program_handle_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

/// Created a UR program object from a HIP program handle.
/// TODO: Implement this.
/// NOTE: The created UR object takes ownership of the native handle.
///
/// \param[in] hNativeProgram The native handle to create UR program object
/// from. \param[in] hContext The UR context of the program. \param[out]
/// phProgram Set to the UR program object created from native handle.
///
/// \return UR_RESULT_ERROR_UNSUPPORTED_FEATURE
UR_APIEXPORT ur_result_t UR_APICALL urProgramCreateWithNativeHandle(
    ur_native_handle_t, ur_context_handle_t,
    const ur_program_native_properties_t *, ur_program_handle_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL
urProgramGetBuildInfo(ur_program_handle_t hProgram, ur_device_handle_t,
                      ur_program_build_info_t propName, size_t propSize,
                      void *pPropValue, size_t *pPropSizeRet) {
  // Ignore unused parameter
  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);

  switch (propName) {
  case UR_PROGRAM_BUILD_INFO_STATUS: {
    return ReturnValue(hProgram->BuildStatus);
  }
  case UR_PROGRAM_BUILD_INFO_OPTIONS:
    return ReturnValue(hProgram->BuildOptions.c_str());
  case UR_PROGRAM_BUILD_INFO_LOG:
    return ReturnValue(hProgram->InfoLog, hProgram->MAX_LOG_SIZE);
  default:
    break;
  }
  return UR_RESULT_ERROR_INVALID_ENUMERATION;
}

UR_APIEXPORT ur_result_t UR_APICALL
urProgramGetInfo(ur_program_handle_t hProgram, ur_program_info_t propName,
                 size_t propSize, void *pProgramInfo, size_t *pPropSizeRet) {
  UrReturnHelper ReturnValue(propSize, pProgramInfo, pPropSizeRet);

  switch (propName) {
  case UR_PROGRAM_INFO_REFERENCE_COUNT:
    return ReturnValue(hProgram->getReferenceCount());
  case UR_PROGRAM_INFO_CONTEXT:
    return ReturnValue(hProgram->Context);
  case UR_PROGRAM_INFO_NUM_DEVICES:
    return ReturnValue(1u);
  case UR_PROGRAM_INFO_DEVICES:
    return ReturnValue(hProgram->getDevice(), 1);
  case UR_PROGRAM_INFO_SOURCE:
    return ReturnValue(hProgram->Binary);
  case UR_PROGRAM_INFO_BINARY_SIZES:
    return ReturnValue(&hProgram->BinarySizeInBytes, 1);
  case UR_PROGRAM_INFO_BINARIES:
    return ReturnValue(&hProgram->Binary, 1);
  case UR_PROGRAM_INFO_KERNEL_NAMES:
    return getKernelNames(hProgram);
  default:
    break;
  }
  return UR_RESULT_ERROR_INVALID_ENUMERATION;
}

UR_APIEXPORT ur_result_t UR_APICALL
urProgramRetain(ur_program_handle_t hProgram) {
  UR_ASSERT(hProgram->getReferenceCount() > 0, UR_RESULT_ERROR_INVALID_PROGRAM);
  hProgram->incrementReferenceCount();
  return UR_RESULT_SUCCESS;
}

/// Decreases the reference count of a ur_program_handle_t object.
/// When the reference count reaches 0, it unloads the module from
/// the context.
UR_APIEXPORT ur_result_t UR_APICALL
urProgramRelease(ur_program_handle_t hProgram) {
  // double delete or someone is messing with the ref count.
  // either way, cannot safely proceed.
  UR_ASSERT(hProgram->getReferenceCount() != 0,
            UR_RESULT_ERROR_INVALID_PROGRAM);

  // decrement ref count. If it is 0, delete the program.
  if (hProgram->decrementReferenceCount() == 0) {

    std::unique_ptr<ur_program_handle_t_> ProgramPtr{hProgram};

    ur_result_t Result = UR_RESULT_ERROR_INVALID_PROGRAM;

    try {
      ScopedContext Active(hProgram->getDevice());
      auto HIPModule = hProgram->get();
      if (HIPModule) {
        UR_CHECK_ERROR(hipModuleUnload(HIPModule));
        Result = UR_RESULT_SUCCESS;
      } else {
        // no module to unload
        Result = UR_RESULT_SUCCESS;
      }
    } catch (...) {
      Result = UR_RESULT_ERROR_OUT_OF_RESOURCES;
    }

    return Result;
  }

  return UR_RESULT_SUCCESS;
}

/// Gets the native HIP handle of a UR program object
///
/// \param[in] hProgram The UR program to get the native HIP object of.
/// \param[out] phNativeProgram Set to the native handle of the UR program
/// object.
///
/// \return UR_RESULT_SUCCESS
UR_APIEXPORT ur_result_t UR_APICALL urProgramGetNativeHandle(
    ur_program_handle_t hProgram, ur_native_handle_t *phNativeProgram) {
  *phNativeProgram = reinterpret_cast<ur_native_handle_t>(hProgram->get());
  return UR_RESULT_SUCCESS;
}

/// Loads images from a list of PTX or HIPBin binaries.
/// Note: No calls to HIP driver API in this function, only store binaries
/// for later.
///
/// Note: Only supports one device
UR_APIEXPORT ur_result_t UR_APICALL urProgramCreateWithBinary(
    ur_context_handle_t hContext, ur_device_handle_t hDevice, size_t size,
    const uint8_t *pBinary, const ur_program_properties_t *pProperties,
    ur_program_handle_t *phProgram) {
  UR_ASSERT(std::find(hContext->getDevices().begin(),
                      hContext->getDevices().end(),
                      hDevice) != hContext->getDevices().end(),
            UR_RESULT_ERROR_INVALID_CONTEXT);

  ur_result_t Result = UR_RESULT_SUCCESS;

  std::unique_ptr<ur_program_handle_t_> RetProgram{
      new ur_program_handle_t_{hContext, hDevice}};

  // TODO: Set metadata here and use reqd_work_group_size information.
  // See urProgramCreateWithBinary in CUDA adapter.
  if (pProperties) {
    if (pProperties->count > 0 && pProperties->pMetadatas == nullptr) {
      return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    } else if (pProperties->count == 0 && pProperties->pMetadatas != nullptr) {
      return UR_RESULT_ERROR_INVALID_SIZE;
    }
    Result =
        RetProgram->setMetadata(pProperties->pMetadatas, pProperties->count);
  }
  UR_ASSERT(Result == UR_RESULT_SUCCESS, Result);

  auto pBinary_string = reinterpret_cast<const char *>(pBinary);
  if (size == 0) {
    size = strlen(pBinary_string) + 1;
  }

  UR_ASSERT(size, UR_RESULT_ERROR_INVALID_SIZE);

  Result = RetProgram->setBinary(pBinary_string, size);
  UR_ASSERT(Result == UR_RESULT_SUCCESS, Result);

  *phProgram = RetProgram.release();

  return Result;
}

// This entry point is only used for native specialization constants (SPIR-V),
// and the HIP plugin is AOT only so this entry point is not supported.
UR_APIEXPORT ur_result_t UR_APICALL urProgramSetSpecializationConstants(
    ur_program_handle_t, uint32_t, const ur_specialization_constant_info_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramGetFunctionPointer(
    ur_device_handle_t hDevice, ur_program_handle_t hProgram,
    const char *pFunctionName, void **ppFunctionPointer) {
  // Check if device passed is the same the device bound to the context
  UR_ASSERT(hDevice == hProgram->getDevice(), UR_RESULT_ERROR_INVALID_DEVICE);

  hipFunction_t Func;
  hipError_t Ret = hipModuleGetFunction(&Func, hProgram->get(), pFunctionName);
  *ppFunctionPointer = Func;
  ur_result_t Result = UR_RESULT_SUCCESS;

  if (Ret != hipSuccess && Ret != hipErrorNotFound)
    UR_CHECK_ERROR(Ret);
  if (Ret == hipErrorNotFound) {
    *ppFunctionPointer = 0;
    Result = UR_RESULT_ERROR_INVALID_FUNCTION_NAME;
  }

  return Result;
}
