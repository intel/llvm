//===--------- program.cpp - CUDA Adapter ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "program.hpp"

bool getMaxRegistersJitOptionValue(const std::string &BuildOptions,
                                   unsigned int &Value) {
  using namespace std::string_view_literals;
  const std::size_t OptionPos = BuildOptions.find_first_of("maxrregcount"sv);
  if (OptionPos == std::string::npos) {
    return false;
  }

  const std::size_t DelimPos = BuildOptions.find('=', OptionPos + 1u);
  if (DelimPos == std::string::npos) {
    return false;
  }

  const std::size_t Length = BuildOptions.length();
  const std::size_t StartPos = DelimPos + 1u;
  if (DelimPos == std::string::npos || StartPos >= Length) {
    return false;
  }

  std::size_t Pos = StartPos;
  while (Pos < Length &&
         std::isdigit(static_cast<unsigned char>(BuildOptions[Pos]))) {
    Pos++;
  }

  const std::string ValueString = BuildOptions.substr(StartPos, Pos - StartPos);
  if (ValueString.empty()) {
    return false;
  }

  Value = static_cast<unsigned int>(std::stoi(ValueString));
  return true;
}

ur_program_handle_t_::ur_program_handle_t_(ur_context_handle_t Context)
    : Module{nullptr}, Binary{}, BinarySizeInBytes{0}, RefCount{1},
      Context{Context}, KernelReqdWorkGroupSizeMD{} {
  urContextRetain(Context);
}

ur_program_handle_t_::~ur_program_handle_t_() { urContextRelease(Context); }

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

    if (Tag == __SYCL_UR_PROGRAM_METADATA_TAG_REQD_WORK_GROUP_SIZE) {
      // If metadata is reqd_work_group_size, record it for the corresponding
      // kernel name.
      size_t MDElemsSize = MetadataElement.size - sizeof(std::uint64_t);

      // Expect between 1 and 3 32-bit integer values.
      UR_ASSERT(MDElemsSize >= sizeof(std::uint32_t) &&
                    MDElemsSize <= sizeof(std::uint32_t) * 3,
                UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE);

      // Get pointer to data, skipping 64-bit size at the start of the data.
      const char *ValuePtr =
          reinterpret_cast<const char *>(MetadataElement.value.pData) +
          sizeof(std::uint64_t);
      // Read values and pad with 1's for values not present.
      std::uint32_t ReqdWorkGroupElements[] = {1, 1, 1};
      std::memcpy(ReqdWorkGroupElements, ValuePtr, MDElemsSize);
      KernelReqdWorkGroupSizeMD[Prefix] =
          std::make_tuple(ReqdWorkGroupElements[0], ReqdWorkGroupElements[1],
                          ReqdWorkGroupElements[2]);
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

ur_result_t ur_program_handle_t_::buildProgram(const char *BuildOptions) {
  if (BuildOptions) {
    this->BuildOptions = BuildOptions;
  }

  constexpr const unsigned int NumberOfOptions = 4u;

  std::vector<CUjit_option> Options(NumberOfOptions);
  std::vector<void *> OptionVals(NumberOfOptions);

  // Pass a buffer for info messages
  Options[0] = CU_JIT_INFO_LOG_BUFFER;
  OptionVals[0] = (void *)InfoLog;
  // Pass the size of the info buffer
  Options[1] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
  OptionVals[1] = (void *)(long)MaxLogSize;
  // Pass a buffer for error message
  Options[2] = CU_JIT_ERROR_LOG_BUFFER;
  OptionVals[2] = (void *)ErrorLog;
  // Pass the size of the error buffer
  Options[3] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
  OptionVals[3] = (void *)(long)MaxLogSize;

  if (!this->BuildOptions.empty()) {
    unsigned int MaxRegs;
    bool Valid = getMaxRegistersJitOptionValue(BuildOptions, MaxRegs);
    if (Valid) {
      Options.push_back(CU_JIT_MAX_REGISTERS);
      OptionVals.push_back(reinterpret_cast<void *>(MaxRegs));
    }
  }

  UR_CHECK_ERROR(cuModuleLoadDataEx(&Module, static_cast<const void *>(Binary),
                                    Options.size(), Options.data(),
                                    OptionVals.data()));

  BuildStatus = UR_PROGRAM_BUILD_STATUS_SUCCESS;

  // If no exception, result is correct
  return UR_RESULT_SUCCESS;
}

/// Finds kernel names by searching for entry points in the PTX source, as the
/// CUDA driver API doesn't expose an operation for this.
/// Note: This is currently only being used by the SYCL program class for the
///       has_kernel method, so an alternative would be to move the has_kernel
///       query to UR and use cuModuleGetFunction to check for a kernel.
/// Note: Another alternative is to add kernel names as metadata, like with
///       reqd_work_group_size.
ur_result_t getKernelNames(ur_program_handle_t) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

/// CUDA will handle the PTX/CUBIN binaries internally through CUmodule object.
/// So, urProgramCreateWithIL and urProgramCreateWithBinary are equivalent in
/// terms of CUDA adapter. See \ref urProgramCreateWithBinary.
UR_APIEXPORT ur_result_t UR_APICALL
urProgramCreateWithIL(ur_context_handle_t hContext, const void *pIL,
                      size_t length, const ur_program_properties_t *pProperties,
                      ur_program_handle_t *phProgram) {
  ur_device_handle_t hDevice = hContext->getDevice();
  auto pBinary = reinterpret_cast<const uint8_t *>(pIL);

  return urProgramCreateWithBinary(hContext, hDevice, length, pBinary,
                                   pProperties, phProgram);
}

/// CUDA will handle the PTX/CUBIN binaries internally through a call to
/// cuModuleLoadDataEx. So, urProgramCompile and urProgramBuild are equivalent
/// in terms of CUDA adapter. \TODO Implement asynchronous compilation
UR_APIEXPORT ur_result_t UR_APICALL
urProgramCompile(ur_context_handle_t hContext, ur_program_handle_t hProgram,
                 const char *pOptions) {
  return urProgramBuild(hContext, hProgram, pOptions);
}

/// Loads the images from a UR program into a CUmodule that can be
/// used later on to extract functions (kernels).
/// See \ref ur_program_handle_t for implementation details.
UR_APIEXPORT ur_result_t UR_APICALL urProgramBuild(ur_context_handle_t hContext,
                                                   ur_program_handle_t hProgram,
                                                   const char *pOptions) {
  std::ignore = hContext;

  ur_result_t Result = UR_RESULT_SUCCESS;

  try {
    ScopedContext Active(hProgram->getContext());

    hProgram->buildProgram(pOptions);

  } catch (ur_result_t Err) {
    Result = Err;
  }
  return Result;
}

/// Creates a new UR program object that is the outcome of linking all input
/// programs.
/// \TODO Implement linker options, requires mapping of OpenCL to CUDA
UR_APIEXPORT ur_result_t UR_APICALL
urProgramLink(ur_context_handle_t hContext, uint32_t count,
              const ur_program_handle_t *phPrograms, const char *pOptions,
              ur_program_handle_t *phProgram) {
  ur_result_t Result = UR_RESULT_SUCCESS;

  try {
    ScopedContext Active(hContext);

    CUlinkState State;
    std::unique_ptr<ur_program_handle_t_> RetProgram{
        new ur_program_handle_t_{hContext}};

    UR_CHECK_ERROR(cuLinkCreate(0, nullptr, nullptr, &State));
    try {
      for (size_t i = 0; i < count; ++i) {
        ur_program_handle_t Program = phPrograms[i];
        UR_CHECK_ERROR(cuLinkAddData(
            State, CU_JIT_INPUT_PTX, const_cast<char *>(Program->Binary),
            Program->BinarySizeInBytes, nullptr, 0, nullptr, nullptr));
      }
      void *CuBin = nullptr;
      size_t CuBinSize = 0;
      UR_CHECK_ERROR(cuLinkComplete(State, &CuBin, &CuBinSize));

      Result =
          RetProgram->setBinary(static_cast<const char *>(CuBin), CuBinSize);

      Result = RetProgram->buildProgram(pOptions);
    } catch (...) {
      // Upon error attempt cleanup
      UR_CHECK_ERROR(cuLinkDestroy(State));
      throw;
    }

    UR_CHECK_ERROR(cuLinkDestroy(State));
    *phProgram = RetProgram.release();

  } catch (ur_result_t Err) {
    Result = Err;
  }
  return Result;
}

/// Created a UR program object from a CUDA program handle.
/// TODO: Implement this.
/// NOTE: The created UR object takes ownership of the native handle.
///
/// \param[in] nativeHandle The native handle to create UR program object from.
/// \param[in] context The UR context of the program.
/// \param[out] program Set to the UR program object created from native handle.
///
/// \return TBD
UR_APIEXPORT ur_result_t UR_APICALL urProgramCreateWithNativeHandle(
    ur_native_handle_t, ur_context_handle_t,
    const ur_program_native_properties_t *, ur_program_handle_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL
urProgramGetBuildInfo(ur_program_handle_t hProgram, ur_device_handle_t hDevice,
                      ur_program_build_info_t propName, size_t propSize,
                      void *pPropValue, size_t *pPropSizeRet) {
  std::ignore = hDevice;

  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);

  switch (propName) {
  case UR_PROGRAM_BUILD_INFO_STATUS: {
    return ReturnValue(hProgram->BuildStatus);
  }
  case UR_PROGRAM_BUILD_INFO_OPTIONS:
    return ReturnValue(hProgram->BuildOptions.c_str());
  case UR_PROGRAM_BUILD_INFO_LOG:
    return ReturnValue(hProgram->InfoLog, hProgram->MaxLogSize);
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
    return ReturnValue(&hProgram->Context->DeviceID, 1);
  case UR_PROGRAM_INFO_SOURCE:
    return ReturnValue(hProgram->Binary);
  case UR_PROGRAM_INFO_BINARY_SIZES:
    return ReturnValue(&hProgram->BinarySizeInBytes, 1);
  case UR_PROGRAM_INFO_BINARIES:
    return ReturnValue(&hProgram->Binary, 1);
  case UR_PROGRAM_INFO_KERNEL_NAMES:
    /* TODO: Add implementation for getKernelNames */
    UR_ASSERT(getKernelNames(hProgram), UR_RESULT_ERROR_UNSUPPORTED_FEATURE);
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
  case UR_PROGRAM_INFO_NUM_KERNELS:
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
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
      ScopedContext Active(hProgram->getContext());
      auto cuModule = hProgram->get();
      // "0" is a valid handle for a cuModule, so the best way to check if we
      // actually loaded a module and need to unload it is to look at the build
      // status.
      if (hProgram->BuildStatus == UR_PROGRAM_BUILD_STATUS_SUCCESS) {
        UR_CHECK_ERROR(cuModuleUnload(cuModule));
        Result = UR_RESULT_SUCCESS;
      } else if (hProgram->BuildStatus == UR_PROGRAM_BUILD_STATUS_NONE) {
        // Nothing to free.
        Result = UR_RESULT_SUCCESS;
      }
    } catch (...) {
      Result = UR_RESULT_ERROR_OUT_OF_RESOURCES;
    }

    return Result;
  }

  return UR_RESULT_SUCCESS;
}

/// Gets the native CUDA handle of a UR program object
///
/// \param[in] program The UR program handle to get the native CUDA object of.
/// \param[out] nativeHandle Set to the native handle of the UR program object.
///
/// \return ur_result_t
UR_APIEXPORT ur_result_t UR_APICALL urProgramGetNativeHandle(
    ur_program_handle_t hProgram, ur_native_handle_t *nativeHandle) {
  *nativeHandle = reinterpret_cast<ur_native_handle_t>(hProgram->get());
  return UR_RESULT_SUCCESS;
}

/// Loads images from a list of PTX or CUBIN binaries.
/// Note: No calls to CUDA driver API in this function, only store binaries
/// for later.
///
/// Note: Only supports one device
///
UR_APIEXPORT ur_result_t UR_APICALL urProgramCreateWithBinary(
    ur_context_handle_t hContext, ur_device_handle_t hDevice, size_t size,
    const uint8_t *pBinary, const ur_program_properties_t *pProperties,
    ur_program_handle_t *phProgram) {
  UR_ASSERT(hContext->getDevice()->get() == hDevice->get(),
            UR_RESULT_ERROR_INVALID_CONTEXT);
  UR_ASSERT(size, UR_RESULT_ERROR_INVALID_SIZE);

  ur_result_t Result = UR_RESULT_SUCCESS;

  std::unique_ptr<ur_program_handle_t_> RetProgram{
      new ur_program_handle_t_{hContext}};

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

  Result = RetProgram->setBinary(pBinary_string, size);
  UR_ASSERT(Result == UR_RESULT_SUCCESS, Result);

  *phProgram = RetProgram.release();

  return Result;
}

// This entry point is only used for native specialization constants (SPIR-V),
// and the CUDA plugin is AOT only so this entry point is not supported.
UR_APIEXPORT ur_result_t UR_APICALL urProgramSetSpecializationConstants(
    ur_program_handle_t, uint32_t, const ur_specialization_constant_info_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramGetFunctionPointer(
    ur_device_handle_t hDevice, ur_program_handle_t hProgram,
    const char *pFunctionName, void **ppFunctionPointer) {
  // Check if device passed is the same the device bound to the context
  UR_ASSERT(hDevice == hProgram->getContext()->getDevice(),
            UR_RESULT_ERROR_INVALID_DEVICE);

  CUfunction Func;
  CUresult Ret = cuModuleGetFunction(&Func, hProgram->get(), pFunctionName);
  *ppFunctionPointer = Func;
  ur_result_t Result = UR_RESULT_SUCCESS;

  if (Ret != CUDA_SUCCESS && Ret != CUDA_ERROR_NOT_FOUND)
    UR_CHECK_ERROR(Ret);
  if (Ret == CUDA_ERROR_NOT_FOUND) {
    *ppFunctionPointer = 0;
    Result = UR_RESULT_ERROR_INVALID_FUNCTION_NAME;
  }

  return Result;
}
