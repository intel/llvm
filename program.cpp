//===--------- program.cpp - HIP Adapter ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "program.hpp"

ur_program_handle_t_::ur_program_handle_t_(ur_context_handle_t Ctxt)
    : Module{nullptr}, Binary{}, BinarySizeInBytes{0}, RefCount{1},
      Context{Ctxt} {
  urContextRetain(Context);
}

ur_program_handle_t_::~ur_program_handle_t_() { urContextRelease(Context); }

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

/// HIP will handle the PTX/HIPBIN binaries internally through hipModule_t
/// object. So, urProgramCreateWithIL and urProgramCreateWithBinary are
/// equivalent in terms of HIP adapter. See \ref urProgramCreateWithBinary.
UR_APIEXPORT ur_result_t UR_APICALL
urProgramCreateWithIL(ur_context_handle_t hContext, const void *pIL,
                      size_t length, const ur_program_properties_t *pProperties,
                      ur_program_handle_t *phProgram) {
  ur_device_handle_t hDevice = hContext->getDevice();
  const auto pBinary = reinterpret_cast<const uint8_t *>(pIL);

  return urProgramCreateWithBinary(hContext, hDevice, length, pBinary,
                                   pProperties, phProgram);
}

/// HIP will handle the PTX/HIPBIN binaries internally through a call to
/// hipModuleLoadDataEx. So, urProgramCompile and urProgramBuild are equivalent
/// in terms of HIP adapter. \TODO Implement asynchronous compilation
UR_APIEXPORT ur_result_t UR_APICALL
urProgramCompile(ur_context_handle_t hContext, ur_program_handle_t hProgram,
                 const char *pOptions) {
  return urProgramBuild(hContext, hProgram, pOptions);
}

/// Loads the images from a UR program into a hipModule_t that can be
/// used later on to extract functions (kernels).
/// See \ref ur_program_handle_t for implementation details.
UR_APIEXPORT ur_result_t UR_APICALL urProgramBuild(ur_context_handle_t,
                                                   ur_program_handle_t hProgram,
                                                   const char *pOptions) {
  ur_result_t Result = UR_RESULT_SUCCESS;

  try {
    ScopedContext Active(hProgram->getContext()->getDevice());

    hProgram->buildProgram(pOptions);

  } catch (ur_result_t Err) {
    Result = Err;
  }
  return Result;
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
    return ReturnValue(&hProgram->Context->DeviceId, 1);
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
      ScopedContext Active(hProgram->getContext()->getDevice());
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
    const uint8_t *pBinary, const ur_program_properties_t *,
    ur_program_handle_t *phProgram) {
  UR_ASSERT(pBinary != nullptr && size != 0, UR_RESULT_ERROR_INVALID_BINARY);
  UR_ASSERT(hContext->getDevice()->get() == hDevice->get(),
            UR_RESULT_ERROR_INVALID_CONTEXT);

  ur_result_t Result = UR_RESULT_SUCCESS;

  std::unique_ptr<ur_program_handle_t_> RetProgram{
      new ur_program_handle_t_{hContext}};

  // TODO: Set metadata here and use reqd_work_group_size information.
  // See urProgramCreateWithBinary in CUDA adapter.

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
  UR_ASSERT(hDevice == hProgram->getContext()->getDevice(),
            UR_RESULT_ERROR_INVALID_DEVICE);

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
