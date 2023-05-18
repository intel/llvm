//===--------- program.cpp - HIP Adapter ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "program.hpp"

ur_program_handle_t_::ur_program_handle_t_(ur_context_handle_t ctxt)
    : module_{nullptr}, binary_{},
      binarySizeInBytes_{0}, refCount_{1}, context_{ctxt} {
  urContextRetain(context_);
}

ur_program_handle_t_::~ur_program_handle_t_() { urContextRelease(context_); }

ur_result_t ur_program_handle_t_::set_binary(const char *source,
                                             size_t length) {
  // Do not re-set program binary data which has already been set as that will
  // delete the old binary data.
  UR_ASSERT(binary_ == nullptr && binarySizeInBytes_ == 0,
            UR_RESULT_ERROR_INVALID_OPERATION);
  binary_ = source;
  binarySizeInBytes_ = length;
  return UR_RESULT_SUCCESS;
}

ur_result_t ur_program_handle_t_::build_program(const char *build_options) {
  if (build_options) {
    this->buildOptions_ = build_options;
  }

  constexpr const unsigned int numberOfOptions = 4u;

  hipJitOption options[numberOfOptions];
  void *optionVals[numberOfOptions];

  // Pass a buffer for info messages
  options[0] = hipJitOptionInfoLogBuffer;
  optionVals[0] = (void *)infoLog_;
  // Pass the size of the info buffer
  options[1] = hipJitOptionInfoLogBufferSizeBytes;
  optionVals[1] = (void *)(long)MAX_LOG_SIZE;
  // Pass a buffer for error message
  options[2] = hipJitOptionErrorLogBuffer;
  optionVals[2] = (void *)errorLog_;
  // Pass the size of the error buffer
  options[3] = hipJitOptionErrorLogBufferSizeBytes;
  optionVals[3] = (void *)(long)MAX_LOG_SIZE;

  auto result = UR_CHECK_ERROR(
      hipModuleLoadDataEx(&module_, static_cast<const void *>(binary_),
                          numberOfOptions, options, optionVals));

  const auto success = (result == UR_RESULT_SUCCESS);

  buildStatus_ =
      success ? UR_PROGRAM_BUILD_STATUS_SUCCESS : UR_PROGRAM_BUILD_STATUS_ERROR;

  // If no exception, result is correct
  return success ? UR_RESULT_SUCCESS : UR_RESULT_ERROR_PROGRAM_BUILD_FAILURE;
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
///
UR_APIEXPORT ur_result_t UR_APICALL
urProgramCreateWithIL(ur_context_handle_t hContext, const void *pIL,
                      size_t length, const ur_program_properties_t *pProperties,
                      ur_program_handle_t *phProgram) {
  UR_ASSERT(hContext, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  ur_device_handle_t hDevice = hContext->get_device();
  auto pBinary = reinterpret_cast<const uint8_t *>(pIL);

  return urProgramCreateWithBinary(hContext, hDevice, length, pBinary,
                                   pProperties, phProgram);
}

/// HIP will handle the PTX/HIPBIN binaries internally through a call to
/// hipModuleLoadDataEx. So, urProgramCompile and urProgramBuild are equivalent
/// in terms of CUDA adapter. \TODO Implement asynchronous compilation
///
UR_APIEXPORT ur_result_t UR_APICALL
urProgramCompile(ur_context_handle_t hContext, ur_program_handle_t hProgram,
                 const char *pOptions) {
  return urProgramBuild(hContext, hProgram, pOptions);
}

/// Loads the images from a UR program into a CUmodule that can be
/// used later on to extract functions (kernels).
/// See \ref ur_program_handle_t for implementation details.
///
UR_APIEXPORT ur_result_t UR_APICALL urProgramBuild(ur_context_handle_t hContext,
                                                   ur_program_handle_t hProgram,
                                                   const char *pOptions) {
  UR_ASSERT(hProgram, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  ur_result_t retError = UR_RESULT_SUCCESS;

  try {
    ScopedContext active(hProgram->get_context());

    hProgram->build_program(pOptions);

  } catch (ur_result_t err) {
    retError = err;
  }
  return retError;
}

UR_APIEXPORT ur_result_t UR_APICALL
urProgramLink(ur_context_handle_t hContext, uint32_t count,
              const ur_program_handle_t *phPrograms, const char *pOptions,
              ur_program_handle_t *phProgram) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

/// Created a UR program object from a HIP program handle.
/// TODO: Implement this.
/// NOTE: The created UR object takes ownership of the native handle.
///
/// \param[in] nativeHandle The native handle to create UR program object from.
/// \param[in] context The UR context of the program.
/// \param[out] program Set to the UR program object created from native handle.
///
/// \return TBD
UR_APIEXPORT ur_result_t UR_APICALL urProgramCreateWithNativeHandle(
    ur_native_handle_t hNativeProgram, ur_context_handle_t hContext,
    ur_program_handle_t *phProgram) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL
urProgramGetBuildInfo(ur_program_handle_t hProgram, ur_device_handle_t hDevice,
                      ur_program_build_info_t propName, size_t propSize,
                      void *pPropValue, size_t *pPropSizeRet) {
  // Ignore unused parameter
  (void)hDevice;

  UR_ASSERT(hProgram, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);

  switch (propName) {
  case UR_PROGRAM_BUILD_INFO_STATUS: {
    return ReturnValue(hProgram->buildStatus_);
  }
  case UR_PROGRAM_BUILD_INFO_OPTIONS:
    return ReturnValue(hProgram->buildOptions_.c_str());
  case UR_PROGRAM_BUILD_INFO_LOG:
    return ReturnValue(hProgram->infoLog_, hProgram->MAX_LOG_SIZE);
  default:
    break;
  }
  return UR_RESULT_ERROR_INVALID_ENUMERATION;
}

UR_APIEXPORT ur_result_t UR_APICALL
urProgramGetInfo(ur_program_handle_t hProgram, ur_program_info_t propName,
                 size_t propSize, void *pProgramInfo, size_t *pPropSizeRet) {
  UR_ASSERT(hProgram, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  UrReturnHelper ReturnValue(propSize, pProgramInfo, pPropSizeRet);

  switch (propName) {
  case UR_PROGRAM_INFO_REFERENCE_COUNT:
    return ReturnValue(hProgram->get_reference_count());
  case UR_PROGRAM_INFO_CONTEXT:
    return ReturnValue(hProgram->context_);
  case UR_PROGRAM_INFO_NUM_DEVICES:
    return ReturnValue(1u);
  case UR_PROGRAM_INFO_DEVICES:
    return ReturnValue(&hProgram->context_->deviceId_, 1);
  case UR_PROGRAM_INFO_SOURCE:
    return ReturnValue(hProgram->binary_);
  case UR_PROGRAM_INFO_BINARY_SIZES:
    return ReturnValue(&hProgram->binarySizeInBytes_, 1);
  case UR_PROGRAM_INFO_BINARIES:
    return ReturnValue(&hProgram->binary_, 1);
  case UR_PROGRAM_INFO_KERNEL_NAMES:
    return getKernelNames(hProgram);
  default:
    break;
  }
  return UR_RESULT_ERROR_INVALID_ENUMERATION;
}

UR_APIEXPORT ur_result_t UR_APICALL
urProgramRetain(ur_program_handle_t program) {
  UR_ASSERT(program, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(program->get_reference_count() > 0,
            UR_RESULT_ERROR_INVALID_PROGRAM);
  program->increment_reference_count();
  return UR_RESULT_SUCCESS;
}

/// Decreases the reference count of a ur_program_handle_t object.
/// When the reference count reaches 0, it unloads the module from
/// the context.
UR_APIEXPORT ur_result_t UR_APICALL
urProgramRelease(ur_program_handle_t program) {
  UR_ASSERT(program, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  // double delete or someone is messing with the ref count.
  // either way, cannot safely proceed.
  UR_ASSERT(program->get_reference_count() != 0,
            UR_RESULT_ERROR_INVALID_PROGRAM);

  // decrement ref count. If it is 0, delete the program.
  if (program->decrement_reference_count() == 0) {

    std::unique_ptr<ur_program_handle_t_> program_ptr{program};

    ur_result_t result = UR_RESULT_ERROR_INVALID_PROGRAM;

    try {
      ScopedContext active(program->get_context());
      auto hipModule = program->get();
      result = UR_CHECK_ERROR(hipModuleUnload(hipModule));
    } catch (...) {
      result = UR_RESULT_ERROR_OUT_OF_RESOURCES;
    }

    return result;
  }

  return UR_RESULT_SUCCESS;
}

/// Gets the native HIP handle of a UR program object
///
/// \param[in] program The UR program to get the native HIP object of.
/// \param[out] nativeHandle Set to the native handle of the UR program object.
///
/// \return TBD
UR_APIEXPORT ur_result_t UR_APICALL urProgramGetNativeHandle(
    ur_program_handle_t program, ur_native_handle_t *nativeHandle) {
  UR_ASSERT(program, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  *nativeHandle = reinterpret_cast<ur_native_handle_t>(program->get());
  return UR_RESULT_SUCCESS;
}

/// Loads images from a list of PTX or HIPBin binaries.
/// Note: No calls to HIP driver API in this function, only store binaries
/// for later.
///
/// Note: Only supports one device
///
UR_APIEXPORT ur_result_t UR_APICALL urProgramCreateWithBinary(
    ur_context_handle_t hContext, ur_device_handle_t hDevice, size_t size,
    const uint8_t *pBinary, const ur_program_properties_t *pProperties,
    ur_program_handle_t *phProgram) {
  UR_ASSERT(hContext, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(hDevice, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(phProgram, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  UR_ASSERT(pBinary != nullptr && size != 0, UR_RESULT_ERROR_INVALID_BINARY);
  UR_ASSERT(hContext->get_device()->get() == hDevice->get(),
            UR_RESULT_ERROR_INVALID_CONTEXT);

  ur_result_t retError = UR_RESULT_SUCCESS;

  std::unique_ptr<ur_program_handle_t_> retProgram{
      new ur_program_handle_t_{hContext}};

  // TODO: Set metadata here and use reqd_work_group_size information.
  // See urProgramCreateWithBinary in CUDA adapter.

  auto pBinary_string = reinterpret_cast<const char *>(pBinary);
  if (size == 0) {
    size = strlen(pBinary_string) + 1;
  }

  UR_ASSERT(size, UR_RESULT_ERROR_INVALID_SIZE);

  retError = retProgram->set_binary(pBinary_string, size);
  UR_ASSERT(retError == UR_RESULT_SUCCESS, retError);

  *phProgram = retProgram.release();

  return retError;
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
  UR_ASSERT(hDevice, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(hProgram, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(hDevice == hProgram->get_context()->get_device(),
            UR_RESULT_ERROR_INVALID_DEVICE);
  UR_ASSERT(ppFunctionPointer, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  hipFunction_t func;
  hipError_t ret = hipModuleGetFunction(&func, hProgram->get(), pFunctionName);
  *ppFunctionPointer = func;
  ur_result_t retError = UR_RESULT_SUCCESS;

  if (ret != hipSuccess && ret != hipErrorNotFound)
    retError = UR_CHECK_ERROR(ret);
  if (ret == hipErrorNotFound) {
    *ppFunctionPointer = 0;
    retError = UR_RESULT_ERROR_INVALID_FUNCTION_NAME;
  }

  return retError;
}