//===--------- program.cpp - CUDA Adapter ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "program.hpp"

bool getMaxRegistersJitOptionValue(const std::string &build_options,
                                   unsigned int &value) {
  using namespace std::string_view_literals;
  const std::size_t optionPos = build_options.find_first_of("maxrregcount"sv);
  if (optionPos == std::string::npos) {
    return false;
  }

  const std::size_t delimPos = build_options.find('=', optionPos + 1u);
  if (delimPos == std::string::npos) {
    return false;
  }

  const std::size_t length = build_options.length();
  const std::size_t startPos = delimPos + 1u;
  if (delimPos == std::string::npos || startPos >= length) {
    return false;
  }

  std::size_t pos = startPos;
  while (pos < length &&
         std::isdigit(static_cast<unsigned char>(build_options[pos]))) {
    pos++;
  }

  const std::string valueString =
      build_options.substr(startPos, pos - startPos);
  if (valueString.empty()) {
    return false;
  }

  value = static_cast<unsigned int>(std::stoi(valueString));
  return true;
}

ur_program_handle_t_::ur_program_handle_t_(ur_context_handle_t ctxt)
    : module_{nullptr}, binary_{}, binarySizeInBytes_{0}, refCount_{1},
      context_{ctxt}, kernelReqdWorkGroupSizeMD_{} {
  urContextRetain(context_);
}

ur_program_handle_t_::~ur_program_handle_t_() { urContextRelease(context_); }

std::pair<std::string, std::string>
splitMetadataName(const std::string &metadataName) {
  size_t splitPos = metadataName.rfind('@');
  if (splitPos == std::string::npos)
    return std::make_pair(metadataName, std::string{});
  return std::make_pair(metadataName.substr(0, splitPos),
                        metadataName.substr(splitPos, metadataName.length()));
}

ur_result_t
ur_program_handle_t_::set_metadata(const ur_program_metadata_t *metadata,
                                   size_t length) {
  for (size_t i = 0; i < length; ++i) {
    const ur_program_metadata_t metadataElement = metadata[i];
    std::string metadataElementName{metadataElement.pName};

    auto [prefix, tag] = splitMetadataName(metadataElementName);

    if (tag == __SYCL_UR_PROGRAM_METADATA_TAG_REQD_WORK_GROUP_SIZE) {
      // If metadata is reqd_work_group_size, record it for the corresponding
      // kernel name.
      size_t MDElemsSize = metadataElement.size - sizeof(std::uint64_t);

      // Expect between 1 and 3 32-bit integer values.
      UR_ASSERT(MDElemsSize >= sizeof(std::uint32_t) &&
                    MDElemsSize <= sizeof(std::uint32_t) * 3,
                UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE);

      // Get pointer to data, skipping 64-bit size at the start of the data.
      const char *ValuePtr =
          reinterpret_cast<const char *>(metadataElement.value.pData) +
          sizeof(std::uint64_t);
      // Read values and pad with 1's for values not present.
      std::uint32_t reqdWorkGroupElements[] = {1, 1, 1};
      std::memcpy(reqdWorkGroupElements, ValuePtr, MDElemsSize);
      kernelReqdWorkGroupSizeMD_[prefix] =
          std::make_tuple(reqdWorkGroupElements[0], reqdWorkGroupElements[1],
                          reqdWorkGroupElements[2]);
    } else if (tag == __SYCL_UR_PROGRAM_METADATA_GLOBAL_ID_MAPPING) {
      const char *metadataValPtr =
          reinterpret_cast<const char *>(metadataElement.value.pData) +
          sizeof(std::uint64_t);
      const char *metadataValPtrEnd =
          metadataValPtr + metadataElement.size - sizeof(std::uint64_t);
      globalIDMD_[prefix] = std::string{metadataValPtr, metadataValPtrEnd};
    }
  }
  return UR_RESULT_SUCCESS;
}

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

  this->buildOptions_ = build_options;

  constexpr const unsigned int numberOfOptions = 4u;

  std::vector<CUjit_option> options(numberOfOptions);
  std::vector<void *> optionVals(numberOfOptions);

  // Pass a buffer for info messages
  options[0] = CU_JIT_INFO_LOG_BUFFER;
  optionVals[0] = (void *)infoLog_;
  // Pass the size of the info buffer
  options[1] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
  optionVals[1] = (void *)(long)MAX_LOG_SIZE;
  // Pass a buffer for error message
  options[2] = CU_JIT_ERROR_LOG_BUFFER;
  optionVals[2] = (void *)errorLog_;
  // Pass the size of the error buffer
  options[3] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
  optionVals[3] = (void *)(long)MAX_LOG_SIZE;

  if (!buildOptions_.empty()) {
    unsigned int maxRegs;
    bool valid = getMaxRegistersJitOptionValue(buildOptions_, maxRegs);
    if (valid) {
      options.push_back(CU_JIT_MAX_REGISTERS);
      optionVals.push_back(reinterpret_cast<void *>(maxRegs));
    }
  }

  auto result = UR_CHECK_ERROR(
      cuModuleLoadDataEx(&module_, static_cast<const void *>(binary_),
                         options.size(), options.data(), optionVals.data()));

  const auto success = (result == UR_RESULT_SUCCESS);

  buildStatus_ =
      success ? UR_PROGRAM_BUILD_STATUS_SUCCESS : UR_PROGRAM_BUILD_STATUS_ERROR;

  // If no exception, result is correct
  return success ? UR_RESULT_SUCCESS : UR_RESULT_ERROR_PROGRAM_BUILD_FAILURE;
}

/// Finds kernel names by searching for entry points in the PTX source, as the
/// CUDA driver API doesn't expose an operation for this.
/// Note: This is currently only being used by the SYCL program class for the
///       has_kernel method, so an alternative would be to move the has_kernel
///       query to PI and use cuModuleGetFunction to check for a kernel.
/// Note: Another alternative is to add kernel names as metadata, like with
///       reqd_work_group_size.
std::string getKernelNames(ur_program_handle_t) {
  sycl::detail::ur::die("getKernelNames not implemented");
  return {};
}

/// CUDA will handle the PTX/CUBIN binaries internally through CUmodule object.
/// So, urProgramCreateWithIL and urProgramCreateWithBinary are equivalent in
/// terms of CUDA adapter. See \ref urProgramCreateWithBinary.
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

/// CUDA will handle the PTX/CUBIN binaries internally through a call to
/// cuModuleLoadDataEx. So, urProgramCompile and urProgramBuild are equivalent
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

/// Creates a new UR program object that is the outcome of linking all input
/// programs.
/// \TODO Implement linker options, requires mapping of OpenCL to CUDA
///
UR_APIEXPORT ur_result_t UR_APICALL
urProgramLink(ur_context_handle_t hContext, uint32_t count,
              const ur_program_handle_t *phPrograms, const char *pOptions,
              ur_program_handle_t *phProgram) {
  UR_ASSERT(hContext, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(count, UR_RESULT_ERROR_PROGRAM_LINK_FAILURE);
  UR_ASSERT(phPrograms, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  UR_ASSERT(phProgram, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  ur_result_t retError = UR_RESULT_SUCCESS;

  try {
    ScopedContext active(hContext);

    CUlinkState state;
    std::unique_ptr<ur_program_handle_t_> retProgram{
        new ur_program_handle_t_{hContext}};

    retError = UR_CHECK_ERROR(cuLinkCreate(0, nullptr, nullptr, &state));
    try {
      for (size_t i = 0; i < count; ++i) {
        ur_program_handle_t program = phPrograms[i];
        retError = UR_CHECK_ERROR(cuLinkAddData(
            state, CU_JIT_INPUT_PTX, const_cast<char *>(program->binary_),
            program->binarySizeInBytes_, nullptr, 0, nullptr, nullptr));
      }
      void *cubin = nullptr;
      size_t cubinSize = 0;
      retError = UR_CHECK_ERROR(cuLinkComplete(state, &cubin, &cubinSize));

      retError =
          retProgram->set_binary(static_cast<const char *>(cubin), cubinSize);

      retError = retProgram->build_program(pOptions);
    } catch (...) {
      // Upon error attempt cleanup
      UR_CHECK_ERROR(cuLinkDestroy(state));
      throw;
    }

    retError = UR_CHECK_ERROR(cuLinkDestroy(state));
    *phProgram = retProgram.release();

  } catch (ur_result_t err) {
    retError = err;
  }
  return retError;
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
    ur_native_handle_t hNativeProgram, ur_context_handle_t hContext,
    ur_program_handle_t *phProgram) {
  sycl::detail::ur::die(
      "Creation of UR program from native handle not implemented");
  return {};
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
  case UR_PROGRAM_INFO_NUM_KERNELS:
    return ReturnValue(getKernelNames(hProgram).c_str());
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
      auto cuModule = program->get();
      result = UR_CHECK_ERROR(cuModuleUnload(cuModule));
    } catch (...) {
      result = UR_RESULT_ERROR_OUT_OF_RESOURCES;
    }

    return result;
  }

  return UR_RESULT_SUCCESS;
}

/// Gets the native CUDA handle of a UR program object
///
/// \param[in] program The PI program to get the native CUDA object of.
/// \param[out] nativeHandle Set to the native handle of the PI program object.
///
/// \return TBD
UR_APIEXPORT ur_result_t UR_APICALL urProgramGetNativeHandle(
    ur_program_handle_t program, ur_native_handle_t *nativeHandle) {
  UR_ASSERT(program, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  *nativeHandle = reinterpret_cast<ur_native_handle_t>(program->get());
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
  UR_ASSERT(hContext, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(hDevice, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(phProgram, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  UR_ASSERT(pBinary != nullptr && size != 0, UR_RESULT_ERROR_INVALID_BINARY);
  UR_ASSERT(hContext->get_device()->get() == hDevice->get(),
            UR_RESULT_ERROR_INVALID_CONTEXT);

  ur_result_t retError = UR_RESULT_SUCCESS;

  std::unique_ptr<ur_program_handle_t_> retProgram{
      new ur_program_handle_t_{hContext}};

  retError =
      retProgram->set_metadata(pProperties->pMetadatas, pProperties->count);
  UR_ASSERT(retError == UR_RESULT_SUCCESS, retError);

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
