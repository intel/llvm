//===--------- common.hpp - HIP Adapter -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//
#pragma once

#include <hip/hip_runtime.h>
#include <sycl/detail/defines.hpp>
#include <ur/ur.hpp>

// Hipify doesn't support cuArrayGetDescriptor, on AMD the hipArray can just be
// indexed, but on NVidia it is an opaque type and needs to go through
// cuArrayGetDescriptor so implement a utility function to get the array
// properties
inline void getArrayDesc(hipArray *array, hipArray_Format &format,
                         size_t &channels) {
#if defined(__HIP_PLATFORM_AMD__)
  format = array->Format;
  channels = array->NumChannels;
#elif defined(__HIP_PLATFORM_NVIDIA__)
  CUDA_ARRAY_DESCRIPTOR arrayDesc;
  cuArrayGetDescriptor(&arrayDesc, (CUarray)array);

  format = arrayDesc.Format;
  channels = arrayDesc.NumChannels;
#else
#error("Must define exactly one of __HIP_PLATFORM_AMD__ or __HIP_PLATFORM_NVIDIA__");
#endif
}

// NVidia HIP headers guard hipArray3DCreate behind __CUDACC__, this does not
// seem to be required and we're not using nvcc to build the UR HIP adapter so
// add the translation function here
#if defined(__HIP_PLATFORM_NVIDIA__) && !defined(__CUDACC__)
inline static hipError_t
hipArray3DCreate(hiparray *pHandle,
                 const HIP_ARRAY3D_DESCRIPTOR *pAllocateArray) {
  return hipCUResultTohipError(cuArray3DCreate(pHandle, pAllocateArray));
}
#endif

// hipArray gets turned into cudaArray when using the HIP NVIDIA platform, and
// some CUDA APIs use cudaArray* and others use CUarray, these two represent the
// same type, however when building cudaArray appears as an opaque type, so it
// needs to be explicitly casted to CUarray. In order for this to work for both
// AMD and NVidia we introduce an second hipArray type that will be CUarray for
// NVIDIA and hipArray* for AMD so that we can place the explicit casts when
// necessary for NVIDIA and they will be no-ops for AMD.
#if defined(__HIP_PLATFORM_NVIDIA__)
typedef CUarray hipCUarray;
#elif defined(__HIP_PLATFORM_AMD__)
typedef hipArray *hipCUarray;
#else
#error("Must define exactly one of __HIP_PLATFORM_AMD__ or __HIP_PLATFORM_NVIDIA__");
#endif

// Add missing HIP to CUDA defines
#if defined(__HIP_PLATFORM_NVIDIA__)
#define hipMemoryType CUmemorytype
#define hipMemoryTypeHost CU_MEMORYTYPE_HOST
#define hipMemoryTypeDevice CU_MEMORYTYPE_DEVICE
#define hipMemoryTypeArray CU_MEMORYTYPE_ARRAY
#define hipMemoryTypeUnified CU_MEMORYTYPE_UNIFIED
#endif

ur_result_t map_error_ur(hipError_t result);

ur_result_t check_error_ur(hipError_t result, const char *function, int line,
                           const char *file);

#define UR_CHECK_ERROR(result)                                                 \
  check_error_ur(result, __func__, __LINE__, __FILE__)

std::string getHipVersionString();

/// ------ Error handling, matching OpenCL plugin semantics.
namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {
namespace ur {

// Report error and no return (keeps compiler from printing warnings).
// TODO: Probably change that to throw a catchable exception,
//       but for now it is useful to see every failure.
//
[[noreturn]] void die(const char *Message);

// Reports error messages
void hipPrint(const char *Message);

void assertion(bool Condition, const char *Message = nullptr);

} // namespace ur
} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl