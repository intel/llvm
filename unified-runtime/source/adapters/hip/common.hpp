//===--------- common.hpp - HIP Adapter -----------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#ifdef SYCL_ENABLE_KERNEL_FUSION
#ifdef UR_COMGR_VERSION4_INCLUDE
#include <amd_comgr.h>
#else
#include <amd_comgr/amd_comgr.h>
#endif
#endif
#include <hip/hip_runtime.h>
#include <ur/ur.hpp>

// Before ROCm 6, hipify doesn't support cuArrayGetDescriptor, on AMD the
// hipArray can just be indexed, but on NVidia it is an opaque type and needs to
// go through cuArrayGetDescriptor so implement a utility function to get the
// array properties
inline static hipError_t getArrayDesc(hipArray *Array, hipArray_Format &Format,
                                      size_t &Channels) {
#if HIP_VERSION_MAJOR >= 6
  HIP_ARRAY_DESCRIPTOR ArrayDesc;
  hipError_t err = hipArrayGetDescriptor(&ArrayDesc, Array);
  if (err == hipSuccess) {
    Format = ArrayDesc.Format;
    Channels = ArrayDesc.NumChannels;
  }
  return err;
#else
#if defined(__HIP_PLATFORM_AMD__)
  Format = Array->Format;
  Channels = Array->NumChannels;
  return hipSuccess;
#elif defined(__HIP_PLATFORM_NVIDIA__)
  CUDA_ARRAY_DESCRIPTOR ArrayDesc;
  CUresult err = cuArrayGetDescriptor(&ArrayDesc, (CUarray)Array);
  if (err == CUDA_SUCCESS) {
    Format = ArrayDesc.Format;
    Channels = ArrayDesc.NumChannels;
    return hipSuccess;
  } else {
    return hipErrorUnknown; // No easy way to map CUerror to hipError
  }
#else
#error("Must define exactly one of __HIP_PLATFORM_AMD__ or __HIP_PLATFORM_NVIDIA__");
#endif
#endif
}

// HIP on NVIDIA headers guard hipArray3DCreate behind __CUDACC__, this does not
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

ur_result_t mapErrorUR(hipError_t Result);

#ifdef SYCL_ENABLE_KERNEL_FUSION
void checkErrorUR(amd_comgr_status_t Result, const char *Function, int Line,
                  const char *File);
#endif
void checkErrorUR(hipError_t Result, const char *Function, int Line,
                  const char *File);
void checkErrorUR(ur_result_t Result, const char *Function, int Line,
                  const char *File);

#define UR_CHECK_ERROR(result)                                                 \
  checkErrorUR(result, __func__, __LINE__, __FILE__)

hipError_t getHipVersionString(std::string &Version);

constexpr size_t MaxMessageSize = 256;
extern thread_local int32_t ErrorMessageCode;
extern thread_local char ErrorMessage[MaxMessageSize];

// Utility function for setting a message and warning
[[maybe_unused]] void setErrorMessage(const char *Message, int32_t ErrorCode);

// Helper method to return a (non-null) pointer's attributes, or std::nullopt in
// the case that the pointer is unknown to the HIP subsystem.
inline static std::optional<hipPointerAttribute_t>
getPointerAttributes(const void *pMem) {
  // do not throw if hipPointerGetAttributes returns hipErrorInvalidValue
  hipPointerAttribute_t hipPointerAttributes;
  hipError_t Ret = hipPointerGetAttributes(&hipPointerAttributes, pMem);
  if (Ret == hipErrorInvalidValue && pMem) {
    // pointer non-null but not known to the HIP subsystem
    return std::nullopt;
  }
  // Direct usage of the function, instead of UR_CHECK_ERROR, so we can get
  // the line offset.
  checkErrorUR(Ret, __func__, __LINE__ - 7, __FILE__);
  // ROCm 6.0.0 introduces hipMemoryTypeUnregistered in the hipMemoryType
  // enum to mark unregistered allocations (i.e., via system allocators).
#if HIP_VERSION_MAJOR >= 6
  if (hipPointerAttributes.type == hipMemoryTypeUnregistered) {
    // pointer not known to the HIP subsystem
    return std::nullopt;
  }
#endif
  return hipPointerAttributes;
}

// Helper method to abstract away the fact that retrieving a pointer's memory
// type differs depending on the version of HIP.
inline static unsigned getMemoryType(hipPointerAttribute_t hipPointerAttrs) {
#if HIP_VERSION >= 50600000
  return hipPointerAttrs.type;
#else
  return hipPointerAttrs.memoryType;
#endif
}
