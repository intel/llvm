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
extern thread_local ur_result_t ErrorMessageCode;
extern thread_local char ErrorMessage[MaxMessageSize];

// Utility function for setting a message and warning
[[maybe_unused]] void setErrorMessage(const char *Message,
                                      ur_result_t ErrorCode);

/// ------ Error handling, matching OpenCL plugin semantics.
namespace detail {
namespace ur {

// Report error and no return (keeps compiler from printing warnings).
// TODO: Probably change that to throw a catchable exception,
//       but for now it is useful to see every failure.
//
[[noreturn]] void die(const char *pMessage);

// Reports error messages
void hipPrint(const char *pMessage);

void assertion(bool Condition, const char *pMessage = nullptr);

} // namespace ur
} // namespace detail

/// RAII object that calls the reference count release function on the held UR
/// object on destruction.
///
/// The `dismiss` function stops the release from happening on destruction.
template <typename T> class ReleaseGuard {
private:
  T Captive;

  static ur_result_t callRelease(ur_device_handle_t Captive) {
    return urDeviceRelease(Captive);
  }

  static ur_result_t callRelease(ur_context_handle_t Captive) {
    return urContextRelease(Captive);
  }

  static ur_result_t callRelease(ur_mem_handle_t Captive) {
    return urMemRelease(Captive);
  }

  static ur_result_t callRelease(ur_program_handle_t Captive) {
    return urProgramRelease(Captive);
  }

  static ur_result_t callRelease(ur_kernel_handle_t Captive) {
    return urKernelRelease(Captive);
  }

  static ur_result_t callRelease(ur_queue_handle_t Captive) {
    return urQueueRelease(Captive);
  }

  static ur_result_t callRelease(ur_event_handle_t Captive) {
    return urEventRelease(Captive);
  }

public:
  ReleaseGuard() = delete;
  /// Obj can be `nullptr`.
  explicit ReleaseGuard(T Obj) : Captive(Obj) {}
  ReleaseGuard(ReleaseGuard &&Other) noexcept : Captive(Other.Captive) {
    Other.Captive = nullptr;
  }

  ReleaseGuard(const ReleaseGuard &) = delete;

  /// Calls the related UR object release function if the object held is not
  /// `nullptr` or if `dismiss` has not been called.
  ~ReleaseGuard() {
    if (Captive != nullptr) {
      ur_result_t ret = callRelease(Captive);
      if (ret != UR_RESULT_SUCCESS) {
        // A reported HIP error is either an implementation or an asynchronous
        // HIP error for which it is unclear if the function that reported it
        // succeeded or not. Either way, the state of the program is compromised
        // and likely unrecoverable.
        detail::ur::die("Unrecoverable program state reached in piMemRelease");
      }
    }
  }

  ReleaseGuard &operator=(const ReleaseGuard &) = delete;

  ReleaseGuard &operator=(ReleaseGuard &&Other) {
    Captive = Other.Captive;
    Other.Captive = nullptr;
    return *this;
  }

  /// End the guard and do not release the reference count of the held
  /// UR object.
  void dismiss() { Captive = nullptr; }
};
