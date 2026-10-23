//===--------- common.hpp - HIP Adapter -----------------------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
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

#include <cstdint>
#include <type_traits>

// NVIDIA headers typedef hipDeviceptr_t to CUdeviceptr and hipArray_t to
// cudaArray*, with no hipArray struct. That shape is unchanged from ROCm 5.5
// through 10.1. Give hipArray* that meaning so the adapter can keep using
// hipArray* on both platforms.
#if defined(__HIP_PLATFORM_NVIDIA__)
typedef struct cudaArray hipArray;
#endif

// hipDeviceptr_t is void* on AMD and an integer on NVIDIA. These helpers are
// the only conversions that are valid for both.
inline hipDeviceptr_t asHipDevicePtr(const void *Ptr) {
#if defined(__HIP_PLATFORM_NVIDIA__)
  return static_cast<hipDeviceptr_t>(reinterpret_cast<uintptr_t>(Ptr));
#else
  return const_cast<void *>(Ptr);
#endif
}

template <class T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
inline hipDeviceptr_t asHipDevicePtr(T Value) {
#if defined(__HIP_PLATFORM_NVIDIA__)
  return static_cast<hipDeviceptr_t>(Value);
#else
  return reinterpret_cast<hipDeviceptr_t>(static_cast<uintptr_t>(Value));
#endif
}

inline void *asVoidPtr(hipDeviceptr_t Ptr) {
#if defined(__HIP_PLATFORM_NVIDIA__)
  return reinterpret_cast<void *>(static_cast<uintptr_t>(Ptr));
#else
  return Ptr;
#endif
}

// Handles are pointers on AMD and 64-bit integers on NVIDIA, while UR native
// handles are uintptr_t. reinterpret_cast cannot convert between those integer
// types.
template <class To, class From> inline To hipBitCast(From Value) {
  if constexpr (std::is_same_v<To, From>) {
    return Value;
  } else if constexpr (std::is_pointer_v<To> && std::is_pointer_v<From>) {
    return reinterpret_cast<To>(Value);
  } else if constexpr (std::is_pointer_v<To> && std::is_integral_v<From>) {
    return reinterpret_cast<To>(static_cast<uintptr_t>(Value));
  } else if constexpr (std::is_integral_v<To> && std::is_pointer_v<From>) {
    return static_cast<To>(reinterpret_cast<uintptr_t>(Value));
  } else {
    return static_cast<To>(Value);
  }
}

// Before ROCm 6, hipify doesn't support cuArrayGetDescriptor, on AMD the
// hipArray can just be indexed, but on NVidia it is an opaque type and needs to
// go through cuArrayGetDescriptor so implement a utility function to get the
// array properties
inline static hipError_t getArrayDesc(hipArray *Array, hipArray_Format &Format,
                                      size_t &Channels) {
#if defined(__HIP_PLATFORM_NVIDIA__)
  // hipArrayGetDescriptor is behind __CUDACC__ in those NVIDIA headers, so
  // the host compiler never sees it. The array is opaque, so query the driver.
  CUDA_ARRAY_DESCRIPTOR ArrayDesc;
  CUresult Err =
      cuArrayGetDescriptor(&ArrayDesc, reinterpret_cast<CUarray>(Array));
  if (Err == CUDA_SUCCESS) {
    Format = ArrayDesc.Format;
    Channels = ArrayDesc.NumChannels;
    return hipSuccess;
  }
  return hipErrorUnknown;
#elif HIP_VERSION_MAJOR >= 6
  HIP_ARRAY_DESCRIPTOR ArrayDesc;
  hipError_t Err = hipArrayGetDescriptor(&ArrayDesc, Array);
  if (Err == hipSuccess) {
    Format = ArrayDesc.Format;
    Channels = ArrayDesc.NumChannels;
  }
  return Err;
#elif defined(__HIP_PLATFORM_AMD__)
  Format = Array->Format;
  Channels = Array->NumChannels;
  return hipSuccess;
#else
#error ("Must define exactly one of __HIP_PLATFORM_AMD__ or __HIP_PLATFORM_NVIDIA__");
#endif
}

// Those NVIDIA headers define the array, texture, and graph wrappers only
// when __CUDACC__ is set, from ROCm 5.5 through 10.1. The adapter is built
// with the host compiler, so provide the same CUDA translations here.
#if defined(__HIP_PLATFORM_NVIDIA__) && !defined(__CUDACC__)
inline static hipError_t
hipArray3DCreate(hipArray_t *pHandle,
                 const HIP_ARRAY3D_DESCRIPTOR *pAllocateArray) {
  return hipCUResultTohipError(
      cuArray3DCreate(reinterpret_cast<CUarray *>(pHandle), pAllocateArray));
}

inline static hipError_t hipArrayDestroy(hipArray_t Array) {
  return hipCUResultTohipError(
      cuArrayDestroy(reinterpret_cast<CUarray>(Array)));
}

inline static hipError_t hipArrayGetDescriptor(HIP_ARRAY_DESCRIPTOR *Descriptor,
                                               hipArray_t Array) {
  return hipCUResultTohipError(
      cuArrayGetDescriptor(Descriptor, reinterpret_cast<CUarray>(Array)));
}

inline static hipError_t
hipArray3DGetDescriptor(HIP_ARRAY3D_DESCRIPTOR *Descriptor, hipArray_t Array) {
  return hipCUResultTohipError(
      cuArray3DGetDescriptor(Descriptor, reinterpret_cast<CUarray>(Array)));
}

inline static hipError_t
hipTexObjectCreate(hipTextureObject_t *Texture,
                   const HIP_RESOURCE_DESC *ResDesc,
                   const HIP_TEXTURE_DESC *TexDesc,
                   const HIP_RESOURCE_VIEW_DESC *ResViewDesc) {
  return hipCUResultTohipError(
      cuTexObjectCreate(reinterpret_cast<CUtexObject *>(Texture),
                        reinterpret_cast<CUDA_RESOURCE_DESC *>(
                            const_cast<HIP_RESOURCE_DESC *>(ResDesc)),
                        TexDesc, ResViewDesc));
}

inline static hipError_t hipTexObjectDestroy(hipTextureObject_t Texture) {
  return hipCUResultTohipError(
      cuTexObjectDestroy(static_cast<CUtexObject>(Texture)));
}

inline static hipError_t hipMemcpyHtoAAsync(hipArray_t Dst, size_t DstOffset,
                                            const void *Src, size_t ByteCount,
                                            hipStream_t Stream) {
  return hipCUResultTohipError(cuMemcpyHtoAAsync(
      reinterpret_cast<CUarray>(Dst), DstOffset, Src, ByteCount, Stream));
}

inline static hipError_t hipMemcpyAtoHAsync(void *Dst, hipArray_t Src,
                                            size_t SrcOffset, size_t ByteCount,
                                            hipStream_t Stream) {
  return hipCUResultTohipError(cuMemcpyAtoHAsync(
      Dst, reinterpret_cast<CUarray>(Src), SrcOffset, ByteCount, Stream));
}

inline static hipError_t hipGraphCreate(hipGraph_t *Graph, unsigned int Flags) {
  return hipCUDAErrorTohipError(cudaGraphCreate(Graph, Flags));
}

inline static hipError_t hipGraphDestroy(hipGraph_t Graph) {
  return hipCUDAErrorTohipError(cudaGraphDestroy(Graph));
}

inline static hipError_t hipGraphExecDestroy(hipGraphExec_t GraphExec) {
  return hipCUDAErrorTohipError(cudaGraphExecDestroy(GraphExec));
}

inline static hipError_t
hipGraphInstantiateWithFlags(hipGraphExec_t *GraphExec, hipGraph_t Graph,
                             unsigned long long Flags) {
  return hipCUDAErrorTohipError(
      cudaGraphInstantiateWithFlags(GraphExec, Graph, Flags));
}

inline static hipError_t hipGraphLaunch(hipGraphExec_t GraphExec,
                                        hipStream_t Stream) {
  return hipCUDAErrorTohipError(cudaGraphLaunch(GraphExec, Stream));
}

inline static hipError_t hipGraphAddKernelNode(
    hipGraphNode_t *Node, hipGraph_t Graph, const hipGraphNode_t *Dependencies,
    size_t NumDependencies, const hipKernelNodeParams *Params) {
  return hipCUDAErrorTohipError(cudaGraphAddKernelNode(
      Node, Graph, Dependencies, NumDependencies, Params));
}

inline static hipError_t
hipGraphAddMemcpyNode(hipGraphNode_t *Node, hipGraph_t Graph,
                      const hipGraphNode_t *Dependencies,
                      size_t NumDependencies, const hipMemcpy3DParms *Params) {
  return hipCUDAErrorTohipError(cudaGraphAddMemcpyNode(
      Node, Graph, Dependencies, NumDependencies, Params));
}

inline static hipError_t
hipGraphAddMemcpyNode1D(hipGraphNode_t *Node, hipGraph_t Graph,
                        const hipGraphNode_t *Dependencies,
                        size_t NumDependencies, void *Dst, const void *Src,
                        size_t Count, hipMemcpyKind Kind) {
  return hipCUDAErrorTohipError(cudaGraphAddMemcpyNode1D(
      Node, Graph, Dependencies, NumDependencies, Dst, Src, Count, Kind));
}

inline static hipError_t
hipGraphAddMemsetNode(hipGraphNode_t *Node, hipGraph_t Graph,
                      const hipGraphNode_t *Dependencies,
                      size_t NumDependencies, const hipMemsetParams *Params) {
  return hipCUDAErrorTohipError(cudaGraphAddMemsetNode(
      Node, Graph, Dependencies, NumDependencies, Params));
}

inline static hipError_t
hipGraphAddEmptyNode(hipGraphNode_t *Node, hipGraph_t Graph,
                     const hipGraphNode_t *Dependencies,
                     size_t NumDependencies) {
  return hipCUDAErrorTohipError(
      cudaGraphAddEmptyNode(Node, Graph, Dependencies, NumDependencies));
}

inline static hipError_t
hipGraphExecKernelNodeSetParams(hipGraphExec_t GraphExec, hipGraphNode_t Node,
                                const hipKernelNodeParams *Params) {
  return hipCUDAErrorTohipError(
      cudaGraphExecKernelNodeSetParams(GraphExec, Node, Params));
}

inline static hipError_t
hipGraphAddChildGraphNode(hipGraphNode_t *Node, hipGraph_t Graph,
                          const hipGraphNode_t *Dependencies,
                          size_t NumDependencies, hipGraph_t ChildGraph) {
  return hipCUDAErrorTohipError(cudaGraphAddChildGraphNode(
      Node, Graph, Dependencies, NumDependencies, ChildGraph));
}
#endif

// The driver entry points take hipmipmappedArray (CUmipmappedArray). The
// adapter stores hipMipmappedArray_t, which is a different pointer type on
// NVIDIA and the same type on AMD.
#if defined(__HIP_PLATFORM_NVIDIA__)
inline hipError_t hipMipmappedArrayCreate(hipMipmappedArray_t *Handle,
                                          HIP_ARRAY3D_DESCRIPTOR *Descriptor,
                                          unsigned int NumLevels) {
  hipmipmappedArray DriverHandle{};
  hipError_t Err =
      hipMipmappedArrayCreate(&DriverHandle, Descriptor, NumLevels);
  if (Handle) {
    *Handle = reinterpret_cast<hipMipmappedArray_t>(DriverHandle);
  }
  return Err;
}

inline hipError_t hipMipmappedArrayDestroy(hipMipmappedArray_t Handle) {
  return hipMipmappedArrayDestroy(reinterpret_cast<hipmipmappedArray>(Handle));
}

inline hipError_t hipMipmappedArrayGetLevel(hipArray_t *Level,
                                            hipMipmappedArray_t Handle,
                                            unsigned int MipLevel) {
  return hipMipmappedArrayGetLevel(
      Level, reinterpret_cast<hipmipmappedArray>(Handle), MipLevel);
}

inline hipError_t hipMalloc(hipDeviceptr_t *Ptr, size_t Size) {
  void *Raw = nullptr;
  hipError_t Err = ::hipMalloc(&Raw, Size);
  if (Ptr && Err == hipSuccess) {
    *Ptr = asHipDevicePtr(Raw);
  }
  return Err;
}

inline hipError_t hipHostGetDevicePointer(hipDeviceptr_t *DevPtr, void *HostPtr,
                                          unsigned int Flags) {
  void *Raw = nullptr;
  hipError_t Err = ::hipHostGetDevicePointer(&Raw, HostPtr, Flags);
  if (DevPtr && Err == hipSuccess) {
    *DevPtr = asHipDevicePtr(Raw);
  }
  return Err;
}

inline hipError_t hipMemPtrGetInfo(void *Ptr, size_t *Size) {
  hipDeviceptr_t Base{};
  return hipMemGetAddressRange(&Base, Size, asHipDevicePtr(Ptr));
}

#ifndef hipExternalMemoryDedicated
#define hipExternalMemoryDedicated cudaExternalMemoryDedicated
#endif

// NVIDIA headers name this hipFilter_mode. AMD headers name it HIPfilter_mode.
using HIPfilter_mode = CUfilter_mode;
#endif

// Memcpy descriptor array fields are hipArray_t on current NVIDIA headers and
// hipArray* on AMD. One alias lets the existing casts compile on both.
#if defined(__HIP_PLATFORM_NVIDIA__)
typedef hipArray_t hipCUarray;
#elif defined(__HIP_PLATFORM_AMD__)
typedef hipArray *hipCUarray;
#else
#error ("Must define exactly one of __HIP_PLATFORM_AMD__ or __HIP_PLATFORM_NVIDIA__");
#endif

// Headers before ROCm 6 do not define hipMemoryType on NVIDIA. Later headers
// do, and redefining it changes the enum values stored in memcpy descriptors.
#if defined(__HIP_PLATFORM_NVIDIA__) && HIP_VERSION_MAJOR < 6
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

namespace ur::hip {
struct ddi_getter {
  static const ur_dditable_t *value();
};
using handle_base = ur::handle_base<ddi_getter>;
} // namespace ur::hip
