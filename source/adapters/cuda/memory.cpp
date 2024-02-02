//===--------- memory.cpp - CUDA Adapter ----------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cuda.h>

#include "common.hpp"
#include "context.hpp"
#include "memory.hpp"

/// Creates a UR Memory object using a CUDA memory allocation.
/// Can trigger a manual copy depending on the mode.
/// \TODO Implement USE_HOST_PTR using cuHostRegister - See #9789
///
UR_APIEXPORT ur_result_t UR_APICALL urMemBufferCreate(
    ur_context_handle_t hContext, ur_mem_flags_t flags, size_t size,
    const ur_buffer_properties_t *pProperties, ur_mem_handle_t *phBuffer) {
  // Validate flags
  if (flags &
      (UR_MEM_FLAG_USE_HOST_POINTER | UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER)) {
    UR_ASSERT(pProperties && pProperties->pHost,
              UR_RESULT_ERROR_INVALID_HOST_PTR);
  }
  UR_ASSERT(size != 0, UR_RESULT_ERROR_INVALID_BUFFER_SIZE);

  // Currently, USE_HOST_PTR is not implemented using host register
  // since this triggers a weird segfault after program ends.
  // Setting this constant to true enables testing that behavior.
  const bool EnableUseHostPtr = false;
  const bool PerformInitialCopy =
      (flags & UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER) ||
      ((flags & UR_MEM_FLAG_USE_HOST_POINTER) && !EnableUseHostPtr);
  ur_result_t Result = UR_RESULT_SUCCESS;
  ur_mem_handle_t MemObj = nullptr;

  try {
    ScopedContext Active(hContext);
    CUdeviceptr Ptr = 0;
    auto HostPtr = pProperties ? pProperties->pHost : nullptr;

    BufferMem::AllocMode AllocMode = BufferMem::AllocMode::Classic;

    if ((flags & UR_MEM_FLAG_USE_HOST_POINTER) && EnableUseHostPtr) {
      UR_CHECK_ERROR(
          cuMemHostRegister(HostPtr, size, CU_MEMHOSTREGISTER_DEVICEMAP));
      UR_CHECK_ERROR(cuMemHostGetDevicePointer(&Ptr, HostPtr, 0));
      AllocMode = BufferMem::AllocMode::UseHostPtr;
    } else if (flags & UR_MEM_FLAG_ALLOC_HOST_POINTER) {
      UR_CHECK_ERROR(cuMemAllocHost(&HostPtr, size));
      UR_CHECK_ERROR(cuMemHostGetDevicePointer(&Ptr, HostPtr, 0));
      AllocMode = BufferMem::AllocMode::AllocHostPtr;
    } else {
      UR_CHECK_ERROR(cuMemAlloc(&Ptr, size));
      if (flags & UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER) {
        AllocMode = BufferMem::AllocMode::CopyIn;
      }
    }

    ur_mem_handle_t parentBuffer = nullptr;

    auto URMemObj = std::unique_ptr<ur_mem_handle_t_>(new ur_mem_handle_t_{
        hContext, parentBuffer, flags, AllocMode, Ptr, HostPtr, size});
    if (URMemObj != nullptr) {
      MemObj = URMemObj.release();
      if (PerformInitialCopy) {
        // Operates on the default stream of the current CUDA context.
        UR_CHECK_ERROR(cuMemcpyHtoD(Ptr, HostPtr, size));
        // Synchronize with default stream implicitly used by cuMemcpyHtoD
        // to make buffer data available on device before any other UR call
        // uses it.
        CUstream defaultStream = 0;
        UR_CHECK_ERROR(cuStreamSynchronize(defaultStream));
      }
    } else {
      Result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }
  } catch (ur_result_t Err) {
    Result = Err;
  } catch (...) {
    Result = UR_RESULT_ERROR_OUT_OF_RESOURCES;
  }

  *phBuffer = MemObj;

  return Result;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemRetain(ur_mem_handle_t hMem) {
  UR_ASSERT(hMem->getReferenceCount() > 0, UR_RESULT_ERROR_INVALID_MEM_OBJECT);
  hMem->incrementReferenceCount();
  return UR_RESULT_SUCCESS;
}

/// Decreases the reference count of the Mem object.
/// If this is zero, calls the relevant CUDA Free function
/// \return UR_RESULT_SUCCESS unless deallocation error
UR_APIEXPORT ur_result_t UR_APICALL urMemRelease(ur_mem_handle_t hMem) {
  ur_result_t Result = UR_RESULT_SUCCESS;

  try {

    // Do nothing if there are other references
    if (hMem->decrementReferenceCount() > 0) {
      return UR_RESULT_SUCCESS;
    }

    // make sure hMem is released in case checkErrorUR throws
    std::unique_ptr<ur_mem_handle_t_> MemObjPtr(hMem);

    if (hMem->isSubBuffer()) {
      return UR_RESULT_SUCCESS;
    }

    ScopedContext Active(MemObjPtr->getContext());

    if (hMem->MemType == ur_mem_handle_t_::Type::Buffer) {
      auto &BufferImpl = std::get<BufferMem>(MemObjPtr->Mem);
      switch (BufferImpl.MemAllocMode) {
      case BufferMem::AllocMode::CopyIn:
      case BufferMem::AllocMode::Classic:
        UR_CHECK_ERROR(cuMemFree(BufferImpl.Ptr));
        break;
      case BufferMem::AllocMode::UseHostPtr:
        UR_CHECK_ERROR(cuMemHostUnregister(BufferImpl.HostPtr));
        break;
      case BufferMem::AllocMode::AllocHostPtr:
        UR_CHECK_ERROR(cuMemFreeHost(BufferImpl.HostPtr));
      };
    } else if (hMem->MemType == ur_mem_handle_t_::Type::Surface) {
      auto &SurfaceImpl = std::get<SurfaceMem>(MemObjPtr->Mem);
      UR_CHECK_ERROR(cuSurfObjectDestroy(SurfaceImpl.getSurface()));
      UR_CHECK_ERROR(cuArrayDestroy(SurfaceImpl.getArray()));
    }

  } catch (ur_result_t Err) {
    Result = Err;
  } catch (...) {
    Result = UR_RESULT_ERROR_OUT_OF_RESOURCES;
  }

  if (Result != UR_RESULT_SUCCESS) {
    // A reported CUDA error is either an implementation or an asynchronous CUDA
    // error for which it is unclear if the function that reported it succeeded
    // or not. Either way, the state of the program is compromised and likely
    // unrecoverable.
    detail::ur::die("Unrecoverable program state reached in urMemRelease");
  }

  return UR_RESULT_SUCCESS;
}

/// Gets the native CUDA handle of a UR mem object
///
/// \param[in] hMem The UR mem to get the native CUDA object of.
/// \param[out] phNativeMem Set to the native handle of the UR mem object.
///
/// \return UR_RESULT_SUCCESS
UR_APIEXPORT ur_result_t UR_APICALL urMemGetNativeHandle(
    ur_mem_handle_t hMem, ur_device_handle_t, ur_native_handle_t *phNativeMem) {
  *phNativeMem = reinterpret_cast<ur_native_handle_t>(
      std::get<BufferMem>(hMem->Mem).get());
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemGetInfo(ur_mem_handle_t hMemory,
                                                 ur_mem_info_t MemInfoType,
                                                 size_t propSize,
                                                 void *pMemInfo,
                                                 size_t *pPropSizeRet) {
  UR_ASSERT(hMemory->isBuffer(), UR_RESULT_ERROR_INVALID_MEM_OBJECT);

  UrReturnHelper ReturnValue(propSize, pMemInfo, pPropSizeRet);

  ScopedContext Active(hMemory->getContext());

  switch (MemInfoType) {
  case UR_MEM_INFO_SIZE: {
    try {
      size_t AllocSize = 0;
      UR_CHECK_ERROR(cuMemGetAddressRange(
          nullptr, &AllocSize, std::get<BufferMem>(hMemory->Mem).Ptr));
      return ReturnValue(AllocSize);
    } catch (ur_result_t Err) {
      return Err;
    } catch (...) {
      return UR_RESULT_ERROR_UNKNOWN;
    }
  }
  case UR_MEM_INFO_CONTEXT: {
    return ReturnValue(hMemory->getContext());
  }

  default:
    return UR_RESULT_ERROR_INVALID_ENUMERATION;
  }
}

UR_APIEXPORT ur_result_t UR_APICALL urMemBufferCreateWithNativeHandle(
    ur_native_handle_t, ur_context_handle_t, const ur_mem_native_properties_t *,
    ur_mem_handle_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemImageCreateWithNativeHandle(
    ur_native_handle_t, ur_context_handle_t, const ur_image_format_t *,
    const ur_image_desc_t *, const ur_mem_native_properties_t *,
    ur_mem_handle_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

/// \TODO Not implemented
UR_APIEXPORT ur_result_t UR_APICALL urMemImageCreate(
    ur_context_handle_t hContext, ur_mem_flags_t flags,
    const ur_image_format_t *pImageFormat, const ur_image_desc_t *pImageDesc,
    void *pHost, ur_mem_handle_t *phMem) {
  if (flags &
      (UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER | UR_MEM_FLAG_USE_HOST_POINTER)) {
    UR_ASSERT(pHost, UR_RESULT_ERROR_INVALID_HOST_PTR);
  }
  const bool PerformInitialCopy =
      (flags & UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER) ||
      ((flags & UR_MEM_FLAG_USE_HOST_POINTER));

  UR_ASSERT(pImageDesc->stype == UR_STRUCTURE_TYPE_IMAGE_DESC,
            UR_RESULT_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR);
  UR_ASSERT(pImageDesc->type <= UR_MEM_TYPE_IMAGE1D_BUFFER,
            UR_RESULT_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR);
  UR_ASSERT(pImageDesc->numMipLevel == 0,
            UR_RESULT_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR);
  UR_ASSERT(pImageDesc->numSamples == 0,
            UR_RESULT_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR);
  if (!pHost) {
    UR_ASSERT(pImageDesc->rowPitch == 0,
              UR_RESULT_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR);
    UR_ASSERT(pImageDesc->slicePitch == 0,
              UR_RESULT_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR);
  }

  ur_result_t Result = UR_RESULT_SUCCESS;

  // We only support RBGA channel order
  // TODO: check SYCL CTS and spec. May also have to support BGRA
  UR_ASSERT(pImageFormat->channelOrder == UR_IMAGE_CHANNEL_ORDER_RGBA,
            UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION);

  // We have to use cuArray3DCreate, which has some caveats. The height and
  // depth parameters must be set to 0 produce 1D or 2D arrays. pImageDesc gives
  // a minimum value of 1, so we need to convert the answer.
  CUDA_ARRAY3D_DESCRIPTOR ArrayDesc;
  ArrayDesc.NumChannels = 4; // Only support 4 channel image
  ArrayDesc.Flags = 0;       // No flags required
  ArrayDesc.Width = pImageDesc->width;
  if (pImageDesc->type == UR_MEM_TYPE_IMAGE1D) {
    ArrayDesc.Height = 0;
    ArrayDesc.Depth = 0;
  } else if (pImageDesc->type == UR_MEM_TYPE_IMAGE2D) {
    ArrayDesc.Height = pImageDesc->height;
    ArrayDesc.Depth = 0;
  } else if (pImageDesc->type == UR_MEM_TYPE_IMAGE3D) {
    ArrayDesc.Height = pImageDesc->height;
    ArrayDesc.Depth = pImageDesc->depth;
  }

  // We need to get this now in bytes for calculating the total image size later
  size_t PixelTypeSizeBytes;

  switch (pImageFormat->channelType) {
  case UR_IMAGE_CHANNEL_TYPE_UNORM_INT8:
  case UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8:
    ArrayDesc.Format = CU_AD_FORMAT_UNSIGNED_INT8;
    PixelTypeSizeBytes = 1;
    break;
  case UR_IMAGE_CHANNEL_TYPE_SIGNED_INT8:
    ArrayDesc.Format = CU_AD_FORMAT_SIGNED_INT8;
    PixelTypeSizeBytes = 1;
    break;
  case UR_IMAGE_CHANNEL_TYPE_UNORM_INT16:
  case UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16:
    ArrayDesc.Format = CU_AD_FORMAT_UNSIGNED_INT16;
    PixelTypeSizeBytes = 2;
    break;
  case UR_IMAGE_CHANNEL_TYPE_SIGNED_INT16:
    ArrayDesc.Format = CU_AD_FORMAT_SIGNED_INT16;
    PixelTypeSizeBytes = 2;
    break;
  case UR_IMAGE_CHANNEL_TYPE_HALF_FLOAT:
    ArrayDesc.Format = CU_AD_FORMAT_HALF;
    PixelTypeSizeBytes = 2;
    break;
  case UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32:
    ArrayDesc.Format = CU_AD_FORMAT_UNSIGNED_INT32;
    PixelTypeSizeBytes = 4;
    break;
  case UR_IMAGE_CHANNEL_TYPE_SIGNED_INT32:
    ArrayDesc.Format = CU_AD_FORMAT_SIGNED_INT32;
    PixelTypeSizeBytes = 4;
    break;
  case UR_IMAGE_CHANNEL_TYPE_FLOAT:
    ArrayDesc.Format = CU_AD_FORMAT_FLOAT;
    PixelTypeSizeBytes = 4;
    break;
  default:
    detail::ur::die(
        "urMemImageCreate given unsupported image_channel_data_type");
  }

  // When a dimension isn't used pImageDesc has the size set to 1
  size_t PixelSizeBytes =
      PixelTypeSizeBytes * 4; // 4 is the only number of channels we support
  size_t ImageSizeBytes = PixelSizeBytes * pImageDesc->width *
                          pImageDesc->height * pImageDesc->depth;

  ScopedContext Active(hContext);
  CUarray ImageArray = nullptr;
  try {
    UR_CHECK_ERROR(cuArray3DCreate(&ImageArray, &ArrayDesc));
  } catch (ur_result_t Err) {
    if (Err == UR_RESULT_ERROR_INVALID_VALUE) {
      return UR_RESULT_ERROR_INVALID_IMAGE_SIZE;
    }
    return Err;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }

  try {
    if (PerformInitialCopy) {
      // We have to use a different copy function for each image dimensionality
      if (pImageDesc->type == UR_MEM_TYPE_IMAGE1D) {
        UR_CHECK_ERROR(cuMemcpyHtoA(ImageArray, 0, pHost, ImageSizeBytes));
      } else if (pImageDesc->type == UR_MEM_TYPE_IMAGE2D) {
        CUDA_MEMCPY2D CpyDesc;
        memset(&CpyDesc, 0, sizeof(CpyDesc));
        CpyDesc.srcMemoryType = CUmemorytype_enum::CU_MEMORYTYPE_HOST;
        CpyDesc.srcHost = pHost;
        CpyDesc.dstMemoryType = CUmemorytype_enum::CU_MEMORYTYPE_ARRAY;
        CpyDesc.dstArray = ImageArray;
        CpyDesc.WidthInBytes = PixelSizeBytes * pImageDesc->width;
        CpyDesc.Height = pImageDesc->height;
        UR_CHECK_ERROR(cuMemcpy2D(&CpyDesc));
      } else if (pImageDesc->type == UR_MEM_TYPE_IMAGE3D) {
        CUDA_MEMCPY3D CpyDesc;
        memset(&CpyDesc, 0, sizeof(CpyDesc));
        CpyDesc.srcMemoryType = CUmemorytype_enum::CU_MEMORYTYPE_HOST;
        CpyDesc.srcHost = pHost;
        CpyDesc.dstMemoryType = CUmemorytype_enum::CU_MEMORYTYPE_ARRAY;
        CpyDesc.dstArray = ImageArray;
        CpyDesc.WidthInBytes = PixelSizeBytes * pImageDesc->width;
        CpyDesc.Height = pImageDesc->height;
        CpyDesc.Depth = pImageDesc->depth;
        UR_CHECK_ERROR(cuMemcpy3D(&CpyDesc));
      }
    }

    // CUDA_RESOURCE_DESC is a union of different structs, shown here
    // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TEXOBJECT.html
    // We need to fill it as described here to use it for a surface or texture
    // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__SURFOBJECT.html
    // CUDA_RESOURCE_DESC::resType must be CU_RESOURCE_TYPE_ARRAY and
    // CUDA_RESOURCE_DESC::res::array::hArray must be set to a valid CUDA array
    // handle.
    // CUDA_RESOURCE_DESC::flags must be set to zero

    CUDA_RESOURCE_DESC ImageResDesc;
    ImageResDesc.res.array.hArray = ImageArray;
    ImageResDesc.resType = CU_RESOURCE_TYPE_ARRAY;
    ImageResDesc.flags = 0;

    CUsurfObject Surface;
    UR_CHECK_ERROR(cuSurfObjectCreate(&Surface, &ImageResDesc));

    auto MemObj = std::unique_ptr<ur_mem_handle_t_>(new ur_mem_handle_t_(
        hContext, ImageArray, Surface, flags, pImageDesc->type, phMem));

    if (MemObj == nullptr) {
      return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

    *phMem = MemObj.release();
  } catch (ur_result_t Err) {
    if (ImageArray) {
      cuArrayDestroy(ImageArray);
    }
    return Err;
  } catch (...) {
    if (ImageArray) {
      cuArrayDestroy(ImageArray);
    }
    return UR_RESULT_ERROR_UNKNOWN;
  }

  return Result;
}

/// \TODO Not implemented
UR_APIEXPORT ur_result_t UR_APICALL urMemImageGetInfo(ur_mem_handle_t,
                                                      ur_image_info_t, size_t,
                                                      void *, size_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

/// Implements a buffer partition in the CUDA backend.
/// A buffer partition (or a sub-buffer, in OpenCL terms) is simply implemented
/// as an offset over an existing CUDA allocation.
UR_APIEXPORT ur_result_t UR_APICALL urMemBufferPartition(
    ur_mem_handle_t hBuffer, ur_mem_flags_t flags,
    ur_buffer_create_type_t bufferCreateType, const ur_buffer_region_t *pRegion,
    ur_mem_handle_t *phMem) {
  UR_ASSERT(hBuffer, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT((flags & UR_MEM_FLAGS_MASK) == 0,
            UR_RESULT_ERROR_INVALID_ENUMERATION);
  UR_ASSERT(hBuffer->isBuffer(), UR_RESULT_ERROR_INVALID_MEM_OBJECT);
  UR_ASSERT(!hBuffer->isSubBuffer(), UR_RESULT_ERROR_INVALID_MEM_OBJECT);

  // Default value for flags means UR_MEM_FLAG_READ_WRITE.
  if (flags == 0) {
    flags = UR_MEM_FLAG_READ_WRITE;
  }

  UR_ASSERT(!(flags &
              (UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER |
               UR_MEM_FLAG_ALLOC_HOST_POINTER | UR_MEM_FLAG_USE_HOST_POINTER)),
            UR_RESULT_ERROR_INVALID_VALUE);
  if (hBuffer->MemFlags & UR_MEM_FLAG_WRITE_ONLY) {
    UR_ASSERT(!(flags & (UR_MEM_FLAG_READ_WRITE | UR_MEM_FLAG_READ_ONLY)),
              UR_RESULT_ERROR_INVALID_VALUE);
  }
  if (hBuffer->MemFlags & UR_MEM_FLAG_READ_ONLY) {
    UR_ASSERT(!(flags & (UR_MEM_FLAG_READ_WRITE | UR_MEM_FLAG_WRITE_ONLY)),
              UR_RESULT_ERROR_INVALID_VALUE);
  }

  UR_ASSERT(bufferCreateType == UR_BUFFER_CREATE_TYPE_REGION,
            UR_RESULT_ERROR_INVALID_ENUMERATION);
  UR_ASSERT(pRegion != nullptr, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  UR_ASSERT(phMem, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  UR_ASSERT(pRegion->size != 0u, UR_RESULT_ERROR_INVALID_BUFFER_SIZE);

  auto &BufferImpl = std::get<BufferMem>(hBuffer->Mem);

  assert((pRegion->origin <= (pRegion->origin + pRegion->size)) && "Overflow");
  UR_ASSERT(((pRegion->origin + pRegion->size) <= BufferImpl.getSize()),
            UR_RESULT_ERROR_INVALID_BUFFER_SIZE);
  // Retained indirectly due to retaining parent buffer below.
  ur_context_handle_t Context = hBuffer->Context;

  BufferMem::AllocMode AllocMode = BufferMem::AllocMode::Classic;

  assert(BufferImpl.Ptr != BufferMem::native_type{0});
  BufferMem::native_type Ptr = BufferImpl.Ptr + pRegion->origin;

  void *HostPtr = nullptr;
  if (BufferImpl.HostPtr) {
    HostPtr = static_cast<char *>(BufferImpl.HostPtr) + pRegion->origin;
  }

  std::unique_ptr<ur_mem_handle_t_> MemObj{nullptr};
  try {
    MemObj = std::unique_ptr<ur_mem_handle_t_>{new ur_mem_handle_t_{
        Context, hBuffer, flags, AllocMode, Ptr, HostPtr, pRegion->size}};
  } catch (ur_result_t Err) {
    *phMem = nullptr;
    return Err;
  } catch (...) {
    *phMem = nullptr;
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  }

  *phMem = MemObj.release();
  return UR_RESULT_SUCCESS;
}
