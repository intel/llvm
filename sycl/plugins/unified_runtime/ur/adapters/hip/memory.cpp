//===--------- memory.cpp - HIP Adapter -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "memory.hpp"
#include "context.hpp"
#include <cassert>

/// Decreases the reference count of the Mem object.
/// If this is zero, calls the relevant HIP Free function
/// \return UR_RESULT_SUCCESS unless deallocation error
UR_APIEXPORT ur_result_t UR_APICALL urMemRelease(ur_mem_handle_t hMem) {
  ur_result_t Result = UR_RESULT_SUCCESS;

  try {

    // Do nothing if there are other references
    if (hMem->decrementReferenceCount() > 0) {
      return UR_RESULT_SUCCESS;
    }

    // make sure memObj is released in case UR_CHECK_ERROR throws
    std::unique_ptr<ur_mem_handle_t_> uniqueMemObj(hMem);

    if (hMem->isSubBuffer()) {
      return UR_RESULT_SUCCESS;
    }

    ScopedContext Active(uniqueMemObj->getContext()->getDevice());

    if (hMem->MemType == ur_mem_handle_t_::Type::Buffer) {
      switch (uniqueMemObj->Mem.BufferMem.MemAllocMode) {
      case ur_mem_handle_t_::MemImpl::BufferMem::AllocMode::CopyIn:
      case ur_mem_handle_t_::MemImpl::BufferMem::AllocMode::Classic:
        UR_CHECK_ERROR(hipFree((void *)uniqueMemObj->Mem.BufferMem.Ptr));
        break;
      case ur_mem_handle_t_::MemImpl::BufferMem::AllocMode::UseHostPtr:
        UR_CHECK_ERROR(hipHostUnregister(uniqueMemObj->Mem.BufferMem.HostPtr));
        break;
      case ur_mem_handle_t_::MemImpl::BufferMem::AllocMode::AllocHostPtr:
        UR_CHECK_ERROR(hipFreeHost(uniqueMemObj->Mem.BufferMem.HostPtr));
      };
    }

    else if (hMem->MemType == ur_mem_handle_t_::Type::Surface) {
      UR_CHECK_ERROR(
          hipDestroySurfaceObject(uniqueMemObj->Mem.SurfaceMem.getSurface()));
      auto Array = uniqueMemObj->Mem.SurfaceMem.getArray();
      UR_CHECK_ERROR(hipFreeArray(Array));
    }

  } catch (ur_result_t Err) {
    Result = Err;
  } catch (...) {
    Result = UR_RESULT_ERROR_OUT_OF_RESOURCES;
  }

  if (Result != UR_RESULT_SUCCESS) {
    // A reported HIP error is either an implementation or an asynchronous HIP
    // error for which it is unclear if the function that reported it succeeded
    // or not. Either way, the state of the program is compromised and likely
    // unrecoverable.
    detail::ur::die("Unrecoverable program state reached in urMemRelease");
  }

  return UR_RESULT_SUCCESS;
}

/// Creates a UR Memory object using a HIP memory allocation.
/// Can trigger a manual copy depending on the mode.
/// \TODO Implement USE_HOST_PTR using hipHostRegister - See #9789
UR_APIEXPORT ur_result_t UR_APICALL urMemBufferCreate(
    ur_context_handle_t hContext, ur_mem_flags_t flags, size_t size,
    const ur_buffer_properties_t *pProperties, ur_mem_handle_t *phBuffer) {
  // Validate flags
  UR_ASSERT((flags & UR_MEM_FLAGS_MASK) == 0,
            UR_RESULT_ERROR_INVALID_ENUMERATION);
  if (flags &
      (UR_MEM_FLAG_USE_HOST_POINTER | UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER)) {
    UR_ASSERT(pProperties && pProperties->pHost,
              UR_RESULT_ERROR_INVALID_HOST_PTR);
  }
  // Need input memory object
  UR_ASSERT(size != 0, UR_RESULT_ERROR_INVALID_BUFFER_SIZE);

  // Currently, USE_HOST_PTR is not implemented using host register
  // since this triggers a weird segfault after program ends.
  // Setting this constant to true enables testing that behavior.
  const bool EnableUseHostPtr = false;
  const bool PerformInitialCopy =
      (flags & UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER) ||
      ((flags & UR_MEM_FLAG_USE_HOST_POINTER) && !EnableUseHostPtr);
  ur_result_t Result = UR_RESULT_SUCCESS;
  ur_mem_handle_t RetMemObj = nullptr;

  try {
    ScopedContext Active(hContext->getDevice());
    void *Ptr;
    auto pHost = pProperties ? pProperties->pHost : nullptr;
    ur_mem_handle_t_::MemImpl::BufferMem::AllocMode AllocMode =
        ur_mem_handle_t_::MemImpl::BufferMem::AllocMode::Classic;

    if ((flags & UR_MEM_FLAG_USE_HOST_POINTER) && EnableUseHostPtr) {
      UR_CHECK_ERROR(hipHostRegister(pHost, size, hipHostRegisterMapped));
      UR_CHECK_ERROR(hipHostGetDevicePointer(&Ptr, pHost, 0));
      AllocMode = ur_mem_handle_t_::MemImpl::BufferMem::AllocMode::UseHostPtr;
    } else if (flags & UR_MEM_FLAG_ALLOC_HOST_POINTER) {
      UR_CHECK_ERROR(hipHostMalloc(&pHost, size));
      UR_CHECK_ERROR(hipHostGetDevicePointer(&Ptr, pHost, 0));
      AllocMode = ur_mem_handle_t_::MemImpl::BufferMem::AllocMode::AllocHostPtr;
    } else {
      UR_CHECK_ERROR(hipMalloc(&Ptr, size));
      if (flags & UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER) {
        AllocMode = ur_mem_handle_t_::MemImpl::BufferMem::AllocMode::CopyIn;
      }
    }

    if (Result == UR_RESULT_SUCCESS) {
      ur_mem_handle_t parentBuffer = nullptr;

      auto DevPtr =
          reinterpret_cast<ur_mem_handle_t_::MemImpl::BufferMem::native_type>(
              Ptr);
      auto URMemObj = std::unique_ptr<ur_mem_handle_t_>(new ur_mem_handle_t_{
          hContext, parentBuffer, flags, AllocMode, DevPtr, pHost, size});
      if (URMemObj != nullptr) {
        RetMemObj = URMemObj.release();
        if (PerformInitialCopy) {
          // Operates on the default stream of the current HIP context.
          UR_CHECK_ERROR(hipMemcpyHtoD(DevPtr, pHost, size));
          // Synchronize with default stream implicitly used by hipMemcpyHtoD
          // to make buffer data available on device before any other UR call
          // uses it.
          if (Result == UR_RESULT_SUCCESS) {
            hipStream_t defaultStream = 0;
            UR_CHECK_ERROR(hipStreamSynchronize(defaultStream));
          }
        }
      } else {
        Result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
      }
    }
  } catch (ur_result_t Err) {
    Result = Err;
  } catch (...) {
    Result = UR_RESULT_ERROR_OUT_OF_RESOURCES;
  }

  *phBuffer = RetMemObj;

  return Result;
}

/// Implements a buffer partition in the HIP backend.
/// A buffer partition (or a sub-buffer, in OpenCL terms) is simply implemented
/// as an offset over an existing HIP allocation.
UR_APIEXPORT ur_result_t UR_APICALL urMemBufferPartition(
    ur_mem_handle_t hBuffer, ur_mem_flags_t flags,
    ur_buffer_create_type_t bufferCreateType, const ur_buffer_region_t *pRegion,
    ur_mem_handle_t *phMem) {
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

  UR_ASSERT(pRegion->size != 0u, UR_RESULT_ERROR_INVALID_BUFFER_SIZE);

  UR_ASSERT(
      ((pRegion->origin + pRegion->size) <= hBuffer->Mem.BufferMem.getSize()),
      UR_RESULT_ERROR_INVALID_BUFFER_SIZE);
  // Retained indirectly due to retaining parent buffer below.
  ur_context_handle_t Context = hBuffer->Context;
  ur_mem_handle_t_::MemImpl::BufferMem::AllocMode AllocMode =
      ur_mem_handle_t_::MemImpl::BufferMem::AllocMode::Classic;

  UR_ASSERT(hBuffer->Mem.BufferMem.Ptr !=
                ur_mem_handle_t_::MemImpl::BufferMem::native_type{0},
            UR_RESULT_ERROR_INVALID_MEM_OBJECT);
  ur_mem_handle_t_::MemImpl::BufferMem::native_type Ptr =
      hBuffer->Mem.BufferMem.getWithOffset(pRegion->origin);

  void *HostPtr = nullptr;
  if (hBuffer->Mem.BufferMem.HostPtr) {
    HostPtr =
        static_cast<char *>(hBuffer->Mem.BufferMem.HostPtr) + pRegion->origin;
  }

  ReleaseGuard<ur_mem_handle_t> ReleaseGuard(hBuffer);

  std::unique_ptr<ur_mem_handle_t_> RetMemObj{nullptr};
  try {
    ScopedContext Active(Context->getDevice());

    RetMemObj = std::unique_ptr<ur_mem_handle_t_>{new ur_mem_handle_t_{
        Context, hBuffer, flags, AllocMode, Ptr, HostPtr, pRegion->size}};
  } catch (ur_result_t Err) {
    *phMem = nullptr;
    return Err;
  } catch (...) {
    *phMem = nullptr;
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  }

  ReleaseGuard.dismiss();
  *phMem = RetMemObj.release();
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemGetInfo(ur_mem_handle_t hMemory,
                                                 ur_mem_info_t MemInfoType,
                                                 size_t propSize,
                                                 void *pMemInfo,
                                                 size_t *pPropSizeRet) {

  UR_ASSERT(MemInfoType <= UR_MEM_INFO_CONTEXT,
            UR_RESULT_ERROR_INVALID_ENUMERATION);
  UR_ASSERT(hMemory->isBuffer(), UR_RESULT_ERROR_INVALID_MEM_OBJECT);

  UrReturnHelper ReturnValue(propSize, pMemInfo, pPropSizeRet);

  ScopedContext Active(hMemory->getContext()->getDevice());

  switch (MemInfoType) {
  case UR_MEM_INFO_SIZE: {
    try {
      size_t AllocSize = 0;
      UR_CHECK_ERROR(hipMemGetAddressRange(nullptr, &AllocSize,
                                           hMemory->Mem.BufferMem.Ptr));
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

/// Gets the native HIP handle of a UR mem object
///
/// \param[in] hMem The UR mem to get the native HIP object of.
/// \param[out] phNativeMem Set to the native handle of the UR mem object.
///
/// \return UR_RESULT_SUCCESS
UR_APIEXPORT ur_result_t UR_APICALL
urMemGetNativeHandle(ur_mem_handle_t hMem, ur_native_handle_t *phNativeMem) {
#if defined(__HIP_PLATFORM_NVIDIA__)
  if (sizeof(ur_mem_handle_t_::MemImpl::BufferMem::native_type) >
      sizeof(ur_native_handle_t)) {
    // Check that all the upper bits that cannot be represented by
    // ur_native_handle_t are empty.
    // NOTE: The following shift might trigger a warning, but the check in the
    // if above makes sure that this does not underflow.
    ur_mem_handle_t_::MemImpl::BufferMem::native_type UpperBits =
        hMem->Mem.BufferMem.get() >> (sizeof(ur_native_handle_t) * CHAR_BIT);
    if (UpperBits) {
      // Return an error if any of the remaining bits is non-zero.
      return UR_RESULT_ERROR_INVALID_MEM_OBJECT;
    }
  }
  *phNativeMem =
      reinterpret_cast<ur_native_handle_t>(hMem->Mem.BufferMem.get());
#elif defined(__HIP_PLATFORM_AMD__)
  *phNativeMem =
      reinterpret_cast<ur_native_handle_t>(hMem->Mem.BufferMem.get());
#else
#error("Must define exactly one of __HIP_PLATFORM_AMD__ or __HIP_PLATFORM_NVIDIA__");
#endif
  return UR_RESULT_SUCCESS;
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

  // Need input memory object
  UR_ASSERT((flags & UR_MEM_FLAGS_MASK) == 0,
            UR_RESULT_ERROR_INVALID_ENUMERATION);
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

  // We have to use hipArray3DCreate, which has some caveats. The height and
  // depth parameters must be set to 0 produce 1D or 2D arrays. image_desc gives
  // a minimum value of 1, so we need to convert the answer.
  HIP_ARRAY3D_DESCRIPTOR ArrayDesc;
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
    ArrayDesc.Format = HIP_AD_FORMAT_UNSIGNED_INT8;
    PixelTypeSizeBytes = 1;
    break;
  case UR_IMAGE_CHANNEL_TYPE_SIGNED_INT8:
    ArrayDesc.Format = HIP_AD_FORMAT_SIGNED_INT8;
    PixelTypeSizeBytes = 1;
    break;
  case UR_IMAGE_CHANNEL_TYPE_UNORM_INT16:
  case UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16:
    ArrayDesc.Format = HIP_AD_FORMAT_UNSIGNED_INT16;
    PixelTypeSizeBytes = 2;
    break;
  case UR_IMAGE_CHANNEL_TYPE_SIGNED_INT16:
    ArrayDesc.Format = HIP_AD_FORMAT_SIGNED_INT16;
    PixelTypeSizeBytes = 2;
    break;
  case UR_IMAGE_CHANNEL_TYPE_HALF_FLOAT:
    ArrayDesc.Format = HIP_AD_FORMAT_HALF;
    PixelTypeSizeBytes = 2;
    break;
  case UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32:
    ArrayDesc.Format = HIP_AD_FORMAT_UNSIGNED_INT32;
    PixelTypeSizeBytes = 4;
    break;
  case UR_IMAGE_CHANNEL_TYPE_SIGNED_INT32:
    ArrayDesc.Format = HIP_AD_FORMAT_SIGNED_INT32;
    PixelTypeSizeBytes = 4;
    break;
  case UR_IMAGE_CHANNEL_TYPE_FLOAT:
    ArrayDesc.Format = HIP_AD_FORMAT_FLOAT;
    PixelTypeSizeBytes = 4;
    break;
  default:
    // urMemImageCreate given unsupported image_channel_data_type
    return UR_RESULT_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR;
  }

  // When a dimension isn't used image_desc has the size set to 1
  size_t PixelSizeBytes =
      PixelTypeSizeBytes * 4; // 4 is the only number of channels we support
  size_t ImageSizeBytes = PixelSizeBytes * pImageDesc->width *
                          pImageDesc->height * pImageDesc->depth;

  ScopedContext Active(hContext->getDevice());
  hipArray *ImageArray;
  UR_CHECK_ERROR(hipArray3DCreate(reinterpret_cast<hipCUarray *>(&ImageArray),
                                  &ArrayDesc));

  try {
    if (PerformInitialCopy) {
      // We have to use a different copy function for each image dimensionality
      if (pImageDesc->type == UR_MEM_TYPE_IMAGE1D) {
        UR_CHECK_ERROR(hipMemcpyHtoA(ImageArray, 0, pHost, ImageSizeBytes));
      } else if (pImageDesc->type == UR_MEM_TYPE_IMAGE2D) {
        hip_Memcpy2D CpyDesc;
        memset(&CpyDesc, 0, sizeof(CpyDesc));
        CpyDesc.srcMemoryType = hipMemoryType::hipMemoryTypeHost;
        CpyDesc.srcHost = pHost;
        CpyDesc.dstMemoryType = hipMemoryType::hipMemoryTypeArray;
        CpyDesc.dstArray = reinterpret_cast<hipCUarray>(ImageArray);
        CpyDesc.WidthInBytes = PixelSizeBytes * pImageDesc->width;
        CpyDesc.Height = pImageDesc->height;
        UR_CHECK_ERROR(hipMemcpyParam2D(&CpyDesc));
      } else if (pImageDesc->type == UR_MEM_TYPE_IMAGE3D) {
        HIP_MEMCPY3D CpyDesc;
        memset(&CpyDesc, 0, sizeof(CpyDesc));
        CpyDesc.srcMemoryType = hipMemoryType::hipMemoryTypeHost;
        CpyDesc.srcHost = pHost;
        CpyDesc.dstMemoryType = hipMemoryType::hipMemoryTypeArray;
        CpyDesc.dstArray = reinterpret_cast<hipCUarray>(ImageArray);
        CpyDesc.WidthInBytes = PixelSizeBytes * pImageDesc->width;
        CpyDesc.Height = pImageDesc->height;
        CpyDesc.Depth = pImageDesc->depth;
        UR_CHECK_ERROR(hipDrvMemcpy3D(&CpyDesc));
      }
    }

    // HIP_RESOURCE_DESC is a union of different structs, shown here
    // We need to fill it as described here to use it for a surface or texture
    // HIP_RESOURCE_DESC::resType must be HIP_RESOURCE_TYPE_ARRAY and
    // HIP_RESOURCE_DESC::res::array::hArray must be set to a valid HIP array
    // handle.
    // HIP_RESOURCE_DESC::flags must be set to zero

    hipResourceDesc ImageResDesc;
    ImageResDesc.res.array.array = ImageArray;
    ImageResDesc.resType = hipResourceTypeArray;

    hipSurfaceObject_t Surface;
    UR_CHECK_ERROR(hipCreateSurfaceObject(&Surface, &ImageResDesc));

    auto URMemObj = std::unique_ptr<ur_mem_handle_t_>(new ur_mem_handle_t_{
        hContext, ImageArray, Surface, flags, pImageDesc->type, pHost});

    if (URMemObj == nullptr) {
      return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

    *phMem = URMemObj.release();
  } catch (ur_result_t Err) {
    UR_CHECK_ERROR(hipFreeArray(ImageArray));
    return Err;
  } catch (...) {
    UR_CHECK_ERROR(hipFreeArray(ImageArray));
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

UR_APIEXPORT ur_result_t UR_APICALL urMemRetain(ur_mem_handle_t hMem) {
  UR_ASSERT(hMem->getReferenceCount() > 0, UR_RESULT_ERROR_INVALID_MEM_OBJECT);
  hMem->incrementReferenceCount();
  return UR_RESULT_SUCCESS;
}
