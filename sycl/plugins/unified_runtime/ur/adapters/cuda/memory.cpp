//===--------- memory.cpp - CUDA Adapter ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include <cuda.h>

#include "common.hpp"
#include "context.hpp"
#include "event.hpp"
#include "memory.hpp"

ur_mem_handle_t_::~ur_mem_handle_t_() {
  urContextRelease(Context);
  if (DeviceWithNativeAllocation) {
    urDeviceRelease(DeviceWithNativeAllocation);
  }
  if (LastEventWritingToMemObj != nullptr) {
    urEventRelease(LastEventWritingToMemObj);
  }
}

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
    auto HostPtr = pProperties ? pProperties->pHost : nullptr;
    ur_buffer_::AllocMode AllocMode = ur_buffer_::AllocMode::Classic;

    if ((flags & UR_MEM_FLAG_USE_HOST_POINTER) && EnableUseHostPtr) {
      AllocMode = ur_buffer_::AllocMode::UseHostPtr;
    } else if (flags & UR_MEM_FLAG_ALLOC_HOST_POINTER) {
      Result = UR_CHECK_ERROR(cuMemAllocHost(&HostPtr, size));
      AllocMode = ur_buffer_::AllocMode::AllocHostPtr;
    }

    if (Result == UR_RESULT_SUCCESS) {
      ur_buffer_ *parentBuffer = nullptr;

      auto URMemObj = std::unique_ptr<ur_mem_handle_t_>(new ur_buffer_{
          hContext, parentBuffer, flags, AllocMode, HostPtr, size});
      if (URMemObj != nullptr) {

        // First allocation will be made at urMemBufferCreate if context only
        // has one device
        if (PerformInitialCopy && hContext->NumDevices == 1) {
          // Operates on the default stream of the current CUDA context.
          auto Device = hContext->getDevices()[0];
          Result = URMemObj->allocateMemObjOnDeviceIfNeeded(Device);

          if (PerformInitialCopy && Result == UR_RESULT_SUCCESS && HostPtr) {
            ScopedContext Active(Device);
            auto &Ptr = ur_cast<ur_buffer_ *>(URMemObj.get())->getPtrs()[0];
            Result = UR_CHECK_ERROR(cuMemcpyHtoD(Ptr, HostPtr, size));
            // Synchronize with default stream implicitly used by cuMemcpyHtoD
            // to make buffer data available on device before any other UR
            // call uses it.
            if (Result == UR_RESULT_SUCCESS) {
              CUstream defaultStream = 0;
              Result = UR_CHECK_ERROR(cuStreamSynchronize(defaultStream));
            }
          }
        }
        MemObj = URMemObj.release();
      } else {
        Result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
      }
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

    // Free buffer
    if (hMem->isBuffer()) {
      // make sure hMem is released in case checkErrorUR throws
      std::unique_ptr<ur_buffer_> Buffer(ur_cast<ur_buffer_ *>(hMem));
      if (Buffer->isSubBuffer()) {
        return UR_RESULT_SUCCESS;
      }

      switch (Buffer->MemAllocMode) {
      case ur_buffer_::AllocMode::CopyIn:
      case ur_buffer_::AllocMode::Classic:
        for (auto i = 0u; i < hMem->getContext()->NumDevices; ++i) {
          if (Buffer->getPtrs()[i] != ur_buffer_::native_type{0}) {
            ScopedContext Active(Buffer->getContext()->getDevices()[i]);
            Result = UR_CHECK_ERROR(cuMemFree(Buffer->Ptrs[i]));
          }
        }
        break;
      case ur_buffer_::AllocMode::UseHostPtr:
        Result = UR_CHECK_ERROR(cuMemHostUnregister(Buffer->HostPtr));
        break;
      case ur_buffer_::AllocMode::AllocHostPtr:
        Result = UR_CHECK_ERROR(cuMemFreeHost(Buffer->HostPtr));
      };
    } else {
      UR_ASSERT(hMem->isImage(), UR_RESULT_ERROR_INVALID_VALUE);
      // Images are allocated on the first device in a context
      ScopedContext Active(hMem->getContext()->getDevices()[0]);
      std::unique_ptr<ur_image_> Image(ur_cast<ur_image_ *>(hMem));
      if (Image->Mem.SurfaceMem.getSurface() != CUsurfObject{0}) {
        Result = UR_CHECK_ERROR(
            cuSurfObjectDestroy(Image->Mem.SurfaceMem.getSurface()));
      }
      if (Image->Mem.SurfaceMem.getArray() != CUarray{0}) {
        Result =
            UR_CHECK_ERROR(cuArrayDestroy(Image->Mem.SurfaceMem.getArray()));
      }
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
UR_APIEXPORT ur_result_t UR_APICALL
urMemGetNativeHandle(ur_mem_handle_t hMem, ur_native_handle_t *phNativeMem) {
  UR_ASSERT(hMem, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(phNativeMem, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  // TODO(hdelan): I need to have some way to remember the last active device
  // in the context so I can give a handle to that. At the moment I am just
  // returning a handle to the allocation of device at index 0
  if (hMem->isBuffer()) {
    *phNativeMem = reinterpret_cast<ur_native_handle_t>(
        ur_cast<ur_buffer_ *>(hMem)->getPtrs()[0]);
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemGetInfo(ur_mem_handle_t hMemory,
                                                 ur_mem_info_t MemInfoType,
                                                 size_t propSize,
                                                 void *pMemInfo,
                                                 size_t *pPropSizeRet) {
  UR_ASSERT(hMemory->isBuffer(), UR_RESULT_ERROR_INVALID_MEM_OBJECT);

  UrReturnHelper ReturnValue(propSize, pMemInfo, pPropSizeRet);

  // TODO(hdelan): I need to have some way to remember the last active device
  // in the context so I can give a handle to that. At the moment I am just
  // returning a handle to the allocation of device at index 0
  ScopedContext Active(hMemory->getContext()->getDevices()[0]);

  switch (MemInfoType) {
  case UR_MEM_INFO_SIZE: {
    try {
      size_t AllocSize = 0;
      // TODO here as well
      UR_CHECK_ERROR(cuMemGetAddressRange(
          nullptr, &AllocSize, ur_cast<ur_buffer_ *>(hMemory)->getPtrs()[0]));
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

  // TODO make image support on multi device context. We are just using the
  // first device in context here
  ScopedContext Active(hContext->getDevices()[0]);
  CUarray ImageArray = nullptr;

  try {
    Result = UR_CHECK_ERROR(cuArray3DCreate(&ImageArray, &ArrayDesc));
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
        Result =
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
        Result = UR_CHECK_ERROR(cuMemcpy2D(&CpyDesc));
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
        Result = UR_CHECK_ERROR(cuMemcpy3D(&CpyDesc));
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
    Result = UR_CHECK_ERROR(cuSurfObjectCreate(&Surface, &ImageResDesc));

    auto MemObj = std::unique_ptr<ur_mem_handle_t_>(new ur_image_(
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

  ur_buffer_ *Buffer = ur_cast<ur_buffer_ *>(hBuffer);
  UR_ASSERT(!Buffer->isSubBuffer(), UR_RESULT_ERROR_INVALID_MEM_OBJECT);

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

  assert((pRegion->origin <= (pRegion->origin + pRegion->size)) && "Overflow");
  UR_ASSERT(((pRegion->origin + pRegion->size) <= Buffer->getSize()),
            UR_RESULT_ERROR_INVALID_BUFFER_SIZE);
  // Retained indirectly due to retaining parent buffer below.
  ur_context_handle_t Context = hBuffer->Context;

  ur_buffer_::AllocMode AllocMode = ur_buffer_::AllocMode::Classic;

  std::vector<ur_buffer_::native_type> NewPtrs(Buffer->getPtrs().size());

  void *HostPtr = nullptr;
  if (Buffer->HostPtr) {
    HostPtr = static_cast<char *>(Buffer->HostPtr) + pRegion->origin;
  }

  std::unique_ptr<ur_mem_handle_t_> MemObj{nullptr};
  try {
    MemObj = std::unique_ptr<ur_mem_handle_t_>{new ur_buffer_{
        Context, Buffer, flags, AllocMode, HostPtr, pRegion->size}};
    auto &SubBufferPtrs = ur_cast<ur_buffer_ *>(MemObj.get())->getPtrs();
    for (auto i = 0u; i < Buffer->getPtrs().size(); ++i) {
      // If we want to partition our buffer into a subbuffer, we must allocate
      // on all devices in context now
      Buffer->allocateMemObjOnDeviceIfNeeded(
          Buffer->getContext()->getDevices()[i]);
      UR_ASSERT(Buffer->getPtrs()[i] != ur_buffer_::native_type{0},
                UR_RESULT_ERROR_INVALID_MEM_OBJECT);
      SubBufferPtrs[i] = Buffer->getPtrs()[i] + pRegion->origin;
    }
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

ur_result_t
ur_buffer_::allocateMemObjOnDeviceIfNeeded(ur_device_handle_t hDevice) {
  UR_ASSERT(hDevice, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  ScopedContext Active(hDevice);
  ur_lock_guard LockGuard(MemoryAllocationMutex);

  ur_result_t Result = UR_RESULT_SUCCESS;
  CUdeviceptr &DevPtr = getNativePtr(hDevice);

  // Allocation has already been made
  if (DevPtr != ur_buffer_::native_type{0}) {
    return UR_RESULT_SUCCESS;
  }

  if (MemAllocMode == ur_buffer_::AllocMode::AllocHostPtr) {
    // Host allocation has already been made
    Result = UR_CHECK_ERROR(cuMemHostGetDevicePointer(&DevPtr, HostPtr, 0));
  } else if (MemAllocMode == ur_buffer_::AllocMode::UseHostPtr) {
    Result = UR_CHECK_ERROR(
        cuMemHostRegister(HostPtr, Size, CU_MEMHOSTREGISTER_DEVICEMAP));
    Result = UR_CHECK_ERROR(cuMemHostGetDevicePointer(&DevPtr, HostPtr, 0));
  } else {
    Result = UR_CHECK_ERROR(cuMemAlloc(&DevPtr, Size));
  }
  // TODO(hdelan): add some bailouts here that will free all mem if these fail
  return Result;
}

ur_result_t
ur_image_::allocateMemObjOnDeviceIfNeeded(ur_device_handle_t hDevice) {
  UR_ASSERT(hDevice, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  // TODO(hdelan): Add some logic here
  return UR_RESULT_SUCCESS;
}

// We should have already waited on the necessary events before calling this
// function
ur_result_t
ur_buffer_::migrateMemoryToDeviceIfNeeded(ur_device_handle_t hDevice) {
  UR_ASSERT(hDevice, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(Ptrs[hDevice->getIndex()], UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  ScopedContext Active(hDevice);

  ur_result_t Result = UR_RESULT_SUCCESS;

  // If no kernels have written to the memobj then initialize the device
  // allocation from host if it has not been initialized already
  if (LastEventWritingToMemObj == nullptr) {
    // Device allocation has already been initialized and no previously
    // submitted kernel has written to MemObj
    if (HaveMigratedToDevice[hDevice->getIndex()]) {
      return Result;
    }
    // Device allocation being initialized from host for the first time
    // TODO(hdelan): HostPtr is sometimes nullptr here. Why is that?
    if (HostPtr) {
      Result = UR_CHECK_ERROR(
          cuMemcpyHtoD(getPtrs()[hDevice->getIndex()], HostPtr, Size));
      if (Result == UR_RESULT_SUCCESS) {
        CUstream defaultStream = 0;
        Result = UR_CHECK_ERROR(cuStreamSynchronize(defaultStream));
      }
    }
  } else if (LastEventWritingToMemObj->getDevice() != hDevice) {
    UR_ASSERT(LastEventWritingToMemObj, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
    Result = UR_CHECK_ERROR(cuMemcpyDtoD(
        Ptrs[hDevice->getIndex()],
        Ptrs[LastEventWritingToMemObj->getDevice()->getIndex()], Size));
    if (Result == UR_RESULT_SUCCESS) {
      // Synchronize on the destination device using the scoped context
      CUstream defaultStream = 0;
      Result = UR_CHECK_ERROR(cuStreamSynchronize(defaultStream));
    }
  }
  HaveMigratedToDevice[hDevice->getIndex()] = true;

  return Result;
}

ur_result_t
ur_image_::migrateMemoryToDeviceIfNeeded(ur_device_handle_t hDevice) {
  UR_ASSERT(hDevice, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  // TODO(hdelan): Add some logic here
  return UR_RESULT_SUCCESS;
}
