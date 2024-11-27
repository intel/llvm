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
#include "enqueue.hpp"
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
  ur_mem_handle_t MemObj = nullptr;

  try {
    auto HostPtr = pProperties ? pProperties->pHost : nullptr;
    BufferMem::AllocMode AllocMode = BufferMem::AllocMode::Classic;

    if ((flags & UR_MEM_FLAG_USE_HOST_POINTER) && EnableUseHostPtr) {
      UR_CHECK_ERROR(
          cuMemHostRegister(HostPtr, size, CU_MEMHOSTREGISTER_DEVICEMAP));
      AllocMode = BufferMem::AllocMode::UseHostPtr;
    } else if (flags & UR_MEM_FLAG_ALLOC_HOST_POINTER) {
      UR_CHECK_ERROR(cuMemAllocHost(&HostPtr, size));
      AllocMode = BufferMem::AllocMode::AllocHostPtr;
    } else if (flags & UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER) {
      AllocMode = BufferMem::AllocMode::CopyIn;
    }

    auto URMemObj = std::unique_ptr<ur_mem_handle_t_>(
        new ur_mem_handle_t_{hContext, flags, AllocMode, HostPtr, size});

    // First allocation will be made at urMemBufferCreate if context only
    // has one device
    if (PerformInitialCopy && HostPtr) {
      // Perform initial copy to every device in context
      for (auto &Device : hContext->getDevices()) {
        ScopedContext Active(Device);
        // getPtr may allocate mem if not already allocated
        const auto &Ptr = std::get<BufferMem>(URMemObj->Mem).getPtr(Device);
        UR_CHECK_ERROR(cuMemcpyHtoD(Ptr, HostPtr, size));
      }
    }
    MemObj = URMemObj.release();
  } catch (ur_result_t Err) {
    return Err;
  } catch (std::bad_alloc &) {
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return UR_RESULT_ERROR_OUT_OF_RESOURCES;
  }

  *phBuffer = MemObj;

  return UR_RESULT_SUCCESS;
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

    // Call destructor
    std::unique_ptr<ur_mem_handle_t_> MemObjPtr(hMem);

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
urMemGetNativeHandle(ur_mem_handle_t hMem, ur_device_handle_t Device,
                     ur_native_handle_t *phNativeMem) {
  UR_ASSERT(Device != nullptr, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  try {
    *phNativeMem = std::get<BufferMem>(hMem->Mem).getPtr(Device);
  } catch (ur_result_t Err) {
    return Err;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
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

  // Any device in context will do
  auto Device = hMemory->getContext()->getDevices()[0];
  ScopedContext Active(Device);

  switch (MemInfoType) {
  case UR_MEM_INFO_SIZE: {
    try {
      size_t AllocSize = 0;
      UR_CHECK_ERROR(cuMemGetAddressRange(
          nullptr, &AllocSize,
          std::get<BufferMem>(hMemory->Mem).getPtr(Device)));
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
  case UR_MEM_INFO_REFERENCE_COUNT: {
    return ReturnValue(hMemory->getReferenceCount());
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
  UR_ASSERT(pImageDesc->type <= UR_MEM_TYPE_IMAGE1D_ARRAY,
            UR_RESULT_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR);

  // We only support RBGA channel order
  // TODO: check SYCL CTS and spec. May also have to support BGRA
  UR_ASSERT(pImageFormat->channelOrder == UR_IMAGE_CHANNEL_ORDER_RGBA,
            UR_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT);

  try {
    auto URMemObj = std::unique_ptr<ur_mem_handle_t_>(new ur_mem_handle_t_{
        hContext, flags, *pImageFormat, *pImageDesc, pHost});
    UR_ASSERT(std::get<SurfaceMem>(URMemObj->Mem).PixelTypeSizeBytes,
              UR_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT);

    if (PerformInitialCopy) {
      for (const auto &Device : hContext->getDevices()) {
        // Synchronous behaviour is best in this case
        ScopedContext Active(Device);
        CUstream Stream{0}; // Use default stream
        UR_CHECK_ERROR(enqueueMigrateMemoryToDeviceIfNeeded(URMemObj.get(),
                                                            Device, Stream));
        UR_CHECK_ERROR(cuStreamSynchronize(Stream));
      }
    }

    *phMem = URMemObj.release();
  } catch (ur_result_t Err) {
    return Err;
  } catch (std::bad_alloc &) {
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemImageGetInfo(ur_mem_handle_t hMemory,
                                                      ur_image_info_t propName,
                                                      size_t propSize,
                                                      void *pPropValue,
                                                      size_t *pPropSizeRet) {
  UR_ASSERT(hMemory->isImage(), UR_RESULT_ERROR_INVALID_MEM_OBJECT);

  auto Context = hMemory->getContext();

  // Any device will do
  auto Device = Context->getDevices()[0];
  ScopedContext Active(Device);
  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);

  try {
    CUDA_ARRAY3D_DESCRIPTOR ArrayInfo;

    UR_CHECK_ERROR(cuArray3DGetDescriptor(
        &ArrayInfo, std::get<SurfaceMem>(hMemory->Mem).getArray(Device)));

    const auto cuda2urFormat = [](CUarray_format CUFormat,
                                  ur_image_channel_type_t *ChannelType) {
      switch (CUFormat) {
      case CU_AD_FORMAT_UNSIGNED_INT8:
        *ChannelType = UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8;
        break;
      case CU_AD_FORMAT_UNSIGNED_INT16:
        *ChannelType = UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16;
        break;
      case CU_AD_FORMAT_UNSIGNED_INT32:
        *ChannelType = UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32;
        break;
      case CU_AD_FORMAT_SIGNED_INT8:
        *ChannelType = UR_IMAGE_CHANNEL_TYPE_SIGNED_INT8;
        break;
      case CU_AD_FORMAT_SIGNED_INT16:
        *ChannelType = UR_IMAGE_CHANNEL_TYPE_SIGNED_INT16;
        break;
      case CU_AD_FORMAT_SIGNED_INT32:
        *ChannelType = UR_IMAGE_CHANNEL_TYPE_SIGNED_INT32;
        break;
      case CU_AD_FORMAT_HALF:
        *ChannelType = UR_IMAGE_CHANNEL_TYPE_HALF_FLOAT;
        break;
      case CU_AD_FORMAT_FLOAT:
        *ChannelType = UR_IMAGE_CHANNEL_TYPE_FLOAT;
        break;
      default:
        return UR_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT;
      }
      return UR_RESULT_SUCCESS;
    };

    const auto cudaFormatToElementSize = [](CUarray_format CUFormat,
                                            size_t *Size) {
      switch (CUFormat) {
      case CU_AD_FORMAT_UNSIGNED_INT8:
      case CU_AD_FORMAT_SIGNED_INT8:
        *Size = 1;
        break;
      case CU_AD_FORMAT_UNSIGNED_INT16:
      case CU_AD_FORMAT_SIGNED_INT16:
      case CU_AD_FORMAT_HALF:
        *Size = 2;
        break;
      case CU_AD_FORMAT_UNSIGNED_INT32:
      case CU_AD_FORMAT_SIGNED_INT32:
      case CU_AD_FORMAT_FLOAT:
        *Size = 4;
        break;
      default:
        return UR_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT;
      }
      return UR_RESULT_SUCCESS;
    };

    switch (propName) {
    case UR_IMAGE_INFO_FORMAT: {
      ur_image_channel_type_t ChannelType{};
      UR_CHECK_ERROR(cuda2urFormat(ArrayInfo.Format, &ChannelType));
      return ReturnValue(
          ur_image_format_t{UR_IMAGE_CHANNEL_ORDER_RGBA, ChannelType});
    }
    case UR_IMAGE_INFO_WIDTH:
      return ReturnValue(ArrayInfo.Width);
    case UR_IMAGE_INFO_HEIGHT:
      return ReturnValue(ArrayInfo.Height);
    case UR_IMAGE_INFO_DEPTH:
      return ReturnValue(ArrayInfo.Depth);
    case UR_IMAGE_INFO_ELEMENT_SIZE: {
      size_t Size = 0;
      UR_CHECK_ERROR(cudaFormatToElementSize(ArrayInfo.Format, &Size));
      return ReturnValue(Size);
    }
    case UR_IMAGE_INFO_ROW_PITCH:
    case UR_IMAGE_INFO_SLICE_PITCH:
      return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;

    default:
      return UR_RESULT_ERROR_INVALID_ENUMERATION;
    }

  } catch (ur_result_t Err) {
    return Err;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
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
  UR_ASSERT(((pRegion->origin + pRegion->size) <= BufferImpl.getSize()),
            UR_RESULT_ERROR_INVALID_BUFFER_SIZE);

  std::unique_ptr<ur_mem_handle_t_> RetMemObj{nullptr};
  try {
    for (auto Device : hBuffer->Context->getDevices()) {
      BufferImpl.getPtr(
          Device); // This is allocating a dev ptr behind the scenes
                   // which is necessary before SubBuffer partition
    }
    RetMemObj = std::unique_ptr<ur_mem_handle_t_>{
        new ur_mem_handle_t_{hBuffer, pRegion->origin}};
  } catch (ur_result_t Err) {
    *phMem = nullptr;
    return Err;
  } catch (...) {
    *phMem = nullptr;
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  }

  *phMem = RetMemObj.release();
  return UR_RESULT_SUCCESS;
}

ur_result_t allocateMemObjOnDeviceIfNeeded(ur_mem_handle_t Mem,
                                           const ur_device_handle_t hDevice) {
  ScopedContext Active(hDevice);
  auto DeviceIdx = Mem->getContext()->getDeviceIndex(hDevice);
  ur_lock LockGuard(Mem->MemoryAllocationMutex);

  if (Mem->isBuffer()) {
    auto &Buffer = std::get<BufferMem>(Mem->Mem);
    auto &DevPtr = Buffer.Ptrs[DeviceIdx];

    // Allocation has already been made
    if (DevPtr != BufferMem::native_type{0}) {
      return UR_RESULT_SUCCESS;
    }

    if (Buffer.MemAllocMode == BufferMem::AllocMode::AllocHostPtr) {
      // Host allocation has already been made
      UR_CHECK_ERROR(cuMemHostGetDevicePointer(&DevPtr, Buffer.HostPtr, 0));
    } else if (Buffer.MemAllocMode == BufferMem::AllocMode::UseHostPtr) {
      UR_CHECK_ERROR(cuMemHostRegister(Buffer.HostPtr, Buffer.Size,
                                       CU_MEMHOSTALLOC_DEVICEMAP));
      UR_CHECK_ERROR(cuMemHostGetDevicePointer(&DevPtr, Buffer.HostPtr, 0));
    } else {
      UR_CHECK_ERROR(cuMemAlloc(&DevPtr, Buffer.Size));
    }
  } else {
    CUarray ImageArray{};
    CUsurfObject Surface;
    try {
      auto &Image = std::get<SurfaceMem>(Mem->Mem);
      // Allocation has already been made
      if (Image.Arrays[DeviceIdx]) {
        return UR_RESULT_SUCCESS;
      }
      UR_CHECK_ERROR(cuArray3DCreate(&ImageArray, &Image.ArrayDesc));
      Image.Arrays[DeviceIdx] = ImageArray;

      // CUDA_RESOURCE_DESC is a union of different structs, shown here
      // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TEXOBJECT.html
      // We need to fill it as described here to use it for a surface or texture
      // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__SURFOBJECT.html
      // CUDA_RESOURCE_DESC::resType must be CU_RESOURCE_TYPE_ARRAY and
      // CUDA_RESOURCE_DESC::res::array::hArray must be set to a valid CUDA
      // array handle. CUDA_RESOURCE_DESC::flags must be set to zero
      CUDA_RESOURCE_DESC ImageResDesc;
      ImageResDesc.res.array.hArray = ImageArray;
      ImageResDesc.resType = CU_RESOURCE_TYPE_ARRAY;
      ImageResDesc.flags = 0;

      UR_CHECK_ERROR(cuSurfObjectCreate(&Surface, &ImageResDesc));
      Image.SurfObjs[DeviceIdx] = Surface;
    } catch (ur_result_t Err) {
      if (ImageArray != CUarray{}) {
        UR_CHECK_ERROR(cuArrayDestroy(ImageArray));
      }
      return Err;
    } catch (...) {
      if (ImageArray != CUarray{}) {
        UR_CHECK_ERROR(cuArrayDestroy(ImageArray));
      }
      return UR_RESULT_ERROR_UNKNOWN;
    }
  }
  return UR_RESULT_SUCCESS;
}

namespace {
ur_result_t enqueueMigrateBufferToDevice(ur_mem_handle_t Mem,
                                         ur_device_handle_t hDevice,
                                         CUstream Stream) {
  auto &Buffer = std::get<BufferMem>(Mem->Mem);
  if (Mem->LastQueueWritingToMemObj == nullptr) {
    // Device allocation being initialized from host for the first time
    if (Buffer.HostPtr) {
      UR_CHECK_ERROR(cuMemcpyHtoDAsync(Buffer.getPtr(hDevice), Buffer.HostPtr,
                                       Buffer.Size, Stream));
    }
  } else if (Mem->LastQueueWritingToMemObj->getDevice() != hDevice) {
    UR_CHECK_ERROR(cuMemcpyDtoDAsync(
        Buffer.getPtr(hDevice),
        Buffer.getPtr(Mem->LastQueueWritingToMemObj->getDevice()), Buffer.Size,
        Stream));
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t enqueueMigrateImageToDevice(ur_mem_handle_t Mem,
                                        ur_device_handle_t hDevice,
                                        CUstream Stream) {
  auto &Image = std::get<SurfaceMem>(Mem->Mem);
  // When a dimension isn't used image_desc has the size set to 1
  size_t PixelSizeBytes = Image.PixelTypeSizeBytes *
                          4; // 4 is the only number of channels we support
  size_t ImageSizeBytes = PixelSizeBytes * Image.ImageDesc.width *
                          Image.ImageDesc.height * Image.ImageDesc.depth;

  CUarray ImageArray = Image.getArray(hDevice);

  CUDA_MEMCPY2D CpyDesc2D;
  CUDA_MEMCPY3D CpyDesc3D;
  // We have to use a different copy function for each image
  // dimensionality
  if (Image.ImageDesc.type == UR_MEM_TYPE_IMAGE2D) {
    memset(&CpyDesc2D, 0, sizeof(CpyDesc2D));
    CpyDesc2D.srcHost = Image.HostPtr;
    CpyDesc2D.dstMemoryType = CUmemorytype_enum::CU_MEMORYTYPE_ARRAY;
    CpyDesc2D.dstArray = ImageArray;
    CpyDesc2D.WidthInBytes = PixelSizeBytes * Image.ImageDesc.width;
    CpyDesc2D.Height = Image.ImageDesc.height;
  } else if (Image.ImageDesc.type == UR_MEM_TYPE_IMAGE3D) {
    memset(&CpyDesc3D, 0, sizeof(CpyDesc3D));
    CpyDesc3D.srcHost = Image.HostPtr;
    CpyDesc3D.dstMemoryType = CUmemorytype_enum::CU_MEMORYTYPE_ARRAY;
    CpyDesc3D.dstArray = ImageArray;
    CpyDesc3D.WidthInBytes = PixelSizeBytes * Image.ImageDesc.width;
    CpyDesc3D.Height = Image.ImageDesc.height;
    CpyDesc3D.Depth = Image.ImageDesc.depth;
  }

  if (Mem->LastQueueWritingToMemObj == nullptr) {
    if (Image.HostPtr) {
      if (Image.ImageDesc.type == UR_MEM_TYPE_IMAGE1D) {
        UR_CHECK_ERROR(cuMemcpyHtoAAsync(ImageArray, 0, Image.HostPtr,
                                         ImageSizeBytes, Stream));
      } else if (Image.ImageDesc.type == UR_MEM_TYPE_IMAGE2D) {
        CpyDesc2D.srcMemoryType = CUmemorytype_enum::CU_MEMORYTYPE_HOST;
        CpyDesc2D.srcHost = Image.HostPtr;
        UR_CHECK_ERROR(cuMemcpy2DAsync(&CpyDesc2D, Stream));
      } else if (Image.ImageDesc.type == UR_MEM_TYPE_IMAGE3D) {
        CpyDesc3D.srcMemoryType = CUmemorytype_enum::CU_MEMORYTYPE_HOST;
        CpyDesc3D.srcHost = Image.HostPtr;
        UR_CHECK_ERROR(cuMemcpy3DAsync(&CpyDesc3D, Stream));
      }
    }
  } else if (Mem->LastQueueWritingToMemObj->getDevice() != hDevice) {
    if (Image.ImageDesc.type == UR_MEM_TYPE_IMAGE1D) {
      // Blocking wait needed
      UR_CHECK_ERROR(urQueueFinish(Mem->LastQueueWritingToMemObj));
      // FIXME: 1D memcpy from DtoD going through the host.
      UR_CHECK_ERROR(cuMemcpyAtoH(
          Image.HostPtr,
          Image.getArray(Mem->LastQueueWritingToMemObj->getDevice()),
          0 /*srcOffset*/, ImageSizeBytes));
      UR_CHECK_ERROR(
          cuMemcpyHtoA(ImageArray, 0, Image.HostPtr, ImageSizeBytes));
    } else if (Image.ImageDesc.type == UR_MEM_TYPE_IMAGE2D) {
      CpyDesc2D.srcMemoryType = CUmemorytype_enum::CU_MEMORYTYPE_DEVICE;
      CpyDesc2D.srcArray =
          Image.getArray(Mem->LastQueueWritingToMemObj->getDevice());
      UR_CHECK_ERROR(cuMemcpy2DAsync(&CpyDesc2D, Stream));
    } else if (Image.ImageDesc.type == UR_MEM_TYPE_IMAGE3D) {
      CpyDesc3D.srcMemoryType = CUmemorytype_enum::CU_MEMORYTYPE_DEVICE;
      CpyDesc3D.srcArray =
          Image.getArray(Mem->LastQueueWritingToMemObj->getDevice());
      UR_CHECK_ERROR(cuMemcpy3DAsync(&CpyDesc3D, Stream));
    }
  }
  return UR_RESULT_SUCCESS;
}
} // namespace

// If calling this entry point it is necessary to lock the memoryMigrationMutex
// beforehand
ur_result_t enqueueMigrateMemoryToDeviceIfNeeded(
    ur_mem_handle_t Mem, const ur_device_handle_t hDevice, CUstream Stream) {
  UR_ASSERT(hDevice, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  // Device allocation has already been initialized with most up to date
  // data in buffer
  if (Mem->HaveMigratedToDeviceSinceLastWrite[Mem->getContext()->getDeviceIndex(
          hDevice)]) {
    return UR_RESULT_SUCCESS;
  }

  ScopedContext Active(hDevice);
  if (Mem->isBuffer()) {
    UR_CHECK_ERROR(enqueueMigrateBufferToDevice(Mem, hDevice, Stream));
  } else {
    UR_CHECK_ERROR(enqueueMigrateImageToDevice(Mem, hDevice, Stream));
  }

  Mem->HaveMigratedToDeviceSinceLastWrite[Mem->getContext()->getDeviceIndex(
      hDevice)] = true;
  return UR_RESULT_SUCCESS;
}

BufferMem::native_type
BufferMem::getPtrWithOffset(const ur_device_handle_t Device, size_t Offset) {
  if (ur_result_t Err = allocateMemObjOnDeviceIfNeeded(OuterMemStruct, Device);
      Err != UR_RESULT_SUCCESS) {
    throw Err;
  }
  return reinterpret_cast<native_type>(
      reinterpret_cast<uint8_t *>(
          Ptrs[OuterMemStruct->getContext()->getDeviceIndex(Device)]) +
      Offset);
}

CUarray SurfaceMem::getArray(const ur_device_handle_t Device) {
  if (ur_result_t Err = allocateMemObjOnDeviceIfNeeded(OuterMemStruct, Device);
      Err != UR_RESULT_SUCCESS) {
    throw Err;
  }
  return Arrays[OuterMemStruct->getContext()->getDeviceIndex(Device)];
}

CUsurfObject SurfaceMem::getSurface(const ur_device_handle_t Device) {
  if (ur_result_t Err = allocateMemObjOnDeviceIfNeeded(OuterMemStruct, Device);
      Err != UR_RESULT_SUCCESS) {
    throw Err;
  }
  return SurfObjs[OuterMemStruct->getContext()->getDeviceIndex(Device)];
}
