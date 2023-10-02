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

ur_mem_handle_t_::~ur_mem_handle_t_() {
  urContextRelease(Context);
  if (DeviceWithNativeAllocation) {
    urDeviceRelease(DeviceWithNativeAllocation);
  }
  if (LastEventWritingToMemObj != nullptr) {
    urEventRelease(LastEventWritingToMemObj);
  }
}

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

    hMem->clear();

  } catch (ur_result_t Err) {
    Result = Err;
  } catch (...) {
    Result = UR_RESULT_ERROR_OUT_OF_RESOURCES;
  }

  if (Result != UR_RESULT_SUCCESS) {
    // A reported HIP error is either an implementation or an asynchronous
    // HIP error for which it is unclear if the function that reported it
    // succeeded or not. Either way, the state of the program is compromised
    // and likely unrecoverable.
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
  ur_mem_handle_t MemObj = nullptr;

  try {
    auto HostPtr = pProperties ? pProperties->pHost : nullptr;
    ur_buffer_::AllocMode AllocMode = ur_buffer_::AllocMode::Classic;

    if ((flags & UR_MEM_FLAG_USE_HOST_POINTER) && EnableUseHostPtr) {
      AllocMode = ur_buffer_::AllocMode::UseHostPtr;
    } else if (flags & UR_MEM_FLAG_ALLOC_HOST_POINTER) {
      UR_CHECK_ERROR(hipMemAllocHost(&HostPtr, size));
      AllocMode = ur_buffer_::AllocMode::AllocHostPtr;
    }

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
          ScopedDevice Active(Device);
          auto &Ptr = ur_cast<ur_buffer_ *>(URMemObj.get())->getPtrs()[0];
          UR_CHECK_ERROR(hipMemcpyHtoD(Ptr, HostPtr, size));
          // Synchronize with default stream implicitly used by cuMemcpyHtoD
          // to make buffer data available on device before any other UR
          // call uses it.
          hipStream_t defaultStream = 0;
          UR_CHECK_ERROR(hipStreamSynchronize(defaultStream));
        }
      }
      MemObj = URMemObj.release();
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

/// Implements a buffer partition in the HIP backend.
/// A buffer partition (or a sub-buffer, in OpenCL terms) is simply
/// implemented as an offset over an existing HIP allocation.
UR_APIEXPORT ur_result_t UR_APICALL urMemBufferPartition(
    ur_mem_handle_t hBuffer, ur_mem_flags_t flags,
    ur_buffer_create_type_t bufferCreateType, const ur_buffer_region_t *pRegion,
    ur_mem_handle_t *phMem) {
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

  UR_ASSERT(pRegion->size != 0u, UR_RESULT_ERROR_INVALID_BUFFER_SIZE);

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

  ReleaseGuard<ur_mem_handle_t> ReleaseGuard(hBuffer);

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
      SubBufferPtrs[i] = static_cast<ur_buffer_::native_type>(
          static_cast<char *>(Buffer->getPtrs()[i]) + pRegion->origin);
    }
  } catch (ur_result_t Err) {
    *phMem = nullptr;
    return Err;
  } catch (...) {
    *phMem = nullptr;
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  }

  ReleaseGuard.dismiss();
  *phMem = MemObj.release();
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

  hMemory->allocateMemObjOnDeviceIfNeeded(
      hMemory->getContext()->getDevices()[0]);
  // TODO: This is just giving info for the native allocation of the first
  // device in the context. Is there a better way of doing this? Should we
  // remember the last device in the context that has been using this memObj
  // and return the native allocation on this device?
  ScopedDevice Active(hMemory->getContext()->getDevices()[0]);

  switch (MemInfoType) {
  case UR_MEM_INFO_SIZE: {
    try {
      if (hMemory->isBuffer()) {
        size_t AllocSize = 0;
        UR_CHECK_ERROR(hipMemGetAddressRange(
            nullptr, &AllocSize, ur_cast<ur_buffer_ *>(hMemory)->getPtrs()[0]));
        return ReturnValue(AllocSize);
      } else {
        // TODO add some support for image queries
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
      }
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
  std::ignore = hMem;
  std::ignore = phNativeMem;
  // FIXME: there is no good way of doing this with a multi device context.
  // If we return a single pointer, how would we know which device's allocation
  // it should be?
  // If we return a vector of pointers, this is OK for read only access but if
  // we write to a buffer, how would we know which one had been written to?
  // Should unused allocations be updated afterwards? We have no way of knowing
  // any of these things in the current API design.
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
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

  // We only support RBGA channel order
  // TODO: check SYCL CTS and spec. May also have to support BGRA
  UR_ASSERT(pImageFormat->channelOrder == UR_IMAGE_CHANNEL_ORDER_RGBA,
            UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION);

  // Device arrays and surfaces are allocated lazily for multi device context
  auto URMemObj = std::unique_ptr<ur_mem_handle_t_>(
      new ur_image_{hContext, flags, *pImageFormat, *pImageDesc, pHost});

  if (URMemObj == nullptr) {
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  }

  // Allocate eagerly if the NumDevices in a context is 1
  if (hContext->NumDevices == 1) {
    URMemObj->allocateMemObjOnDeviceIfNeeded(hContext->getDevices()[0]);
  }

  // Allocate and copy to all devices if PerformInitialCopy is set
  if (PerformInitialCopy) {
    for (auto Dev : hContext->getDevices()) {
      URMemObj->allocateMemObjOnDeviceIfNeeded(Dev);
    }
  }

  *phMem = URMemObj.release();
  return UR_RESULT_SUCCESS;
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

ur_result_t
ur_buffer_::allocateMemObjOnDeviceIfNeeded(ur_device_handle_t hDevice) {
  UR_ASSERT(hDevice, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  ScopedDevice Active(hDevice);
  ur_lock LockGuard(MemoryAllocationMutex);

  hipDeviceptr_t &DevPtr = getPtr(hDevice);

  // Allocation has already been made
  if (DevPtr != ur_buffer_::native_type{0}) {
    return UR_RESULT_SUCCESS;
  }

  if (MemAllocMode == ur_buffer_::AllocMode::AllocHostPtr) {
    // Host allocation has already been made
    UR_CHECK_ERROR(hipHostGetDevicePointer(&DevPtr, HostPtr, 0));
  } else if (MemAllocMode == ur_buffer_::AllocMode::UseHostPtr) {
    UR_CHECK_ERROR(hipHostRegister(HostPtr, Size, hipHostRegisterMapped));
    UR_CHECK_ERROR(hipHostGetDevicePointer(&DevPtr, HostPtr, 0));
  } else {
    UR_CHECK_ERROR(hipMalloc(&DevPtr, Size));
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t
ur_image_::allocateMemObjOnDeviceIfNeeded(ur_device_handle_t hDevice) {
  UR_ASSERT(hDevice, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  ScopedDevice Active(hDevice);
  ur_lock LockGuard(MemoryAllocationMutex);

  // Allocation has already been made
  if (getArray(hDevice) != nullptr) {
    return UR_RESULT_SUCCESS;
  }

  hipArray *ImageArray;
  try {
    UR_CHECK_ERROR(hipArray3DCreate(reinterpret_cast<hipCUarray *>(&ImageArray),
                                    &ArrayDesc));
    Arrays[hDevice->getIndex()] = ImageArray;

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
    SurfObjs[hDevice->getIndex()] = Surface;
  } catch (ur_result_t Err) {
    UR_CHECK_ERROR(hipFreeArray(ImageArray));
    return Err;
  } catch (...) {
    UR_CHECK_ERROR(hipFreeArray(ImageArray));
    return UR_RESULT_ERROR_UNKNOWN;
  }
  return UR_RESULT_SUCCESS;
}

// We should have already waited on the necessary events before calling this
// entry point
ur_result_t
ur_buffer_::migrateMemoryToDeviceIfNeeded(ur_device_handle_t hDevice) {
  UR_ASSERT(hDevice, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(Ptrs[hDevice->getIndex()], UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  // Device allocation has already been initialized with most up to date
  // data in buffer
  if (HaveMigratedToDeviceSinceLastWrite[hDevice->getIndex()]) {
    return UR_RESULT_SUCCESS;
  }

  ScopedDevice Active(hDevice);

  // If no kernels have written to the memobj then initialize the device
  // allocation from host if it has not been initialized already
  if (LastEventWritingToMemObj == nullptr) {
    // Device allocation being initialized from host for the first time
    if (HostPtr) {
      UR_CHECK_ERROR(
          hipMemcpyHtoD(getPtrs()[hDevice->getIndex()], HostPtr, Size));
      hipStream_t defaultStream = 0;
      UR_CHECK_ERROR(hipStreamSynchronize(defaultStream));
    }
  } else if (LastEventWritingToMemObj->getDevice() != hDevice) {
    UR_CHECK_ERROR(hipMemcpyDtoD(
        Ptrs[hDevice->getIndex()],
        Ptrs[LastEventWritingToMemObj->getDevice()->getIndex()], Size));
    // Synchronize on the destination device using the scoped context
    hipStream_t defaultStream = 0;
    UR_CHECK_ERROR(hipStreamSynchronize(defaultStream));
  }
  HaveMigratedToDeviceSinceLastWrite[hDevice->getIndex()] = true;

  return UR_RESULT_SUCCESS;
}

// We should have already waited on the necessary events before calling this
// entry point
ur_result_t
ur_image_::migrateMemoryToDeviceIfNeeded(ur_device_handle_t hDevice) {
  UR_ASSERT(hDevice, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(Arrays[hDevice->getIndex()], UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  // Device allocation has already been initialized with most up to date
  // data in buffer
  if (HaveMigratedToDeviceSinceLastWrite[hDevice->getIndex()]) {
    return UR_RESULT_SUCCESS;
  }

  // When a dimension isn't used image_desc has the size set to 1
  size_t PixelSizeBytes =
      PixelTypeSizeBytes * 4; // 4 is the only number of channels we support
  size_t ImageSizeBytes =
      PixelSizeBytes * ImageDesc.width * ImageDesc.height * ImageDesc.depth;

  ScopedDevice Active(hDevice);

  hipArray *ImageArray = getArray(hDevice);

  hip_Memcpy2D CpyDesc2D;
  HIP_MEMCPY3D CpyDesc3D;
  // We have to use a different copy function for each image
  // dimensionality
  if (ImageDesc.type == UR_MEM_TYPE_IMAGE2D) {
    memset(&CpyDesc2D, 0, sizeof(CpyDesc2D));
    CpyDesc2D.srcMemoryType = hipMemoryType::hipMemoryTypeHost;
    CpyDesc2D.dstMemoryType = hipMemoryType::hipMemoryTypeArray;
    CpyDesc2D.dstArray = reinterpret_cast<hipCUarray>(ImageArray);
    CpyDesc2D.WidthInBytes = PixelSizeBytes * ImageDesc.width;
    CpyDesc2D.Height = ImageDesc.height;
  } else if (ImageDesc.type == UR_MEM_TYPE_IMAGE3D) {
    memset(&CpyDesc3D, 0, sizeof(CpyDesc3D));
    CpyDesc3D.srcMemoryType = hipMemoryType::hipMemoryTypeHost;
    CpyDesc3D.dstMemoryType = hipMemoryType::hipMemoryTypeArray;
    CpyDesc3D.dstArray = reinterpret_cast<hipCUarray>(ImageArray);
    CpyDesc3D.WidthInBytes = PixelSizeBytes * ImageDesc.width;
    CpyDesc3D.Height = ImageDesc.height;
    CpyDesc3D.Depth = ImageDesc.depth;
  }

  if (LastEventWritingToMemObj == nullptr) {
    if (ImageDesc.type == UR_MEM_TYPE_IMAGE1D) {
      UR_CHECK_ERROR(hipMemcpyHtoA(ImageArray, 0, HostPtr, ImageSizeBytes));
    } else if (ImageDesc.type == UR_MEM_TYPE_IMAGE2D) {
      CpyDesc2D.srcHost = HostPtr;
      UR_CHECK_ERROR(hipMemcpyParam2D(&CpyDesc2D));
    } else if (ImageDesc.type == UR_MEM_TYPE_IMAGE3D) {
      CpyDesc3D.srcHost = HostPtr;
      UR_CHECK_ERROR(hipDrvMemcpy3D(&CpyDesc3D));
    }
  } else if (LastEventWritingToMemObj->getDevice() != hDevice) {
    if (ImageDesc.type == UR_MEM_TYPE_IMAGE1D) {
      // FIXME: 1D memcpy from DtoD going through the host.
      UR_CHECK_ERROR(hipMemcpyAtoH(
          HostPtr, getArray(LastEventWritingToMemObj->getDevice()),
          0 /*srcOffset*/, ImageSizeBytes));
      UR_CHECK_ERROR(hipMemcpyHtoA(ImageArray, 0, HostPtr, ImageSizeBytes));
    } else if (ImageDesc.type == UR_MEM_TYPE_IMAGE2D) {
      CpyDesc2D.srcArray = getArray(LastEventWritingToMemObj->getDevice());
      UR_CHECK_ERROR(hipMemcpyParam2D(&CpyDesc2D));
    } else if (ImageDesc.type == UR_MEM_TYPE_IMAGE3D) {
      CpyDesc3D.srcArray = getArray(LastEventWritingToMemObj->getDevice());
      UR_CHECK_ERROR(hipDrvMemcpy3D(&CpyDesc3D));
    }
  }

  HaveMigratedToDeviceSinceLastWrite[hDevice->getIndex()] = true;

  return UR_RESULT_SUCCESS;
}
