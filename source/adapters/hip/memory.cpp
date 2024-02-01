//===--------- memory.cpp - HIP Adapter -----------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "memory.hpp"
#include "context.hpp"
#include <cassert>
#include <ur_util.hpp>

size_t imageElementByteSize(hipArray_Format ArrayFormat) {
  switch (ArrayFormat) {
  case HIP_AD_FORMAT_UNSIGNED_INT8:
  case HIP_AD_FORMAT_SIGNED_INT8:
    return 1;
  case HIP_AD_FORMAT_UNSIGNED_INT16:
  case HIP_AD_FORMAT_SIGNED_INT16:
  case HIP_AD_FORMAT_HALF:
    return 2;
  case HIP_AD_FORMAT_UNSIGNED_INT32:
  case HIP_AD_FORMAT_SIGNED_INT32:
  case HIP_AD_FORMAT_FLOAT:
    return 4;
  default:
    detail::ur::die("Invalid HIP format specifier");
  }
  return 0;
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

    if (hMem->isSubBuffer()) {
      return UR_RESULT_SUCCESS;
    }

    UR_CHECK_ERROR(hMem->clear());

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
    auto HostPtr = pProperties ? pProperties->pHost : nullptr;
    BufferMem::AllocMode AllocMode = BufferMem::AllocMode::Classic;
    if ((flags & UR_MEM_FLAG_USE_HOST_POINTER) && EnableUseHostPtr) {
      AllocMode = BufferMem::AllocMode::UseHostPtr;
    } else if (flags & UR_MEM_FLAG_ALLOC_HOST_POINTER) {
      UR_CHECK_ERROR(hipHostMalloc(&HostPtr, size));
      AllocMode = BufferMem::AllocMode::AllocHostPtr;
    } else if (flags & UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER) {
      AllocMode = BufferMem::AllocMode::CopyIn;
    }

    auto URMemObj = std::unique_ptr<ur_mem_handle_t_>(
        new ur_mem_handle_t_{hContext, flags, AllocMode, HostPtr, size});
    if (URMemObj == nullptr) {
      throw UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

    // First allocation will be made at urMemBufferCreate if context only
    // has one device
    if (PerformInitialCopy && HostPtr) {
      // Perform initial copy to every device in context
      for (auto &Device : hContext->getDevices()) {
        ScopedContext Active(Device);
        // getPtr may allocate mem if not already allocated
        const auto &Ptr = std::get<BufferMem>(URMemObj->Mem).getPtr(Device);
        UR_CHECK_ERROR(hipMemcpyHtoD(Ptr, HostPtr, size));
        // TODO check if we can remove this
        // Synchronize with default stream implicitly used by cuMemcpyHtoD
        // to make buffer data available on device before any other UR
        // call uses it.
        // hipStream_t defaultStream = 0;
        // UR_CHECK_ERROR(hipStreamSynchronize(defaultStream));
      }
    }
    RetMemObj = URMemObj.release();
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

  auto &BufferImpl = std::get<BufferMem>(hBuffer->Mem);
  UR_ASSERT(((pRegion->origin + pRegion->size) <= BufferImpl.getSize()),
            UR_RESULT_ERROR_INVALID_BUFFER_SIZE);
  for (auto Device : hBuffer->Context->getDevices()) {
    BufferImpl.getPtr(Device); // This is allocating a dev ptr behind the scenes
                               // which is necessary before SubBuffer partition
  }

  ReleaseGuard<ur_mem_handle_t> ReleaseGuard(hBuffer);

  std::unique_ptr<ur_mem_handle_t_> RetMemObj{nullptr};
  try {
    RetMemObj = std::unique_ptr<ur_mem_handle_t_>{
        new ur_mem_handle_t_{hBuffer, pRegion->origin}};
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

  // FIXME: Only getting info for the first device in the context. This
  // should be fine in general
  auto Device = hMemory->getContext()->getDevices()[0];
  ScopedContext Active(Device);

  UrReturnHelper ReturnValue(propSize, pMemInfo, pPropSizeRet);

  switch (MemInfoType) {
  case UR_MEM_INFO_SIZE: {
    try {
      const auto MemVisitor = [Device](auto &&Mem) -> size_t {
        using T = std::decay_t<decltype(Mem)>;
        if constexpr (std::is_same_v<T, BufferMem>) {
          size_t AllocSize = 0;
          hipDeviceptr_t BasePtr = nullptr;
          UR_CHECK_ERROR(
              hipMemGetAddressRange(&BasePtr, &AllocSize, Mem.getPtr(Device)));
          return AllocSize;
        } else if constexpr (std::is_same_v<T, SurfaceMem>) {
#if HIP_VERSION < 50600000
          throw UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
#else
          HIP_ARRAY3D_DESCRIPTOR ArrayDescriptor;
          UR_CHECK_ERROR(
              hipArray3DGetDescriptor(&ArrayDescriptor, Mem.getArray(Device)));
          const auto PixelSizeBytes =
              imageElementByteSize(ArrayDescriptor.Format) *
              ArrayDescriptor.NumChannels;
          const auto ImageSizeBytes =
              PixelSizeBytes *
              (ArrayDescriptor.Width ? ArrayDescriptor.Width : 1) *
              (ArrayDescriptor.Height ? ArrayDescriptor.Height : 1) *
              (ArrayDescriptor.Depth ? ArrayDescriptor.Depth : 1);
          return ImageSizeBytes;
#endif
        } else {
          static_assert(ur_always_false_t<T>, "Not exhaustive visitor!");
        }
      };

      const auto AllocSize = std::visit(MemVisitor, hMemory->Mem);
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
urMemGetNativeHandle(ur_mem_handle_t hMem, ur_device_handle_t Device,
                     ur_native_handle_t *phNativeMem) {
#if defined(__HIP_PLATFORM_NVIDIA__)
  if (sizeof(BufferMem::native_type) > sizeof(ur_native_handle_t)) {
    // Check that all the upper bits that cannot be represented by
    // ur_native_handle_t are empty.
    // NOTE: The following shift might trigger a warning, but the check in the
    // if above makes sure that this does not underflow.
    BufferMem::native_type UpperBits =
        std::get<BufferMem>(hMem->Mem).getPtr(Device) >>
        (sizeof(ur_native_handle_t) * CHAR_BIT);
    if (UpperBits) {
      // Return an error if any of the remaining bits is non-zero.
      return UR_RESULT_ERROR_INVALID_MEM_OBJECT;
    }
  }
  *phNativeMem = reinterpret_cast<ur_native_handle_t>(
      std::get<BufferMem>(hMem->Mem).getPtr(Device));
#elif defined(__HIP_PLATFORM_AMD__)
  *phNativeMem = reinterpret_cast<ur_native_handle_t>(
      std::get<BufferMem>(hMem->Mem).getPtr(Device));
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

  auto URMemObj = std::unique_ptr<ur_mem_handle_t_>(
      new ur_mem_handle_t_{hContext, flags, *pImageFormat, *pImageDesc, pHost});

  if (URMemObj == nullptr) {
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  }

  if (PerformInitialCopy) {
    for (const auto &Dev : hContext->getDevices()) {
      UR_CHECK_ERROR(migrateMemoryToDeviceIfNeeded(URMemObj.get(), Dev));
    }
  }
  *phMem = URMemObj.release();
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemImageGetInfo(ur_mem_handle_t hMemory,
                                                      ur_image_info_t propName,
                                                      size_t propSize,
                                                      void *pPropValue,
                                                      size_t *pPropSizeRet) {
  UR_ASSERT(hMemory->isImage(), UR_RESULT_ERROR_INVALID_MEM_OBJECT);
  // FIXME: only getting infor for first image in ctx
  auto Device = hMemory->getContext()->getDevices()[0];
  ScopedContext Active(Device);
  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);

  try {
    HIP_ARRAY3D_DESCRIPTOR ArrayInfo;
#if HIP_VERSION >= 50600000
    UR_CHECK_ERROR(hipArray3DGetDescriptor(
        &ArrayInfo, std::get<SurfaceMem>(hMemory->Mem).getArray(Device)));
#else
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
#endif

    const auto hip2urFormat =
        [](hipArray_Format HipFormat) -> ur_image_channel_type_t {
      switch (HipFormat) {
      case HIP_AD_FORMAT_UNSIGNED_INT8:
        return UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8;
      case HIP_AD_FORMAT_UNSIGNED_INT16:
        return UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16;
      case HIP_AD_FORMAT_UNSIGNED_INT32:
        return UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32;
      case HIP_AD_FORMAT_SIGNED_INT8:
        return UR_IMAGE_CHANNEL_TYPE_SIGNED_INT8;
      case HIP_AD_FORMAT_SIGNED_INT16:
        return UR_IMAGE_CHANNEL_TYPE_SIGNED_INT16;
      case HIP_AD_FORMAT_SIGNED_INT32:
        return UR_IMAGE_CHANNEL_TYPE_SIGNED_INT32;
      case HIP_AD_FORMAT_HALF:
        return UR_IMAGE_CHANNEL_TYPE_HALF_FLOAT;
      case HIP_AD_FORMAT_FLOAT:
        return UR_IMAGE_CHANNEL_TYPE_FLOAT;

      default:
        detail::ur::die("Invalid Hip format specified.");
      }
    };

    switch (propName) {
    case UR_IMAGE_INFO_FORMAT:
      return ReturnValue(ur_image_format_t{UR_IMAGE_CHANNEL_ORDER_RGBA,
                                           hip2urFormat(ArrayInfo.Format)});
    case UR_IMAGE_INFO_WIDTH:
      return ReturnValue(ArrayInfo.Width);
    case UR_IMAGE_INFO_HEIGHT:
      return ReturnValue(ArrayInfo.Height);
    case UR_IMAGE_INFO_DEPTH:
      return ReturnValue(ArrayInfo.Depth);
    case UR_IMAGE_INFO_ELEMENT_SIZE:
      return ReturnValue(imageElementByteSize(ArrayInfo.Format));
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
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemRetain(ur_mem_handle_t hMem) {
  UR_ASSERT(hMem->getReferenceCount() > 0, UR_RESULT_ERROR_INVALID_MEM_OBJECT);
  hMem->incrementReferenceCount();
  return UR_RESULT_SUCCESS;
}

ur_result_t allocateMemObjOnDeviceIfNeeded(ur_mem_handle_t Mem,
                                           const ur_device_handle_t hDevice) {
  ScopedContext Active(hDevice);
  ur_lock LockGuard(Mem->MemoryAllocationMutex);

  if (Mem->isBuffer()) {
    auto &Buffer = std::get<BufferMem>(Mem->Mem);
    hipDeviceptr_t &DevPtr = Buffer.Ptrs[hDevice->getIndex()];

    // Allocation has already been made
    if (DevPtr != BufferMem::native_type{0}) {
      return UR_RESULT_SUCCESS;
    }

    if (Buffer.MemAllocMode == BufferMem::AllocMode::AllocHostPtr) {
      // Host allocation has already been made
      UR_CHECK_ERROR(hipHostGetDevicePointer(&DevPtr, Buffer.HostPtr, 0));
    } else if (Buffer.MemAllocMode == BufferMem::AllocMode::UseHostPtr) {
      UR_CHECK_ERROR(
          hipHostRegister(Buffer.HostPtr, Buffer.Size, hipHostRegisterMapped));
      UR_CHECK_ERROR(hipHostGetDevicePointer(&DevPtr, Buffer.HostPtr, 0));
    } else {
      UR_CHECK_ERROR(hipMalloc(&DevPtr, Buffer.Size));
    }
  } else {
    hipArray *ImageArray;
    hipSurfaceObject_t Surface;
    try {
      auto &Image = std::get<SurfaceMem>(Mem->Mem);
      // Allocation has already been made
      if (Image.Arrays[hDevice->getIndex()]) {
        return UR_RESULT_SUCCESS;
      }
      UR_CHECK_ERROR(hipArray3DCreate(
          reinterpret_cast<hipCUarray *>(&ImageArray), &Image.ArrayDesc));
      Image.Arrays[hDevice->getIndex()] = ImageArray;
      // HIP_RESOURCE_DESC is a union of different structs, shown here
      // We need to fill it as described here to use it for a surface or texture
      // HIP_RESOURCE_DESC::resType must be HIP_RESOURCE_TYPE_ARRAY and
      // HIP_RESOURCE_DESC::res::array::hArray must be set to a valid HIP array
      // handle.
      // HIP_RESOURCE_DESC::flags must be set to zero
      hipResourceDesc ImageResDesc;
      ImageResDesc.res.array.array = ImageArray;
      ImageResDesc.resType = hipResourceTypeArray;

      UR_CHECK_ERROR(hipCreateSurfaceObject(&Surface, &ImageResDesc));
      Image.SurfObjs[hDevice->getIndex()] = Surface;
    } catch (ur_result_t Err) {
      if (ImageArray) {
        UR_CHECK_ERROR(hipFreeArray(ImageArray));
      }
      return Err;
    } catch (...) {
      if (ImageArray) {
        UR_CHECK_ERROR(hipFreeArray(ImageArray));
      }
      return UR_RESULT_ERROR_UNKNOWN;
    }
  }
  return UR_RESULT_SUCCESS;
}

namespace {
inline ur_result_t migrateBufferToDevice(ur_mem_handle_t Mem,
                                         ur_device_handle_t hDevice) {
  auto &Buffer = std::get<BufferMem>(Mem->Mem);
  if (Mem->LastEventWritingToMemObj == nullptr) {
    // Device allocation being initialized from host for the first time
    if (Buffer.HostPtr) {
      UR_CHECK_ERROR(
          hipMemcpyHtoD(Buffer.getPtr(hDevice), Buffer.HostPtr, Buffer.Size));
    }
  } else if (Mem->LastEventWritingToMemObj->getDevice() != hDevice) {
    UR_CHECK_ERROR(
        hipMemcpyDtoD(Buffer.getPtr(hDevice),
                      Buffer.getPtr(Mem->LastEventWritingToMemObj->getDevice()),
                      Buffer.Size));
  }
  return UR_RESULT_SUCCESS;
}

inline ur_result_t migrateImageToDevice(ur_mem_handle_t Mem,
                                        ur_device_handle_t hDevice) {
  auto &Image = std::get<SurfaceMem>(Mem->Mem);
  // When a dimension isn't used image_desc has the size set to 1
  size_t PixelSizeBytes = Image.PixelTypeSizeBytes *
                          4; // 4 is the only number of channels we support
  size_t ImageSizeBytes = PixelSizeBytes * Image.ImageDesc.width *
                          Image.ImageDesc.height * Image.ImageDesc.depth;

  hipArray *ImageArray = Image.getArray(hDevice);

  hip_Memcpy2D CpyDesc2D;
  HIP_MEMCPY3D CpyDesc3D;
  // We have to use a different copy function for each image
  // dimensionality
  if (Image.ImageDesc.type == UR_MEM_TYPE_IMAGE2D) {
    memset(&CpyDesc2D, 0, sizeof(CpyDesc2D));
    CpyDesc2D.srcMemoryType = hipMemoryType::hipMemoryTypeHost;
    CpyDesc2D.dstMemoryType = hipMemoryType::hipMemoryTypeArray;
    CpyDesc2D.dstArray = reinterpret_cast<hipCUarray>(ImageArray);
    CpyDesc2D.WidthInBytes = PixelSizeBytes * Image.ImageDesc.width;
    CpyDesc2D.Height = Image.ImageDesc.height;
  } else if (Image.ImageDesc.type == UR_MEM_TYPE_IMAGE3D) {
    memset(&CpyDesc3D, 0, sizeof(CpyDesc3D));
    CpyDesc3D.srcMemoryType = hipMemoryType::hipMemoryTypeHost;
    CpyDesc3D.dstMemoryType = hipMemoryType::hipMemoryTypeArray;
    CpyDesc3D.dstArray = reinterpret_cast<hipCUarray>(ImageArray);
    CpyDesc3D.WidthInBytes = PixelSizeBytes * Image.ImageDesc.width;
    CpyDesc3D.Height = Image.ImageDesc.height;
    CpyDesc3D.Depth = Image.ImageDesc.depth;
  }

  if (Mem->LastEventWritingToMemObj == nullptr) {
    if (Image.ImageDesc.type == UR_MEM_TYPE_IMAGE1D) {
      UR_CHECK_ERROR(
          hipMemcpyHtoA(ImageArray, 0, Image.HostPtr, ImageSizeBytes));
    } else if (Image.ImageDesc.type == UR_MEM_TYPE_IMAGE2D) {
      CpyDesc2D.srcHost = Image.HostPtr;
      UR_CHECK_ERROR(hipMemcpyParam2D(&CpyDesc2D));
    } else if (Image.ImageDesc.type == UR_MEM_TYPE_IMAGE3D) {
      CpyDesc3D.srcHost = Image.HostPtr;
      UR_CHECK_ERROR(hipDrvMemcpy3D(&CpyDesc3D));
    }
  } else if (Mem->LastEventWritingToMemObj->getDevice() != hDevice) {
    if (Image.ImageDesc.type == UR_MEM_TYPE_IMAGE1D) {
      // FIXME: 1D memcpy from DtoD going through the host.
      UR_CHECK_ERROR(hipMemcpyAtoH(
          Image.HostPtr,
          Image.getArray(Mem->LastEventWritingToMemObj->getDevice()),
          0 /*srcOffset*/, ImageSizeBytes));
      UR_CHECK_ERROR(
          hipMemcpyHtoA(ImageArray, 0, Image.HostPtr, ImageSizeBytes));
    } else if (Image.ImageDesc.type == UR_MEM_TYPE_IMAGE2D) {
      CpyDesc2D.srcArray =
          Image.getArray(Mem->LastEventWritingToMemObj->getDevice());
      UR_CHECK_ERROR(hipMemcpyParam2D(&CpyDesc2D));
    } else if (Image.ImageDesc.type == UR_MEM_TYPE_IMAGE3D) {
      CpyDesc3D.srcArray =
          Image.getArray(Mem->LastEventWritingToMemObj->getDevice());
      UR_CHECK_ERROR(hipDrvMemcpy3D(&CpyDesc3D));
    }
  }
  return UR_RESULT_SUCCESS;
}
} // namespace

// If calling this entry point it is necessary to lock the memoryMigrationMutex
// beforehand
ur_result_t migrateMemoryToDeviceIfNeeded(ur_mem_handle_t Mem,
                                          const ur_device_handle_t hDevice) {
  UR_ASSERT(hDevice, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  // Device allocation has already been initialized with most up to date
  // data in buffer
  if (Mem->HaveMigratedToDeviceSinceLastWrite[hDevice->getIndex()]) {
    return UR_RESULT_SUCCESS;
  }

  ScopedContext Active(hDevice);
  if (Mem->isBuffer()) {
    UR_CHECK_ERROR(migrateBufferToDevice(Mem, hDevice));
  } else {
    UR_CHECK_ERROR(migrateImageToDevice(Mem, hDevice));
  }

  Mem->HaveMigratedToDeviceSinceLastWrite[hDevice->getIndex()] = true;
  return UR_RESULT_SUCCESS;
}
