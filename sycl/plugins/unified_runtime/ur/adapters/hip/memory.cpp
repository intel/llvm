#include "memory.hpp"
#include "context.hpp"
#include <cassert>

/// Decreases the reference count of the Mem object.
/// If this is zero, calls the relevant HIP Free function
/// \return UR_RESULT_SUCCESS unless deallocation error
///
UR_APIEXPORT ur_result_t UR_APICALL urMemRelease(ur_mem_handle_t hMem) {
  UR_ASSERT(hMem, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  ur_result_t ret = UR_RESULT_SUCCESS;

  try {

    // Do nothing if there are other references
    if (hMem->decrement_reference_count() > 0) {
      return UR_RESULT_SUCCESS;
    }

    // make sure memObj is released in case UR_CHECK_ERROR throws
    std::unique_ptr<ur_mem_handle_t_> uniqueMemObj(hMem);

    if (hMem->is_sub_buffer()) {
      return UR_RESULT_SUCCESS;
    }

    ScopedContext active(uniqueMemObj->get_context());

    if (hMem->mem_type_ == ur_mem_handle_t_::mem_type::buffer) {
      switch (uniqueMemObj->mem_.buffer_mem_.allocMode_) {
      case ur_mem_handle_t_::mem_::buffer_mem_::alloc_mode::copy_in:
      case ur_mem_handle_t_::mem_::buffer_mem_::alloc_mode::classic:
        ret = UR_CHECK_ERROR(
            hipFree((void *)uniqueMemObj->mem_.buffer_mem_.ptr_));
        break;
      case ur_mem_handle_t_::mem_::buffer_mem_::alloc_mode::use_host_ptr:
        ret = UR_CHECK_ERROR(
            hipHostUnregister(uniqueMemObj->mem_.buffer_mem_.hostPtr_));
        break;
      case ur_mem_handle_t_::mem_::buffer_mem_::alloc_mode::alloc_host_ptr:
        ret = UR_CHECK_ERROR(
            hipFreeHost(uniqueMemObj->mem_.buffer_mem_.hostPtr_));
      };
    }

    else if (hMem->mem_type_ == ur_mem_handle_t_::mem_type::surface) {
      ret = UR_CHECK_ERROR(hipDestroySurfaceObject(
          uniqueMemObj->mem_.surface_mem_.get_surface()));
      auto array = uniqueMemObj->mem_.surface_mem_.get_array();
      ret = UR_CHECK_ERROR(hipFreeArray(array));
    }

  } catch (ur_result_t err) {
    ret = err;
  } catch (...) {
    ret = UR_RESULT_ERROR_OUT_OF_RESOURCES;
  }

  if (ret != UR_RESULT_SUCCESS) {
    // A reported HIP error is either an implementation or an asynchronous HIP
    // error for which it is unclear if the function that reported it succeeded
    // or not. Either way, the state of the program is compromised and likely
    // unrecoverable.
    sycl::detail::ur::die(
        "Unrecoverable program state reached in urMemRelease");
  }

  return UR_RESULT_SUCCESS;
}

/// Creates a UR Memory object using a HIP memory allocation.
/// Can trigger a manual copy depending on the mode.
/// \TODO Implement USE_HOST_PTR using hipHostRegister
///
UR_APIEXPORT ur_result_t UR_APICALL urMemBufferCreate(
    ur_context_handle_t hContext, ur_mem_flags_t flags, size_t size,
    const ur_buffer_properties_t *pProperties, ur_mem_handle_t *phBuffer) {
  UR_ASSERT(hContext, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  // Validate flags
  UR_ASSERT((flags & UR_MEM_FLAGS_MASK) == 0,
            UR_RESULT_ERROR_INVALID_ENUMERATION);
  if (flags &
      (UR_MEM_FLAG_USE_HOST_POINTER | UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER)) {
    UR_ASSERT(pProperties && pProperties->pHost,
              UR_RESULT_ERROR_INVALID_HOST_PTR);
  }
  // Need input memory object
  UR_ASSERT(phBuffer, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  UR_ASSERT(size != 0, UR_RESULT_ERROR_INVALID_BUFFER_SIZE);

  // Currently, USE_HOST_PTR is not implemented using host register
  // since this triggers a weird segfault after program ends.
  // Setting this constant to true enables testing that behavior.
  const bool enableUseHostPtr = false;
  const bool performInitialCopy =
      (flags & UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER) ||
      ((flags & UR_MEM_FLAG_USE_HOST_POINTER) && !enableUseHostPtr);
  ur_result_t retErr = UR_RESULT_SUCCESS;
  ur_mem_handle_t retMemObj = nullptr;

  try {
    ScopedContext active(hContext);
    void *ptr;
    auto pHost = pProperties ? pProperties->pHost : nullptr;
    ur_mem_handle_t_::mem_::buffer_mem_::alloc_mode allocMode =
        ur_mem_handle_t_::mem_::buffer_mem_::alloc_mode::classic;

    if ((flags & UR_MEM_FLAG_USE_HOST_POINTER) && enableUseHostPtr) {
      retErr =
          UR_CHECK_ERROR(hipHostRegister(pHost, size, hipHostRegisterMapped));
      retErr = UR_CHECK_ERROR(hipHostGetDevicePointer(&ptr, pHost, 0));
      allocMode = ur_mem_handle_t_::mem_::buffer_mem_::alloc_mode::use_host_ptr;
    } else if (flags & UR_MEM_FLAG_ALLOC_HOST_POINTER) {
      retErr = UR_CHECK_ERROR(hipHostMalloc(&pHost, size));
      retErr = UR_CHECK_ERROR(hipHostGetDevicePointer(&ptr, pHost, 0));
      allocMode =
          ur_mem_handle_t_::mem_::buffer_mem_::alloc_mode::alloc_host_ptr;
    } else {
      retErr = UR_CHECK_ERROR(hipMalloc(&ptr, size));
      if (flags & UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER) {
        allocMode = ur_mem_handle_t_::mem_::buffer_mem_::alloc_mode::copy_in;
      }
    }

    if (retErr == UR_RESULT_SUCCESS) {
      ur_mem_handle_t parentBuffer = nullptr;

      auto devPtr = reinterpret_cast<
          ur_mem_handle_t_::mem_::mem_::buffer_mem_::native_type>(ptr);
      auto urMemObj = std::unique_ptr<ur_mem_handle_t_>(new ur_mem_handle_t_{
          hContext, parentBuffer, flags, allocMode, devPtr, pHost, size});
      if (urMemObj != nullptr) {
        retMemObj = urMemObj.release();
        if (performInitialCopy) {
          // Operates on the default stream of the current HIP context.
          retErr = UR_CHECK_ERROR(hipMemcpyHtoD(devPtr, pHost, size));
          // Synchronize with default stream implicitly used by hipMemcpyHtoD
          // to make buffer data available on device before any other UR call
          // uses it.
          if (retErr == UR_RESULT_SUCCESS) {
            hipStream_t defaultStream = 0;
            retErr = UR_CHECK_ERROR(hipStreamSynchronize(defaultStream));
          }
        }
      } else {
        retErr = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
      }
    }
  } catch (ur_result_t err) {
    retErr = err;
  } catch (...) {
    retErr = UR_RESULT_ERROR_OUT_OF_RESOURCES;
  }

  *phBuffer = retMemObj;

  return retErr;
}

/// Implements a buffer partition in the HIP backend.
/// A buffer partition (or a sub-buffer, in OpenCL terms) is simply implemented
/// as an offset over an existing HIP allocation.
///
UR_APIEXPORT ur_result_t UR_APICALL urMemBufferPartition(
    ur_mem_handle_t hBuffer, ur_mem_flags_t flags,
    ur_buffer_create_type_t bufferCreateType, const ur_buffer_region_t *pRegion,
    ur_mem_handle_t *phMem) {
  UR_ASSERT(hBuffer, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT((flags & UR_MEM_FLAGS_MASK) == 0,
            UR_RESULT_ERROR_INVALID_ENUMERATION);
  UR_ASSERT(hBuffer->is_buffer(), UR_RESULT_ERROR_INVALID_MEM_OBJECT);
  UR_ASSERT(!hBuffer->is_sub_buffer(), UR_RESULT_ERROR_INVALID_MEM_OBJECT);

  // Default value for flags means UR_MEM_FLAG_READ_WRITE.
  if (flags == 0) {
    flags = UR_MEM_FLAG_READ_WRITE;
  }

  UR_ASSERT(!(flags &
              (UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER |
               UR_MEM_FLAG_ALLOC_HOST_POINTER | UR_MEM_FLAG_USE_HOST_POINTER)),
            UR_RESULT_ERROR_INVALID_VALUE);
  if (hBuffer->memFlags_ & UR_MEM_FLAG_WRITE_ONLY) {
    UR_ASSERT(!(flags & (UR_MEM_FLAG_READ_WRITE | UR_MEM_FLAG_READ_ONLY)),
              UR_RESULT_ERROR_INVALID_VALUE);
  }
  if (hBuffer->memFlags_ & UR_MEM_FLAG_READ_ONLY) {
    UR_ASSERT(!(flags & (UR_MEM_FLAG_READ_WRITE | UR_MEM_FLAG_WRITE_ONLY)),
              UR_RESULT_ERROR_INVALID_VALUE);
  }

  UR_ASSERT(bufferCreateType == UR_BUFFER_CREATE_TYPE_REGION,
            UR_RESULT_ERROR_INVALID_ENUMERATION);
  UR_ASSERT(pRegion != nullptr, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  UR_ASSERT(phMem, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  UR_ASSERT(pRegion->size != 0u, UR_RESULT_ERROR_INVALID_BUFFER_SIZE);

  UR_ASSERT(((pRegion->origin + pRegion->size) <=
             hBuffer->mem_.buffer_mem_.get_size()),
            UR_RESULT_ERROR_INVALID_BUFFER_SIZE);
  // Retained indirectly due to retaining parent buffer below.
  ur_context_handle_t context = hBuffer->context_;
  ur_mem_handle_t_::mem_::buffer_mem_::alloc_mode allocMode =
      ur_mem_handle_t_::mem_::buffer_mem_::alloc_mode::classic;

  UR_ASSERT(hBuffer->mem_.buffer_mem_.ptr_ !=
                ur_mem_handle_t_::mem_::buffer_mem_::native_type{0},
            UR_RESULT_ERROR_INVALID_MEM_OBJECT);
  ur_mem_handle_t_::mem_::buffer_mem_::native_type ptr =
      hBuffer->mem_.buffer_mem_.get_with_offset(pRegion->origin);

  void *hostPtr = nullptr;
  if (hBuffer->mem_.buffer_mem_.hostPtr_) {
    hostPtr = static_cast<char *>(hBuffer->mem_.buffer_mem_.hostPtr_) +
              pRegion->origin;
  }

  ReleaseGuard<ur_mem_handle_t> releaseGuard(hBuffer);

  std::unique_ptr<ur_mem_handle_t_> retMemObj{nullptr};
  try {
    ScopedContext active(context);

    retMemObj = std::unique_ptr<ur_mem_handle_t_>{new ur_mem_handle_t_{
        context, hBuffer, flags, allocMode, ptr, hostPtr, pRegion->size}};
  } catch (ur_result_t err) {
    *phMem = nullptr;
    return err;
  } catch (...) {
    *phMem = nullptr;
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  }

  releaseGuard.dismiss();
  *phMem = retMemObj.release();
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemGetInfo(ur_mem_handle_t hMemory,
                                                 ur_mem_info_t MemInfoType,
                                                 size_t propSize,
                                                 void *pMemInfo,
                                                 size_t *pPropSizeRet) {

  UR_ASSERT(hMemory, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(MemInfoType <= UR_MEM_INFO_CONTEXT,
            UR_RESULT_ERROR_INVALID_ENUMERATION);
  UR_ASSERT(hMemory->is_buffer(), UR_RESULT_ERROR_INVALID_MEM_OBJECT);

  UrReturnHelper ReturnValue(propSize, pMemInfo, pPropSizeRet);

  ScopedContext active(hMemory->get_context());

  switch (MemInfoType) {
  case UR_MEM_INFO_SIZE: {
    try {
      size_t allocSize = 0;
      UR_CHECK_ERROR(hipMemGetAddressRange(nullptr, &allocSize,
                                           hMemory->mem_.buffer_mem_.ptr_));
      return ReturnValue(allocSize);
    } catch (ur_result_t err) {
      return err;
    } catch (...) {
      return UR_RESULT_ERROR_UNKNOWN;
    }
  }
  case UR_MEM_INFO_CONTEXT: {
    return ReturnValue(hMemory->get_context());
  }

  default:
    return UR_RESULT_ERROR_INVALID_ENUMERATION;
  }
}

/// Gets the native HIP handle of a UR mem object
///
/// \param[in] mem The UR mem to get the native HIP object of.
/// \param[out] nativeHandle Set to the native handle of the UR mem object.
///
/// \return UR_RESULT_SUCCESS
UR_APIEXPORT ur_result_t UR_APICALL
urMemGetNativeHandle(ur_mem_handle_t hMem, ur_native_handle_t *phNativeMem) {
#if defined(__HIP_PLATFORM_NVIDIA__)
  if (sizeof(ur_mem_handle_t_::mem_::buffer_mem_::native_type) >
      sizeof(ur_native_handle_t)) {
    // Check that all the upper bits that cannot be represented by
    // ur_native_handle_t are empty.
    // NOTE: The following shift might trigger a warning, but the check in the
    // if above makes sure that this does not underflow.
    ur_mem_handle_t_::mem_::buffer_mem_::native_type upperBits =
        hMem->mem_.buffer_mem_.get() >> (sizeof(ur_native_handle_t) * CHAR_BIT);
    if (upperBits) {
      // Return an error if any of the remaining bits is non-zero.
      return UR_RESULT_ERROR_INVALID_MEM_OBJECT;
    }
  }
  *phNativeMem =
      reinterpret_cast<ur_native_handle_t>(hMem->mem_.buffer_mem_.get());
#elif defined(__HIP_PLATFORM_AMD__)
  *phNativeMem =
      reinterpret_cast<ur_native_handle_t>(hMem->mem_.buffer_mem_.get());
#else
#error("Must define exactly one of __HIP_PLATFORM_AMD__ or __HIP_PLATFORM_NVIDIA__");
#endif
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemBufferCreateWithNativeHandle(
    ur_native_handle_t hNativeMem, ur_context_handle_t hContext,
    const ur_mem_native_properties_t *pProperties, ur_mem_handle_t *phMem) {
  std::ignore = hNativeMem;
  std::ignore = hContext;
  std::ignore = pProperties;
  std::ignore = phMem;
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemImageCreateWithNativeHandle(
    ur_native_handle_t hNativeMem, ur_context_handle_t hContext,
    const ur_image_format_t *pImageFormat, const ur_image_desc_t *pImageDesc,
    const ur_mem_native_properties_t *pProperties, ur_mem_handle_t *phMem) {
  std::ignore = hNativeMem;
  std::ignore = hContext;
  std::ignore = pImageFormat;
  std::ignore = pImageDesc;
  std::ignore = pProperties;
  std::ignore = phMem;
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

/// \TODO Not implemented
UR_APIEXPORT ur_result_t UR_APICALL urMemImageCreate(
    ur_context_handle_t hContext, ur_mem_flags_t flags,
    const ur_image_format_t *pImageFormat, const ur_image_desc_t *pImageDesc,
    void *pHost, ur_mem_handle_t *phMem) {

  // Need input memory object
  UR_ASSERT(hContext, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(phMem, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  UR_ASSERT(pImageDesc, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  UR_ASSERT((flags & UR_MEM_FLAGS_MASK) == 0,
            UR_RESULT_ERROR_INVALID_ENUMERATION);
  if (flags &
      (UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER | UR_MEM_FLAG_USE_HOST_POINTER)) {
    UR_ASSERT(pHost, UR_RESULT_ERROR_INVALID_HOST_PTR);
  }

  const bool performInitialCopy =
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

  ur_result_t retErr = UR_RESULT_SUCCESS;

  // We only support RBGA channel order
  // TODO: check SYCL CTS and spec. May also have to support BGRA
  UR_ASSERT(pImageFormat->channelOrder == UR_IMAGE_CHANNEL_ORDER_RGBA,
            UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION);

  // We have to use hipArray3DCreate, which has some caveats. The height and
  // depth parameters must be set to 0 produce 1D or 2D arrays. image_desc gives
  // a minimum value of 1, so we need to convert the answer.
  HIP_ARRAY3D_DESCRIPTOR array_desc;
  array_desc.NumChannels = 4; // Only support 4 channel image
  array_desc.Flags = 0;       // No flags required
  array_desc.Width = pImageDesc->width;
  if (pImageDesc->type == UR_MEM_TYPE_IMAGE1D) {
    array_desc.Height = 0;
    array_desc.Depth = 0;
  } else if (pImageDesc->type == UR_MEM_TYPE_IMAGE2D) {
    array_desc.Height = pImageDesc->height;
    array_desc.Depth = 0;
  } else if (pImageDesc->type == UR_MEM_TYPE_IMAGE3D) {
    array_desc.Height = pImageDesc->height;
    array_desc.Depth = pImageDesc->depth;
  }

  // We need to get this now in bytes for calculating the total image size later
  size_t pixel_type_size_bytes;

  switch (pImageFormat->channelType) {

  case UR_IMAGE_CHANNEL_TYPE_UNORM_INT8:
  case UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8:
    array_desc.Format = HIP_AD_FORMAT_UNSIGNED_INT8;
    pixel_type_size_bytes = 1;
    break;
  case UR_IMAGE_CHANNEL_TYPE_SIGNED_INT8:
    array_desc.Format = HIP_AD_FORMAT_SIGNED_INT8;
    pixel_type_size_bytes = 1;
    break;
  case UR_IMAGE_CHANNEL_TYPE_UNORM_INT16:
  case UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16:
    array_desc.Format = HIP_AD_FORMAT_UNSIGNED_INT16;
    pixel_type_size_bytes = 2;
    break;
  case UR_IMAGE_CHANNEL_TYPE_SIGNED_INT16:
    array_desc.Format = HIP_AD_FORMAT_SIGNED_INT16;
    pixel_type_size_bytes = 2;
    break;
  case UR_IMAGE_CHANNEL_TYPE_HALF_FLOAT:
    array_desc.Format = HIP_AD_FORMAT_HALF;
    pixel_type_size_bytes = 2;
    break;
  case UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32:
    array_desc.Format = HIP_AD_FORMAT_UNSIGNED_INT32;
    pixel_type_size_bytes = 4;
    break;
  case UR_IMAGE_CHANNEL_TYPE_SIGNED_INT32:
    array_desc.Format = HIP_AD_FORMAT_SIGNED_INT32;
    pixel_type_size_bytes = 4;
    break;
  case UR_IMAGE_CHANNEL_TYPE_FLOAT:
    array_desc.Format = HIP_AD_FORMAT_FLOAT;
    pixel_type_size_bytes = 4;
    break;
  default:
    // urMemImageCreate given unsupported image_channel_data_type
    return UR_RESULT_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR;
  }

  // When a dimension isn't used image_desc has the size set to 1
  size_t pixel_size_bytes =
      pixel_type_size_bytes * 4; // 4 is the only number of channels we support
  size_t image_size_bytes = pixel_size_bytes * pImageDesc->width *
                            pImageDesc->height * pImageDesc->depth;

  ScopedContext active(hContext);
  hipArray *image_array;
  retErr = UR_CHECK_ERROR(hipArray3DCreate(
      reinterpret_cast<hipCUarray *>(&image_array), &array_desc));

  try {
    if (performInitialCopy) {
      // We have to use a different copy function for each image dimensionality
      if (pImageDesc->type == UR_MEM_TYPE_IMAGE1D) {
        retErr = UR_CHECK_ERROR(
            hipMemcpyHtoA(image_array, 0, pHost, image_size_bytes));
      } else if (pImageDesc->type == UR_MEM_TYPE_IMAGE2D) {
        hip_Memcpy2D cpy_desc;
        memset(&cpy_desc, 0, sizeof(cpy_desc));
        cpy_desc.srcMemoryType = hipMemoryType::hipMemoryTypeHost;
        cpy_desc.srcHost = pHost;
        cpy_desc.dstMemoryType = hipMemoryType::hipMemoryTypeArray;
        cpy_desc.dstArray = reinterpret_cast<hipCUarray>(image_array);
        cpy_desc.WidthInBytes = pixel_size_bytes * pImageDesc->width;
        cpy_desc.Height = pImageDesc->height;
        retErr = UR_CHECK_ERROR(hipMemcpyParam2D(&cpy_desc));
      } else if (pImageDesc->type == UR_MEM_TYPE_IMAGE3D) {
        HIP_MEMCPY3D cpy_desc;
        memset(&cpy_desc, 0, sizeof(cpy_desc));
        cpy_desc.srcMemoryType = hipMemoryType::hipMemoryTypeHost;
        cpy_desc.srcHost = pHost;
        cpy_desc.dstMemoryType = hipMemoryType::hipMemoryTypeArray;
        cpy_desc.dstArray = reinterpret_cast<hipCUarray>(image_array);
        cpy_desc.WidthInBytes = pixel_size_bytes * pImageDesc->width;
        cpy_desc.Height = pImageDesc->height;
        cpy_desc.Depth = pImageDesc->depth;
        retErr = UR_CHECK_ERROR(hipDrvMemcpy3D(&cpy_desc));
      }
    }

    // HIP_RESOURCE_DESC is a union of different structs, shown here
    // We need to fill it as described here to use it for a surface or texture
    // HIP_RESOURCE_DESC::resType must be HIP_RESOURCE_TYPE_ARRAY and
    // HIP_RESOURCE_DESC::res::array::hArray must be set to a valid HIP array
    // handle.
    // HIP_RESOURCE_DESC::flags must be set to zero

    hipResourceDesc image_res_desc;
    image_res_desc.res.array.array = image_array;
    image_res_desc.resType = hipResourceTypeArray;

    hipSurfaceObject_t surface;
    retErr = UR_CHECK_ERROR(hipCreateSurfaceObject(&surface, &image_res_desc));

    auto urMemObj = std::unique_ptr<ur_mem_handle_t_>(new ur_mem_handle_t_{
        hContext, image_array, surface, flags, pImageDesc->type, pHost});

    if (urMemObj == nullptr) {
      return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

    *phMem = urMemObj.release();
  } catch (ur_result_t err) {
    UR_CHECK_ERROR(hipFreeArray(image_array));
    return err;
  } catch (...) {
    UR_CHECK_ERROR(hipFreeArray(image_array));
    return UR_RESULT_ERROR_UNKNOWN;
  }
  return retErr;
}

/// \TODO Not implemented
UR_APIEXPORT ur_result_t UR_APICALL
urMemImageGetInfo(ur_mem_handle_t hMemory, ur_image_info_t ImgInfoType,
                  size_t propSize, void *pImgInfo, size_t *pPropSizeRet) {
  std::ignore = hMemory;
  std::ignore = ImgInfoType;
  std::ignore = propSize;
  std::ignore = pImgInfo;
  std::ignore = pPropSizeRet;

  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemRetain(ur_mem_handle_t hMem) {
  UR_ASSERT(hMem, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(hMem->get_reference_count() > 0,
            UR_RESULT_ERROR_INVALID_MEM_OBJECT);
  hMem->increment_reference_count();
  return UR_RESULT_SUCCESS;
}