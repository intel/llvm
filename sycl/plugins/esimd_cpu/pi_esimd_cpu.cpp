//===---------- pi_esimd_cpu.cpp - CM Emulation Plugin --------------------===//
//
//
//===----------------------------------------------------------------------===//

/// \defgroup sycl_pi_esimd_cpu CM Emulation Plugin
/// \ingroup sycl_pi

/// \file pi_esimd_cpu.cpp
/// Declarations for CM Emulation Plugin. It is the interface between the
/// device-agnostic SYCL runtime layer and underlying CM Emulation
///
/// \ingroup sycl_pi_esimd_cpu

#include <stdint.h>

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <utility>

#include "pi_esimd_cpu.hpp"

namespace {

template <typename T, typename Assign>
pi_result getInfoImpl(size_t param_value_size, void *param_value,
                      size_t *param_value_size_ret, T value, size_t value_size,
                      Assign &&assign_func) {

  if (param_value != nullptr) {

    if (param_value_size < value_size) {
      return PI_INVALID_VALUE;
    }

    assign_func(param_value, value, value_size);
  }

  if (param_value_size_ret != nullptr) {
    *param_value_size_ret = value_size;
  }

  return PI_SUCCESS;
}

template <typename T>
pi_result getInfo(size_t param_value_size, void *param_value,
                  size_t *param_value_size_ret, T value) {

  auto assignment = [](void *param_value, T value, size_t value_size) {
    *static_cast<T *>(param_value) = value;
  };

  return getInfoImpl(param_value_size, param_value, param_value_size_ret, value,
                     sizeof(T), assignment);
}

template <typename T>
pi_result getInfoArray(size_t array_length, size_t param_value_size,
                       void *param_value, size_t *param_value_size_ret,
                       T *value) {
  return getInfoImpl(param_value_size, param_value, param_value_size_ret, value,
                     array_length * sizeof(T), memcpy);
}

template <>
pi_result getInfo<const char *>(size_t param_value_size, void *param_value,
                                size_t *param_value_size_ret,
                                const char *value) {
  return getInfoArray(strlen(value) + 1, param_value_size, param_value,
                      param_value_size_ret, value);
}

class ReturnHelper {
public:
  ReturnHelper(size_t param_value_size, void *param_value,
               size_t *param_value_size_ret)
      : param_value_size(param_value_size), param_value(param_value),
        param_value_size_ret(param_value_size_ret) {}

  template <class T> pi_result operator()(const T &t) {
    return getInfo(param_value_size, param_value, param_value_size_ret, t);
  }

private:
  size_t param_value_size;
  void *param_value;
  size_t *param_value_size_ret;
};

} // anonymous namespace

extern "C" {

#define DIE_NO_IMPLEMENTATION                                                  \
  std::cerr << "Not Implemented : " << __FUNCTION__                            \
            << " - File : " << __FILE__;                                       \
  std::cerr << " / Line : " << __LINE__ << std::endl;                          \
  die("Terminated")

#define DIE_NO_SUPPORT                                                         \
  std::cerr << "Not Supported : " << __FUNCTION__ << " - File : " << __FILE__; \
  std::cerr << " / Line : " << __LINE__ << std::endl;                          \
  die("Terminated")

#define CONTINUE_NO_IMPLEMENTATION                                             \
  std::cerr << "Warning : Not Implemented : " << __FUNCTION__                  \
            << " - File : " << __FILE__;                                       \
  std::cerr << " / Line : " << __LINE__ << std::endl;

pi_result piPlatformsGet(pi_uint32 NumEntries, pi_platform *Platforms,
                         pi_uint32 *NumPlatforms) {
  if (NumEntries == 0 && Platforms != nullptr) {
    return PI_INVALID_VALUE;
  }
  if (Platforms == nullptr && NumPlatforms == nullptr) {
    return PI_INVALID_VALUE;
  }

  if (Platforms && NumEntries > 0) {
    *Platforms = new _pi_platform();
    Platforms[0]->CmEmuVersion = std::string("0.0.1");
  }

  if (NumPlatforms) {
    *NumPlatforms = 1;
  }

  return PI_SUCCESS;
}

pi_result piPlatformGetInfo(pi_platform Platform, pi_platform_info ParamName,
                            size_t ParamValueSize, void *ParamValue,
                            size_t *ParamValueSizeRet) {
  assert(Platform);
  ReturnHelper ReturnValue(ParamValueSize, ParamValue, ParamValueSizeRet);

  switch (ParamName) {
  case PI_PLATFORM_INFO_NAME:
    return ReturnValue("Intel(R) ESIMD_CPU/GPU");

  case PI_PLATFORM_INFO_VENDOR:
    return ReturnValue("Intel(R) Corporation");

  case PI_PLATFORM_INFO_VERSION:
    return ReturnValue(Platform->CmEmuVersion);

  case PI_PLATFORM_INFO_PROFILE:
    return ReturnValue("CM_FULL_PROFILE");

  case PI_PLATFORM_INFO_EXTENSIONS:
    return ReturnValue("");

  default:
    // TODO: implement other parameters
    die("Unsupported ParamName in piPlatformGetInfo");
  }

  return PI_SUCCESS;
}

pi_result piextPlatformGetNativeHandle(pi_platform Platform,
                                       pi_native_handle *NativeHandle) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piextPlatformCreateWithNativeHandle(pi_native_handle NativeHandle,
                                              pi_platform *Platform) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piDevicesGet(pi_platform Platform, pi_device_type DeviceType,
                       pi_uint32 NumEntries, pi_device *Devices,
                       pi_uint32 *NumDevices) {

  if (NumEntries == 0) {
    if (NumDevices) {
      *NumDevices = 1;
    } else {
      return PI_INVALID_VALUE;
    }
  }

  if (NumDevices) {
    *NumDevices = 1;
  } else {
    // assert(NumEntries == 1);
    Devices[0] = new _pi_device(Platform);
  }

  return PI_SUCCESS;
}

pi_result piDeviceRetain(pi_device Device) {
  assert(Device);

  ++(Device->RefCount);

  return PI_SUCCESS;
}

pi_result piDeviceRelease(pi_device Device) {
  CONTINUE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piDeviceGetInfo(pi_device Device, pi_device_info ParamName,
                          size_t ParamValueSize, void *ParamValue,
                          size_t *ParamValueSizeRet) {
  ReturnHelper ReturnValue(ParamValueSize, ParamValue, ParamValueSizeRet);

  switch (ParamName) {
  case PI_DEVICE_INFO_TYPE:
    return ReturnValue(PI_DEVICE_TYPE_GPU);
  case PI_DEVICE_INFO_PARENT_DEVICE:
    return ReturnValue(pi_device{0});
  case PI_DEVICE_INFO_PLATFORM:
    return ReturnValue(Device->Platform);
  case PI_DEVICE_INFO_NAME:
    return ReturnValue("ESIMD_CPU");
  case PI_DEVICE_INFO_IMAGE_SUPPORT:
    return ReturnValue(pi_bool{true});
  case PI_DEVICE_INFO_DRIVER_VERSION:
    return ReturnValue("0.0.1");
  case PI_DEVICE_INFO_VENDOR:
    return ReturnValue("Intel(R) Corporation");
  case PI_DEVICE_INFO_IMAGE2D_MAX_WIDTH:
    return ReturnValue(size_t{8192});
  case PI_DEVICE_INFO_IMAGE2D_MAX_HEIGHT:
    return ReturnValue(size_t{8192});
  case PI_DEVICE_INFO_COMPILER_AVAILABLE:
    return ReturnValue(pi_bool{0});
  case PI_DEVICE_INFO_EXTENSIONS: {
    return ReturnValue("(No Extension)");
  }
  case PI_DEVICE_INFO_HOST_UNIFIED_MEMORY:
    return ReturnValue(pi_bool{1});

#define UNSUPPORTED_INFO(info)                                                 \
  case info:                                                                   \
    std::cerr << std::endl                                                     \
              << "Unsupported device info = " << #info << std::endl;           \
    DIE_NO_IMPLEMENTATION;                                                     \
    break;

    UNSUPPORTED_INFO(PI_DEVICE_INFO_VENDOR_ID)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_LINKER_AVAILABLE)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_MAX_COMPUTE_UNITS)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_MAX_WORK_ITEM_DIMENSIONS)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_MAX_WORK_GROUP_SIZE)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_MAX_WORK_ITEM_SIZES)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_MAX_CLOCK_FREQUENCY)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_ADDRESS_BITS)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_MAX_MEM_ALLOC_SIZE)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_GLOBAL_MEM_SIZE)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_LOCAL_MEM_SIZE)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_AVAILABLE)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_VERSION)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_PARTITION_MAX_SUB_DEVICES)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_REFERENCE_COUNT)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_PARTITION_PROPERTIES)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_PARTITION_TYPE)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_OPENCL_C_VERSION)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_PREFERRED_INTEROP_USER_SYNC)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_PRINTF_BUFFER_SIZE)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_PROFILE)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_BUILT_IN_KERNELS)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_QUEUE_PROPERTIES)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_EXECUTION_CAPABILITIES)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_ENDIAN_LITTLE)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_ERROR_CORRECTION_SUPPORT)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_PROFILING_TIMER_RESOLUTION)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_LOCAL_MEM_TYPE)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_MAX_CONSTANT_ARGS)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_MAX_CONSTANT_BUFFER_SIZE)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_GLOBAL_MEM_CACHE_TYPE)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_GLOBAL_MEM_CACHELINE_SIZE)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_GLOBAL_MEM_CACHE_SIZE)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_MAX_PARAMETER_SIZE)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_MEM_BASE_ADDR_ALIGN)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_MAX_SAMPLERS)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_MAX_READ_IMAGE_ARGS)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_MAX_WRITE_IMAGE_ARGS)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_SINGLE_FP_CONFIG)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_HALF_FP_CONFIG)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_DOUBLE_FP_CONFIG)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_IMAGE3D_MAX_WIDTH)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_IMAGE3D_MAX_HEIGHT)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_IMAGE3D_MAX_DEPTH)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_IMAGE_MAX_BUFFER_SIZE)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_IMAGE_MAX_ARRAY_SIZE)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_CHAR)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_CHAR)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_SHORT)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_SHORT)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_INT)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_INT)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_LONG)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_LONG)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_FLOAT)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_FLOAT)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_DOUBLE)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_DOUBLE)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_HALF)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_HALF)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_MAX_NUM_SUB_GROUPS)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_SUB_GROUP_SIZES_INTEL)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_IL_VERSION)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_USM_HOST_SUPPORT)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_USM_DEVICE_SUPPORT)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_USM_CROSS_SHARED_SUPPORT)
    UNSUPPORTED_INFO(PI_DEVICE_INFO_USM_SYSTEM_SHARED_SUPPORT)

#undef UNSUPPORTED_INFO
  default:
    DIE_NO_IMPLEMENTATION;
  }
  return PI_SUCCESS;
}

pi_result piDevicePartition(pi_device Device,
                            const pi_device_partition_property *Properties,
                            pi_uint32 NumDevices, pi_device *OutDevices,
                            pi_uint32 *OutNumDevices) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piextDeviceGetNativeHandle(pi_device Device,
                                     pi_native_handle *NativeHandle) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piextDeviceCreateWithNativeHandle(pi_native_handle nativeHandle,
                                            pi_platform platform,
                                            pi_device *device) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piContextCreate(const pi_context_properties *Properties,
                          pi_uint32 NumDevices, const pi_device *Devices,
                          void (*PFnNotify)(const char *ErrInfo,
                                            const void *PrivateInfo, size_t CB,
                                            void *UserData),
                          void *UserData, pi_context *RetContext) {
  if (NumDevices != 1) {
    return PI_INVALID_VALUE;
  }
  assert(Devices);
  assert(RetContext);

  cm_support::CmDevice *device = nullptr;
  unsigned int version = 0;

  int result = cm_support::CreateCmDevice(device, version);

  if (result != cm_support::CM_SUCCESS) {
    return PI_INVALID_VALUE;
  }

  try {
    *RetContext = new _pi_context(*Devices, device);
  } catch (const std::bad_alloc &) {
    return PI_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }

  return PI_SUCCESS;
}

pi_result piContextGetInfo(pi_context Context, pi_context_info ParamName,
                           size_t ParamValueSize, void *ParamValue,
                           size_t *ParamValueSizeRet) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piextContextSetExtendedDeleter(pi_context Context,
                                         pi_context_extended_deleter Function,
                                         void *UserData) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piextContextGetNativeHandle(pi_context Context,
                                      pi_native_handle *NativeHandle) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piextContextCreateWithNativeHandle(pi_native_handle NativeHandle,
                                             pi_uint32 NumDevices,
                                             const pi_device *Devices,
                                             pi_context *RetContext) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piContextRetain(pi_context Context) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piContextRelease(pi_context Context) {
  if ((Context == nullptr) || (Context->CmDevicePtr == nullptr)) {
    return PI_INVALID_CONTEXT;
  }

  int result = cm_support::DestroyCmDevice(Context->CmDevicePtr);
  if (result != cm_support::CM_SUCCESS) {
    return PI_INVALID_CONTEXT;
  }

  delete Context;
  return PI_SUCCESS;
}

pi_result piQueueCreate(pi_context Context, pi_device Device,
                        pi_queue_properties Properties, pi_queue *Queue) {
  cm_support::CmQueue *cmQueue;

  int result = Context->CmDevicePtr->CreateQueue(cmQueue);
  if (result != cm_support::CM_SUCCESS) {
    return PI_INVALID_CONTEXT;
  }

  try {
    *Queue = new _pi_queue(Context, cmQueue);
  } catch (const std::bad_alloc &) {
    return PI_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }

  return PI_SUCCESS;
}

pi_result piQueueGetInfo(pi_queue Queue, pi_queue_info ParamName,
                         size_t ParamValueSize, void *ParamValue,
                         size_t *ParamValueSizeRet) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piQueueRetain(pi_queue Queue) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piQueueRelease(pi_queue Queue) {
  if ((Queue == nullptr) || (Queue->CmQueuePtr == nullptr)) {
    return PI_INVALID_QUEUE;
  }

  // TODO : Destory 'Queue->CmQueuePtr'?
  delete Queue;

  return PI_SUCCESS;
}

pi_result piQueueFinish(pi_queue Queue) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piextQueueGetNativeHandle(pi_queue Queue,
                                    pi_native_handle *NativeHandle) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piextQueueCreateWithNativeHandle(pi_native_handle nativeHandle,
                                           pi_context context,
                                           pi_queue *queue) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piMemBufferCreate(pi_context Context, pi_mem_flags Flags, size_t Size,
                            void *HostPtr, pi_mem *RetMem,
                            const pi_mem_properties *properties) {
  assert((Flags & PI_MEM_FLAGS_ACCESS_RW) != 0);
  assert(Context);
  assert(RetMem);

  cm_support::CmBuffer *CmBuf = nullptr;
  cm_support::SurfaceIndex *CmIndex;

  int status = Context->CmDevicePtr->CreateBuffer(
      static_cast<unsigned int>(Size), CmBuf);

  if (status != cm_support::CM_SUCCESS) {
    // TODO : Convert "CM_" return value to "PI_" return value
    return PI_OUT_OF_HOST_MEMORY;
  }

  status = CmBuf->GetIndex(CmIndex);

  if ((Flags & PI_MEM_FLAGS_HOST_PTR_USE) != 0 ||
      (Flags & PI_MEM_FLAGS_HOST_PTR_COPY) != 0) {
    status =
        CmBuf->WriteSurface(reinterpret_cast<const unsigned char *>(HostPtr),
                            nullptr, static_cast<unsigned int>(Size));
  }

  auto HostPtrOrNull =
      (Flags & PI_MEM_FLAGS_HOST_PTR_USE) ? pi_cast<char *>(HostPtr) : nullptr;

  try {
    *RetMem = new _pi_buffer(Context, HostPtrOrNull, CmBuf, CmIndex);
  } catch (const std::bad_alloc &) {
    return PI_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }

  return PI_SUCCESS;
}

pi_result piMemGetInfo(pi_mem Mem,
                       cl_mem_info ParamName, // TODO: untie from OpenCL
                       size_t ParamValueSize, void *ParamValue,
                       size_t *ParamValueSizeRet) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piMemRetain(pi_mem Mem) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piMemRelease(pi_mem Mem) {
  if (Mem->getMemType() == PI_MEM_TYPE_BUFFER) {
    _pi_buffer *pi_buf = static_cast<_pi_buffer *>(Mem);
    int result = Mem->Context->CmDevicePtr->DestroySurface(pi_buf->CmBufferPtr);

    if (result != cm_support::CM_SUCCESS) {
      return PI_INVALID_MEM_OBJECT;
    }
  } else if (Mem->getMemType() == PI_MEM_TYPE_IMAGE2D) {
    _pi_image *pi_image = static_cast<_pi_image *>(Mem);
    int result =
        Mem->Context->CmDevicePtr->DestroySurface(pi_image->CmSurfacePtr);

    if (result != cm_support::CM_SUCCESS) {
      return PI_INVALID_MEM_OBJECT;
    }
  } else if (Mem->getMemType() == PI_MEM_TYPE_IMAGE2D) {
    _pi_image *pi_img = static_cast<_pi_image *>(Mem);
    int result =
        Mem->Context->CmDevicePtr->DestroySurface(pi_img->CmSurfacePtr);
    if (result != cm_support::CM_SUCCESS) {
      return PI_INVALID_MEM_OBJECT;
    }
  } else {
    return PI_INVALID_MEM_OBJECT;
  }

  return PI_SUCCESS;
}

cm_support::CM_SURFACE_FORMAT
piImageFormatToCmFormat(const pi_image_format *piFormat) {
  using ULongPair = std::pair<unsigned long, unsigned long>;
  using FmtMap = std::map<ULongPair, cm_support::CM_SURFACE_FORMAT>;
  static const FmtMap pi2cm = {
      {{PI_IMAGE_CHANNEL_TYPE_UNORM_INT8, PI_IMAGE_CHANNEL_ORDER_RGBA},
       cm_support::CM_SURFACE_FORMAT_A8R8G8B8},

      {{PI_IMAGE_CHANNEL_TYPE_UNORM_INT8, PI_IMAGE_CHANNEL_ORDER_ARGB},
       cm_support::CM_SURFACE_FORMAT_A8R8G8B8},

      {{PI_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8, PI_IMAGE_CHANNEL_ORDER_RGBA},
       cm_support::CM_SURFACE_FORMAT_A8R8G8B8},

      {{PI_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32, PI_IMAGE_CHANNEL_ORDER_RGBA},
       cm_support::CM_SURFACE_FORMAT_R32G32B32A32F},
  };
  auto result = pi2cm.find(
      {piFormat->image_channel_data_type, piFormat->image_channel_order});
  if (result != pi2cm.end()) {
    return result->second;
  }
  DIE_NO_IMPLEMENTATION;
  return cm_support::CM_SURFACE_FORMAT_A8R8G8B8;
}

pi_result piMemImageCreate(pi_context Context, pi_mem_flags Flags,
                           const pi_image_format *ImageFormat,
                           const pi_image_desc *ImageDesc, void *HostPtr,
                           pi_mem *RetImage) {
  if (ImageFormat == nullptr || ImageDesc == nullptr)
    return PI_INVALID_VALUE;

  switch (ImageDesc->image_type) {
  case PI_MEM_TYPE_IMAGE2D:
    break;
  default:
    return PI_INVALID_MEM_OBJECT;
  }

  auto bytesPerPixel = 4;
  switch (ImageFormat->image_channel_data_type) {
  case PI_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32:
    bytesPerPixel = 16;
    break;
  case PI_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8:
  case PI_IMAGE_CHANNEL_TYPE_UNORM_INT8:
    bytesPerPixel = 4;
    break;
  default:
    return PI_INVALID_VALUE;
  }

  cm_support::CmSurface2D *CmSurface = nullptr;
  cm_support::SurfaceIndex *CmIndex;

  int status = Context->CmDevicePtr->CreateSurface2D(
      static_cast<unsigned int>(ImageDesc->image_width),
      static_cast<unsigned int>(ImageDesc->image_height),
      piImageFormatToCmFormat(ImageFormat), CmSurface);

  if (status != cm_support::CM_SUCCESS) {
    // TODO : Convert "CM_" return value to "PI_" return value
    return PI_OUT_OF_HOST_MEMORY;
  }

  status = CmSurface->GetIndex(CmIndex);

  if ((Flags & PI_MEM_FLAGS_HOST_PTR_USE) != 0 ||
      (Flags & PI_MEM_FLAGS_HOST_PTR_COPY) != 0) {

    if (HostPtr != nullptr) {

      status = CmSurface->WriteSurface(
          reinterpret_cast<const unsigned char *>(HostPtr), nullptr,
          static_cast<unsigned int>(ImageDesc->image_width *
                                    ImageDesc->image_height * bytesPerPixel));
    }
  }

  auto HostPtrOrNull =
      (Flags & PI_MEM_FLAGS_HOST_PTR_USE) ? pi_cast<char *>(HostPtr) : nullptr;

  try {
    *RetImage = new _pi_image(Context, HostPtrOrNull, CmSurface, CmIndex);
  } catch (const std::bad_alloc &) {
    return PI_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }

  return PI_SUCCESS;
}

pi_result piextMemGetNativeHandle(pi_mem Mem, pi_native_handle *NativeHandle) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piextMemCreateWithNativeHandle(pi_native_handle NativeHandle,
                                         pi_mem *Mem) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piProgramCreate(pi_context Context, const void *IL, size_t Length,
                          pi_program *Program) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piProgramCreateWithBinary(pi_context Context, pi_uint32 NumDevices,
                                    const pi_device *DeviceList,
                                    const size_t *Lengths,
                                    const unsigned char **Binaries,
                                    pi_int32 *BinaryStatus,
                                    pi_program *Program) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piclProgramCreateWithBinary(pi_context Context, pi_uint32 NumDevices,
                                      const pi_device *DeviceList,
                                      const size_t *Lengths,
                                      const unsigned char **Binaries,
                                      pi_int32 *BinaryStatus,
                                      pi_program *RetProgram) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piclProgramCreateWithSource(pi_context Context, pi_uint32 Count,
                                      const char **Strings,
                                      const size_t *Lengths,
                                      pi_program *RetProgram) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piProgramGetInfo(pi_program Program, pi_program_info ParamName,
                           size_t ParamValueSize, void *ParamValue,
                           size_t *ParamValueSizeRet) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piProgramLink(pi_context Context, pi_uint32 NumDevices,
                        const pi_device *DeviceList, const char *Options,
                        pi_uint32 NumInputPrograms,
                        const pi_program *InputPrograms,
                        void (*PFnNotify)(pi_program Program, void *UserData),
                        void *UserData, pi_program *RetProgram) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piProgramCompile(
    pi_program Program, pi_uint32 NumDevices, const pi_device *DeviceList,
    const char *Options, pi_uint32 NumInputHeaders,
    const pi_program *InputHeaders, const char **HeaderIncludeNames,
    void (*PFnNotify)(pi_program Program, void *UserData), void *UserData) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piProgramBuild(pi_program Program, pi_uint32 NumDevices,
                         const pi_device *DeviceList, const char *Options,
                         void (*PFnNotify)(pi_program Program, void *UserData),
                         void *UserData) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piProgramGetBuildInfo(pi_program Program, pi_device Device,
                                cl_program_build_info ParamName,
                                size_t ParamValueSize, void *ParamValue,
                                size_t *ParamValueSizeRet) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piProgramRetain(pi_program Program) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piProgramRelease(pi_program Program) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piextProgramGetNativeHandle(pi_program Program,
                                      pi_native_handle *NativeHandle) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piextProgramCreateWithNativeHandle(pi_native_handle nativeHandle,
                                             pi_context context,
                                             pi_program *program) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piKernelCreate(pi_program Program, const char *KernelName,
                         pi_kernel *RetKernel) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piKernelSetArg(pi_kernel Kernel, pi_uint32 ArgIndex, size_t ArgSize,
                         const void *ArgValue) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piextKernelSetArgMemObj(pi_kernel Kernel, pi_uint32 ArgIndex,
                                  const pi_mem *ArgValue) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

// Special version of piKernelSetArg to accept pi_sampler.
pi_result piextKernelSetArgSampler(pi_kernel Kernel, pi_uint32 ArgIndex,
                                   const pi_sampler *ArgValue) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piKernelGetInfo(pi_kernel Kernel, pi_kernel_info ParamName,
                          size_t ParamValueSize, void *ParamValue,
                          size_t *ParamValueSizeRet) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piKernelGetGroupInfo(pi_kernel Kernel, pi_device Device,
                               pi_kernel_group_info ParamName,
                               size_t ParamValueSize, void *ParamValue,
                               size_t *ParamValueSizeRet) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piKernelGetSubGroupInfo(
    pi_kernel Kernel, pi_device Device,
    pi_kernel_sub_group_info ParamName, // TODO: untie from OpenCL
    size_t InputValueSize, const void *InputValue, size_t ParamValueSize,
    void *ParamValue, size_t *ParamValueSizeRet) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piKernelRetain(pi_kernel Kernel) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piKernelRelease(pi_kernel Kernel) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result
piEnqueueKernelLaunch(pi_queue Queue, pi_kernel Kernel, pi_uint32 WorkDim,
                      const size_t *GlobalWorkOffset,
                      const size_t *GlobalWorkSize, const size_t *LocalWorkSize,
                      pi_uint32 NumEventsInWaitList,
                      const pi_event *EventWaitList, pi_event *Event) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piEventCreate(pi_context Context, pi_event *RetEvent) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piEventGetInfo(pi_event Event, pi_event_info ParamName,
                         size_t ParamValueSize, void *ParamValue,
                         size_t *ParamValueSizeRet) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piEventGetProfilingInfo(pi_event Event, pi_profiling_info ParamName,
                                  size_t ParamValueSize, void *ParamValue,
                                  size_t *ParamValueSizeRet) {
  std::cerr << "Warning : Profiling Not supported under PI_ESIMD_CPU"
            << std::endl;
  return PI_SUCCESS;
}

pi_result piEventsWait(pi_uint32 NumEvents, const pi_event *EventList) {
  for (int i = 0; i < (int)NumEvents; i++) {
    if (EventList[i]->IsDummyEvent) {
      continue;
    }
    int result = EventList[i]->CmEventPtr->WaitForTaskFinished();
    if (result != cm_support::CM_SUCCESS) {
      return PI_OUT_OF_RESOURCES;
    }
  }
  return PI_SUCCESS;
}

pi_result piEventSetCallback(pi_event Event, pi_int32 CommandExecCallbackType,
                             void (*PFnNotify)(pi_event Event,
                                               pi_int32 EventCommandStatus,
                                               void *UserData),
                             void *UserData) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piEventSetStatus(pi_event Event, pi_int32 ExecutionStatus) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piEventRetain(pi_event Event) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piEventRelease(pi_event Event) {
  if (!Event->IsDummyEvent) {
    if ((Event->CmEventPtr == nullptr) || (Event->OwnerQueue == nullptr)) {
      return PI_INVALID_EVENT;
    }
    int result = Event->OwnerQueue->DestroyEvent(Event->CmEventPtr);
    if (result != cm_support::CM_SUCCESS) {
      return PI_INVALID_EVENT;
    }
  }
  delete Event;

  return PI_SUCCESS;
}

pi_result piextEventGetNativeHandle(pi_event Event,
                                    pi_native_handle *NativeHandle) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piextEventCreateWithNativeHandle(pi_native_handle NativeHandle,
                                           pi_event *Event) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}
pi_result piSamplerCreate(pi_context Context,
                          const pi_sampler_properties *SamplerProperties,
                          pi_sampler *RetSampler) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piSamplerGetInfo(pi_sampler Sampler, pi_sampler_info ParamName,
                           size_t ParamValueSize, void *ParamValue,
                           size_t *ParamValueSizeRet) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piSamplerRetain(pi_sampler Sampler) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piSamplerRelease(pi_sampler Sampler) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piEnqueueEventsWait(pi_queue Queue, pi_uint32 NumEventsInWaitList,
                              const pi_event *EventWaitList, pi_event *Event) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piEnqueueEventsWaitWithBarrier(pi_queue Queue,
                                         pi_uint32 NumEventsInWaitList,
                                         const pi_event *EventWaitList,
                                         pi_event *Event) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piEnqueueMemBufferRead(pi_queue Queue, pi_mem Src,
                                 pi_bool BlockingRead, size_t Offset,
                                 size_t Size, void *Dst,
                                 pi_uint32 NumEventsInWaitList,
                                 const pi_event *EventWaitList,
                                 pi_event *Event) {
  /// TODO : Support Blocked read, 'Queue' handling
  assert(BlockingRead == false);
  assert(NumEventsInWaitList == 0);

  _pi_buffer *buf = static_cast<_pi_buffer *>(Src);

  int status =
      buf->CmBufferPtr->ReadSurface(reinterpret_cast<unsigned char *>(Dst),
                                    nullptr, // event
                                    static_cast<uint64_t>(Size));

  if (status != cm_support::CM_SUCCESS) {
    return PI_INVALID_MEM_OBJECT;
  }

  if (Event) {
    try {
      *Event = new _pi_event();
    } catch (const std::bad_alloc &) {
      return PI_OUT_OF_HOST_MEMORY;
    } catch (...) {
      return PI_ERROR_UNKNOWN;
    }
    (*Event)->IsDummyEvent = true;
  }

  return PI_SUCCESS;
}

pi_result piEnqueueMemBufferReadRect(
    pi_queue command_queue, pi_mem buffer, pi_bool blocking_read,
    pi_buff_rect_offset buffer_offset, pi_buff_rect_offset host_offset,
    pi_buff_rect_region region, size_t buffer_row_pitch,
    size_t buffer_slice_pitch, size_t host_row_pitch, size_t host_slice_pitch,
    void *ptr, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piEnqueueMemBufferWrite(pi_queue Queue, pi_mem Buffer,
                                  pi_bool BlockingWrite, size_t Offset,
                                  size_t Size, const void *Ptr,
                                  pi_uint32 NumEventsInWaitList,
                                  const pi_event *EventWaitList,
                                  pi_event *Event) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piEnqueueMemBufferWriteRect(
    pi_queue command_queue, pi_mem buffer, pi_bool blocking_write,
    pi_buff_rect_offset buffer_offset, pi_buff_rect_offset host_offset,
    pi_buff_rect_region region, size_t buffer_row_pitch,
    size_t buffer_slice_pitch, size_t host_row_pitch, size_t host_slice_pitch,
    const void *ptr, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piEnqueueMemBufferCopy(pi_queue Queue, pi_mem SrcBuffer,
                                 pi_mem DstBuffer, size_t SrcOffset,
                                 size_t DstOffset, size_t Size,
                                 pi_uint32 NumEventsInWaitList,
                                 const pi_event *EventWaitList,
                                 pi_event *Event) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piEnqueueMemBufferCopyRect(
    pi_queue command_queue, pi_mem src_buffer, pi_mem dst_buffer,
    pi_buff_rect_offset src_origin, pi_buff_rect_offset dst_origin,
    pi_buff_rect_region region, size_t src_row_pitch, size_t src_slice_pitch,
    size_t dst_row_pitch, size_t dst_slice_pitch,
    pi_uint32 num_events_in_wait_list, const pi_event *event_wait_list,
    pi_event *event) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piEnqueueMemBufferFill(pi_queue Queue, pi_mem Buffer,
                                 const void *Pattern, size_t PatternSize,
                                 size_t Offset, size_t Size,
                                 pi_uint32 NumEventsInWaitList,
                                 const pi_event *EventWaitList,
                                 pi_event *Event) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result
piEnqueueMemBufferMap(pi_queue Queue, pi_mem Buffer, pi_bool BlockingMap,
                      cl_map_flags MapFlags, // TODO: untie from OpenCL
                      size_t Offset, size_t Size, pi_uint32 NumEventsInWaitList,
                      const pi_event *EventWaitList, pi_event *Event,
                      void **RetMap) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piEnqueueMemUnmap(pi_queue Queue, pi_mem MemObj, void *MappedPtr,
                            pi_uint32 NumEventsInWaitList,
                            const pi_event *EventWaitList, pi_event *Event) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piMemImageGetInfo(pi_mem Image, pi_image_info ParamName,
                            size_t ParamValueSize, void *ParamValue,
                            size_t *ParamValueSizeRet) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piEnqueueMemImageRead(pi_queue command_queue, pi_mem image,
                                pi_bool blocking_read, pi_image_offset origin,
                                pi_image_region region, size_t row_pitch,
                                size_t slice_pitch, void *ptr,
                                pi_uint32 num_events_in_wait_list,
                                const pi_event *event_wait_list,
                                pi_event *event) {
  _pi_image *img = static_cast<_pi_image *>(image);
  int status =
      img->CmSurfacePtr->ReadSurface(reinterpret_cast<unsigned char *>(ptr),
                                     nullptr, // event
                                     row_pitch * (region->height));
  if (status != cm_support::CM_SUCCESS) {
    return PI_INVALID_MEM_OBJECT;
  }

  if (event) {
    try {
      *event = new _pi_event();
    } catch (const std::bad_alloc &) {
      return PI_OUT_OF_HOST_MEMORY;
    } catch (...) {
      return PI_ERROR_UNKNOWN;
    }
    (*event)->IsDummyEvent = true;
  }
  return PI_SUCCESS;
}

pi_result piEnqueueMemImageWrite(pi_queue command_queue, pi_mem image,
                                 pi_bool blocking_write, pi_image_offset origin,
                                 pi_image_region region, size_t input_row_pitch,
                                 size_t input_slice_pitch, const void *ptr,
                                 pi_uint32 num_events_in_wait_list,
                                 const pi_event *event_wait_list,
                                 pi_event *event) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piEnqueueMemImageCopy(pi_queue command_queue, pi_mem src_image,
                                pi_mem dst_image, pi_image_offset src_origin,
                                pi_image_offset dst_origin,
                                pi_image_region region,
                                pi_uint32 num_events_in_wait_list,
                                const pi_event *event_wait_list,
                                pi_event *event) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piEnqueueMemImageFill(pi_queue Queue, pi_mem Image,
                                const void *FillColor, const size_t *Origin,
                                const size_t *Region,
                                pi_uint32 NumEventsInWaitList,
                                const pi_event *EventWaitList,
                                pi_event *Event) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piMemBufferPartition(pi_mem Buffer, pi_mem_flags Flags,
                               pi_buffer_create_type BufferCreateType,
                               void *BufferCreateInfo, pi_mem *RetMem) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piEnqueueNativeKernel(pi_queue Queue, void (*UserFunc)(void *),
                                void *Args, size_t CbArgs,
                                pi_uint32 NumMemObjects, const pi_mem *MemList,
                                const void **ArgsMemLoc,
                                pi_uint32 NumEventsInWaitList,
                                const pi_event *EventWaitList,
                                pi_event *Event) {

  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piextGetDeviceFunctionPointer(pi_device Device, pi_program Program,
                                        const char *FunctionName,
                                        pi_uint64 *FunctionPointerRet) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piextUSMHostAlloc(void **ResultPtr, pi_context Context,
                            pi_usm_mem_properties *Properties, size_t Size,
                            pi_uint32 Alignment) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piextUSMDeviceAlloc(void **ResultPtr, pi_context Context,
                              pi_device Device,
                              pi_usm_mem_properties *Properties, size_t Size,
                              pi_uint32 Alignment) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piextUSMSharedAlloc(void **ResultPtr, pi_context Context,
                              pi_device Device,
                              pi_usm_mem_properties *Properties, size_t Size,
                              pi_uint32 Alignment) {
  assert(Context);
  assert(ResultPtr);

  cm_support::CmBufferSVM *buf = nullptr;
  void *pSystemMem = nullptr;
  int32_t ret = Context->CmDevicePtr->CreateBufferSVM(
      Size, pSystemMem, CM_SVM_ACCESS_FLAG_DEFAULT, buf);
  assert(cm_support::CM_SUCCESS == ret);
  *ResultPtr = pSystemMem;
  auto it = Context->Addr2CmBufferSVM.find(pSystemMem);
  assert(Context->Addr2CmBufferSVM.end() == it);
  Context->Addr2CmBufferSVM[pSystemMem] = buf;
  return PI_SUCCESS;
}

pi_result piextUSMFree(pi_context Context, void *Ptr) {
  assert(Context);
  assert(Ptr);

  cm_support::CmBufferSVM *buf = Context->Addr2CmBufferSVM[Ptr];
  assert(buf);
  auto count = Context->Addr2CmBufferSVM.erase(Ptr);
  assert(1 == count);
  int32_t ret = Context->CmDevicePtr->DestroyBufferSVM(buf);
  assert(cm_support::CM_SUCCESS == ret);
  return PI_SUCCESS;
}

pi_result piextKernelSetArgPointer(pi_kernel Kernel, pi_uint32 ArgIndex,
                                   size_t ArgSize, const void *ArgValue) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piextUSMEnqueueMemset(pi_queue Queue, void *Ptr, pi_int32 Value,
                                size_t Count, pi_uint32 NumEventsInWaitlist,
                                const pi_event *EventsWaitlist,
                                pi_event *Event) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piextUSMEnqueueMemcpy(pi_queue Queue, pi_bool Blocking, void *DstPtr,
                                const void *SrcPtr, size_t Size,
                                pi_uint32 NumEventsInWaitlist,
                                const pi_event *EventsWaitlist,
                                pi_event *Event) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piextUSMEnqueueMemAdvise(pi_queue Queue, const void *Ptr,
                                   size_t Length, pi_mem_advice Advice,
                                   pi_event *Event) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piextUSMGetMemAllocInfo(pi_context Context, const void *Ptr,
                                  pi_mem_info ParamName, size_t ParamValueSize,
                                  void *ParamValue, size_t *ParamValueSizeRet) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piKernelSetExecInfo(pi_kernel Kernel, pi_kernel_exec_info ParamName,
                              size_t ParamValueSize, const void *ParamValue) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piextProgramSetSpecializationConstant(pi_program Prog,
                                                pi_uint32 SpecID,
                                                size_t SpecSize,
                                                const void *SpecValue) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result
piextDeviceSelectBinary(pi_device Device, // TODO: does this need to be context?
                        pi_device_binary *Binaries, pi_uint32 NumBinaries,
                        pi_uint32 *SelectedBinaryInd) {
  CONTINUE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piextUSMEnqueuePrefetch(pi_queue Queue, const void *Ptr, size_t Size,
                                  pi_usm_migration_flags Flags,
                                  pi_uint32 NumEventsInWaitlist,
                                  const pi_event *EventsWaitlist,
                                  pi_event *Event) {
  DIE_NO_IMPLEMENTATION;
  return PI_SUCCESS;
}

pi_result piPluginInit(pi_plugin *PluginInit) {
  assert(PluginInit);
  size_t PluginVersionSize = sizeof(PluginInit->PluginVersion);
  assert(strlen(_PI_H_VERSION_STRING) < PluginVersionSize);
  strncpy(PluginInit->PluginVersion, _PI_H_VERSION_STRING, PluginVersionSize);

#define _PI_API(api)                                                           \
  (PluginInit->PiFunctionTable).api = (decltype(&::api))(&api);
#include <CL/sycl/detail/pi.def>

  return PI_SUCCESS;
}
}

template <typename T> using KernelFunc = std::function<void(T)>;

template <int NDims> struct InvokeBaseImpl {
  static sycl::range<NDims> get_range(const size_t *GlobalWorkSize);
};

template <int NDims, typename ArgTy> struct InvokeImpl {

  template <int _NDims = NDims>
  static typename std::enable_if<_NDims == 1, sycl::range<1>>::type
  get_range(const size_t *a) {
    return sycl::range<1>(a[0]);
  }

  template <int _NDims = NDims>
  static typename std::enable_if<_NDims == 2, sycl::range<2>>::type
  get_range(const size_t *a) {
    return sycl::range<2>(a[0], a[1]);
  }

  template <int _NDims = NDims>
  static typename std::enable_if<_NDims == 3, sycl::range<3>>::type
  get_range(const size_t *a) {
    return sycl::range<3>(a[0], a[1], a[2]);
  }

  static void invoke(void *fptr, const sycl::range<NDims> &range) {
    auto f = reinterpret_cast<std::function<void(const ArgTy &)> *>(fptr);
    libCMBatch<KernelFunc<const ArgTy &>, ArgTy, ArgTy, NDims> cmThreading(*f);
    cmThreading.runIterationSpace(range);
  }

  static void invoke(void *fptr, const size_t *GlobalWorkSize) {
    sycl::range<NDims> range = get_range(GlobalWorkSize);
    invoke(fptr, range);
  }

  static void invoke(void *fptr, const size_t *GlobalWorkOffset,
                     const size_t *GlobalWorkSize) {
    auto GlobalSize = get_range(GlobalWorkSize);
    sycl::id<NDims> GlobalOffset = get_range(GlobalWorkOffset);

    auto f = reinterpret_cast<std::function<void(const ArgTy &)> *>(fptr);
    libCMBatch<KernelFunc<const ArgTy &>, ArgTy, ArgTy, NDims> cmThreading(*f);
    cmThreading.runIterationSpace(GlobalSize, GlobalOffset);
  }

  static void invoke(void *fptr, const size_t *GlobalWorkOffset,
                     const size_t *GlobalWorkSize,
                     const size_t *LocalWorkSize) {
    auto GlobalSize = get_range(GlobalWorkSize);
    auto LocalSize = get_range(LocalWorkSize);
    auto NumGroups = GlobalSize / LocalSize;
    sycl::id<NDims> GlobalOffset = get_range(GlobalWorkOffset);

    // TODO: cmThreading should be 'CmThreading'. Not a descriptive name.
    // Keeping as is for for now, to maintain consistency with cg_types.hpp.
    // Will rename when
    // __SYCL_EXPLICIT_SIMD_PLUGIN_DEPRECATED__ is removed.
    auto f = reinterpret_cast<std::function<void(const ArgTy &)> *>(fptr);
    libCMBatch<KernelFunc<const ArgTy &>, ArgTy, ArgTy, NDims> cmThreading(*f);

    // TODO: This is inefficient - groups are run sequentially one
    // after another. Change the loop to run groups in parallel.
    using IDBuilder = sycl::detail::Builder;
    sycl::detail::NDLoop<NDims>::iterate(
        NumGroups, [&](const sycl::id<NDims> &GroupID) {
          sycl::group<NDims> Group = IDBuilder::createGroup<NDims>(
              GlobalSize, LocalSize, NumGroups, GroupID);
          cmThreading.runIterationSpace(GroupID, Group, LocalSize, GlobalSize,
                                        GlobalOffset);
        });
  }
};

template <int NDims>
static pi_result enqueueHostKernelLaunch(void *Kernel,
                                         pi_host_kernel_arg_type ArgType,
                                         const size_t *GlobalWorkOffset,
                                         const size_t *GlobalWorkSize,
                                         const size_t *LocalWorkSize) {
#define IS_NULL(NDims, R)                                                      \
  ((0 == R[0]) && (1 >= NDims || 0 == R[1]) && (2 >= NDims || 0 == R[2]))

  switch (ArgType) {
  case PI_HOST_KERNEL_ARG_TYPE_ID: {
    assert(IS_NULL(NDims, GlobalWorkOffset));
    assert(IS_NULL(NDims, LocalWorkSize));
    InvokeImpl<NDims, sycl::id<NDims>>::invoke(Kernel, GlobalWorkSize);
    break;
  }

  case PI_HOST_KERNEL_ARG_TYPE_ITEM: {
    assert(IS_NULL(NDims, GlobalWorkOffset));
    assert(IS_NULL(NDims, LocalWorkSize));
    InvokeImpl<NDims, sycl::item<NDims, /*Offset=*/false>>::invoke(
        Kernel, GlobalWorkSize);
    break;
  }

  case PI_HOST_KERNEL_ARG_TYPE_ITEM_OFFSET: {
    assert(IS_NULL(NDims, LocalWorkSize));
    InvokeImpl<NDims, sycl::item<NDims, /*Offset=*/true>>::invoke(
        Kernel, GlobalWorkOffset, GlobalWorkSize);
    break;
  }

  case PI_HOST_KERNEL_ARG_TYPE_ND_ITEM: {
    InvokeImpl<NDims, sycl::nd_item<NDims>>::invoke(
        Kernel, GlobalWorkOffset, GlobalWorkSize, LocalWorkSize);
    break;
  }

  default:
    std::cerr << "Error: Invalid kernel argument type." << std::endl;
    return PI_ERROR_UNKNOWN;
  }

  return PI_SUCCESS;
}

// TODO: Support waitlist.
extern "C" {
pi_result piEnqueueHostKernelLaunch(
    pi_queue Queue, void *Kernel, pi_host_kernel_arg_type ArgType,
    pi_uint32 WorkDim, const size_t *GlobalWorkOffset,
    const size_t *GlobalWorkSize, const size_t *LocalWorkSize,
    pi_uint32 NumEventsInWaitList, const pi_event *EventWaitList,
    pi_event *Event) {
  switch (WorkDim) {
  case 1:
    return enqueueHostKernelLaunch<1>(Kernel, ArgType, GlobalWorkOffset,
                                      GlobalWorkSize, LocalWorkSize);
  case 2:
    return enqueueHostKernelLaunch<2>(Kernel, ArgType, GlobalWorkOffset,
                                      GlobalWorkSize, LocalWorkSize);
  case 3:
    return enqueueHostKernelLaunch<3>(Kernel, ArgType, GlobalWorkOffset,
                                      GlobalWorkSize, LocalWorkSize);
  }
}

} // extern C
