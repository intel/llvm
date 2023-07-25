#include <atomic>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <sycl/detail/cg_types.hpp> // NDRDescT
#include <sycl/detail/native_cpu.hpp>
#include <sycl/detail/pi.h>

static bool PrintPiTrace = true;

struct _pi_object {
  _pi_object() : RefCount{1} {}

  std::atomic<pi_uint32> RefCount;
};

struct _pi_device : _pi_object {
  _pi_device(pi_platform ArgPlt) : Platform{ArgPlt} {}

  pi_platform Platform;
};

struct _pi_platform {
  _pi_device TheDevice{this};
};

struct _pi_mem : _pi_object {
  _pi_mem(size_t Size) {
    _mem = (char *)malloc(Size);
    _owns_mem = true;
  }
  _pi_mem(void *HostPtr, size_t Size) {
    _mem = (char *)malloc(Size);
    memcpy(_mem, HostPtr, Size);
    _owns_mem = true;
  }
  _pi_mem(void *HostPtr) {
    _mem = (char *)HostPtr;
    _owns_mem = false;
  }
  ~_pi_mem() {
    if (_owns_mem)
      free(_mem);
  }
  char *_mem;
  bool _owns_mem;
};

struct _pi_context : _pi_object {
  _pi_device *_device;
};

struct _pi_program : _pi_object {
  _pi_context *_ctx;
  const unsigned char *_ptr;
};

using nativecpu_kernel_t = void(const sycl::detail::NativeCPUArgDesc *,
                                __nativecpu_state *);
using nativecpu_ptr_t = nativecpu_kernel_t *;
using nativecpu_task_t = std::function<nativecpu_kernel_t>;
struct _pi_kernel : _pi_object {
  const char *_name;
  nativecpu_task_t _subhandler;
  std::vector<sycl::detail::NativeCPUArgDesc> _args;
};

// taken from pi_cuda.cpp
template <typename T, typename Assign>
pi_result getInfoImpl(size_t param_value_size, void *param_value,
                      size_t *param_value_size_ret, T value, size_t value_size,
                      Assign &&assign_func) {

  if (param_value != nullptr) {
    if (param_value_size < value_size) {
      return PI_ERROR_INVALID_VALUE;
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
    // Ignore unused parameter
    (void)value_size;

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

sycl::detail::NDRDescT getNDRDesc(pi_uint32 WorkDim,
                                  const size_t *GlobalWorkOffset,
                                  const size_t *GlobalWorkSize,
                                  const size_t *LocalWorkSize) {
  // Todo: we flip indexes here, I'm not sure we should, if we don't we need to
  // un-flip them in the spirv builtins definitions as well
  sycl::detail::NDRDescT Res;
  switch (WorkDim) {
  case 1:
    Res.set<1>(sycl::nd_range<1>({GlobalWorkSize[0]}, {LocalWorkSize[0]},
                                 {GlobalWorkOffset[0]}));
    break;
  case 2:
    Res.set<2>(sycl::nd_range<2>({GlobalWorkSize[0], GlobalWorkSize[1]},
                                 {LocalWorkSize[0], LocalWorkSize[1]},
                                 {GlobalWorkOffset[0], GlobalWorkOffset[1]}));
    break;
  case 3:
    Res.set<3>(sycl::nd_range<3>(
        {GlobalWorkSize[0], GlobalWorkSize[1], GlobalWorkSize[2]},
        {LocalWorkSize[0], LocalWorkSize[1], LocalWorkSize[2]},
        {GlobalWorkOffset[0], GlobalWorkOffset[1], GlobalWorkOffset[2]}));
    break;
  }
  return Res;
}

extern "C" {
#define DIE_NO_IMPLEMENTATION                                                  \
  if (PrintPiTrace) {                                                          \
    std::cerr << "Not Implemented : " << __FUNCTION__                          \
              << " - File : " << __FILE__;                                     \
    std::cerr << " / Line : " << __LINE__ << std::endl;                        \
  }                                                                            \
  return PI_ERROR_INVALID_OPERATION;

#define CONTINUE_NO_IMPLEMENTATION                                             \
  if (PrintPiTrace) {                                                          \
    std::cerr << "Warning : Not Implemented : " << __FUNCTION__                \
              << " - File : " << __FILE__;                                     \
    std::cerr << " / Line : " << __LINE__ << std::endl;                        \
  }                                                                            \
  return PI_SUCCESS;

#define CASE_PI_UNSUPPORTED(not_supported)                                     \
  case not_supported:                                                          \
    if (PrintPiTrace) {                                                        \
      std::cerr << std::endl                                                   \
                << "Unsupported PI case : " << #not_supported << " in "        \
                << __FUNCTION__ << ":" << __LINE__ << "(" << __FILE__ << ")"   \
                << std::endl;                                                  \
    }                                                                          \
    return PI_ERROR_INVALID_OPERATION;

pi_result piPlatformsGet(pi_uint32 NumEntries, pi_platform *Platforms,
                         pi_uint32 *NumPlatforms) {
  if (NumPlatforms == nullptr && Platforms == nullptr)
    return PI_ERROR_INVALID_VALUE;
  static const char *PiTrace = std::getenv("SYCL_PI_TRACE");
  static const int PiTraceValue = PiTrace ? std::stoi(PiTrace) : 0;
  if (PiTraceValue == -1) {
    PrintPiTrace = true;
  }
  if (NumPlatforms) {
    *NumPlatforms = 1;
  }

  if (NumEntries == 0) {
    if (Platforms != nullptr) {
      if (PrintPiTrace)
        std::cerr << "Invalid argument combination for piPlatformsGet\n";
      return PI_ERROR_INVALID_VALUE;
    }
    return PI_SUCCESS;
  }
  if (Platforms && NumEntries > 0) {
    static _pi_platform ThePlatform;
    *Platforms = &ThePlatform;
  }
  return PI_SUCCESS;
}

pi_result piPlatformGetInfo(pi_platform Platform, pi_platform_info ParamName,
                            size_t ParamValueSize, void *ParamValue,
                            size_t *ParamValueSizeRet) {
  if (Platform == nullptr) {
    return PI_ERROR_INVALID_PLATFORM;
  }
  auto ReturnValueArray = [=](auto val) {
    return getInfoArray(strlen(val) + 1, ParamValueSize, ParamValue,
                        ParamValueSizeRet, val);
  };

  switch (ParamName) {
  case PI_PLATFORM_INFO_NAME:
    return ReturnValueArray("SYCL_NATIVE_CPU");

  case PI_PLATFORM_INFO_VENDOR:
    return ReturnValueArray("tbd");

  case PI_PLATFORM_INFO_VERSION:
    return ReturnValueArray("0.1");

  case PI_PLATFORM_INFO_PROFILE:
    return ReturnValueArray("FULL_PROFILE");

  case PI_PLATFORM_INFO_EXTENSIONS:
    return ReturnValueArray("");

  case PI_EXT_PLATFORM_INFO_BACKEND:
    return getInfo<pi_platform_backend>(
        ParamValueSize, ParamValue, ParamValueSizeRet,
        _pi_platform_backend::PI_EXT_PLATFORM_BACKEND_NATIVE_CPU);

  default:
    DIE_NO_IMPLEMENTATION;
  }

  return PI_SUCCESS;
}

pi_result piextPlatformGetNativeHandle(pi_platform, pi_native_handle *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextPlatformCreateWithNativeHandle(pi_native_handle, pi_platform *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piDevicesGet(pi_platform Platform, pi_device_type DeviceType,
                       pi_uint32 NumEntries, pi_device *Devices,
                       pi_uint32 *NumDevices) {
  if (Platform == nullptr) {
    return PI_ERROR_INVALID_PLATFORM;
  }

  pi_uint32 DeviceCount = (DeviceType & PI_DEVICE_TYPE_CPU) ? 1 : 0;

  if (NumDevices) {
    *NumDevices = DeviceCount;
  }

  if (NumEntries == 0) {
    /// Runtime queries number of devices
    if (Devices != nullptr) {
      if (PrintPiTrace) {
        std::cerr << "Invalid Arguments for piDevicesGet\n";
      }
      return PI_ERROR_INVALID_VALUE;
    }
    return PI_SUCCESS;
  }

  if (DeviceCount == 0) {
    /// No GPU entry to fill 'Device' array
    return PI_SUCCESS;
  }

  if (Devices) {
    Devices[0] = &Platform->TheDevice;
  }

  return PI_SUCCESS;
}

pi_result piDeviceRetain(pi_device Device) {
  if (Device)
    return PI_SUCCESS;
  return PI_ERROR_INVALID_DEVICE;
}

pi_result piDeviceRelease(pi_device Device) {
  if (Device)
    return PI_SUCCESS;
  return PI_ERROR_INVALID_DEVICE;
}

pi_result piDeviceGetInfo(pi_device Device, pi_device_info ParamName,
                          size_t ParamValueSize, void *ParamValue,
                          size_t *ParamValueSizeRet) {
  auto ReturnValueArray = [=](auto val) {
    return getInfoArray(strlen(val) + 1, ParamValueSize, ParamValue,
                        ParamValueSizeRet, val);
  };
  auto ReturnValue = [=](auto val) {
    return getInfo(ParamValueSize, ParamValue, ParamValueSizeRet, val);
  };

  switch (ParamName) {
  case PI_DEVICE_INFO_TYPE:
    return ReturnValue(PI_DEVICE_TYPE_CPU);
  case PI_DEVICE_INFO_PARENT_DEVICE:
    return ReturnValue(pi_device{0});
  case PI_DEVICE_INFO_PLATFORM:
    return ReturnValue(Device->Platform);
  case PI_DEVICE_INFO_NAME:
    return ReturnValueArray("SYCL Native CPU");
  case PI_DEVICE_INFO_IMAGE_SUPPORT:
    return ReturnValue(pi_bool{false});
  case PI_DEVICE_INFO_DRIVER_VERSION:
    return ReturnValueArray("0.0.0");
  case PI_DEVICE_INFO_VENDOR:
    return ReturnValueArray("Intel(R) Corporation");
  case PI_DEVICE_INFO_IMAGE2D_MAX_WIDTH:
    return ReturnValue(size_t{8192});
  case PI_DEVICE_INFO_IMAGE2D_MAX_HEIGHT:
    return ReturnValue(size_t{8192});
  case PI_DEVICE_INFO_HOST_UNIFIED_MEMORY:
    return ReturnValue(pi_bool{1});
  case PI_DEVICE_INFO_EXTENSIONS:
    // TODO : Populate return string accordingly - e.g. cl_khr_fp16,
    // cl_khr_fp64, cl_khr_int64_base_atomics,
    // cl_khr_int64_extended_atomics
    return ReturnValue("");
  case PI_DEVICE_INFO_VERSION:
    return ReturnValueArray("0.1");
  case PI_DEVICE_INFO_COMPILER_AVAILABLE:
    return ReturnValue(pi_bool{false});
  case PI_DEVICE_INFO_LINKER_AVAILABLE:
    return ReturnValue(pi_bool{false});
  case PI_DEVICE_INFO_MAX_COMPUTE_UNITS:
    return ReturnValue(pi_uint32{256});
  case PI_DEVICE_INFO_PARTITION_MAX_SUB_DEVICES:
    return ReturnValue(pi_uint32{0});
  case PI_DEVICE_INFO_PARTITION_PROPERTIES:
    return ReturnValue(pi_device_partition_property{0});
  case PI_DEVICE_INFO_VENDOR_ID:
    // '0x8086' : 'Intel HD graphics vendor ID'
    return ReturnValue(pi_uint32{0x8086});
  case PI_DEVICE_INFO_MAX_WORK_GROUP_SIZE:
    return ReturnValue(size_t{256});
  case PI_DEVICE_INFO_MEM_BASE_ADDR_ALIGN:
    // Imported from level_zero
    return ReturnValue(pi_uint32{8});
  case PI_DEVICE_INFO_IMAGE3D_MAX_WIDTH:
  case PI_DEVICE_INFO_IMAGE3D_MAX_HEIGHT:
  case PI_DEVICE_INFO_IMAGE3D_MAX_DEPTH:
    // Default minimum values required by the SYCL specification.
    return ReturnValue(size_t{2048});
  case PI_DEVICE_INFO_MAX_WORK_ITEM_DIMENSIONS:
    return ReturnValue(pi_uint32{3});
  case PI_DEVICE_INFO_PARTITION_TYPE:
    return ReturnValue(pi_device_partition_property{0});
  case PI_DEVICE_INFO_OPENCL_C_VERSION:
    return ReturnValue("");
  case PI_DEVICE_INFO_QUEUE_PROPERTIES:
    return ReturnValue(pi_queue_properties{});
  case PI_DEVICE_INFO_MAX_WORK_ITEM_SIZES: {
    struct {
      size_t Arr[3];
    } MaxGroupSize = {{256, 256, 1}};
    return ReturnValue(MaxGroupSize);
  }
  case PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_CHAR:
  case PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_SHORT:
  case PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_INT:
  case PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_LONG:
  case PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_FLOAT:
  case PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_DOUBLE:
  case PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_HALF:
  case PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_CHAR:
  case PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_SHORT:
  case PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_INT:
  case PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_LONG:
  case PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_FLOAT:
  case PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_DOUBLE:
  case PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_HALF:
    return ReturnValue(pi_uint32{1});

  // Imported from level_zero
  case PI_DEVICE_INFO_USM_HOST_SUPPORT:
  case PI_DEVICE_INFO_USM_DEVICE_SUPPORT:
  case PI_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT:
  case PI_DEVICE_INFO_USM_CROSS_SHARED_SUPPORT:
  case PI_DEVICE_INFO_USM_SYSTEM_SHARED_SUPPORT: {
    pi_uint64 Supported = 0;
    // TODO[1.0]: how to query for USM support now?
    if (true) {
      // TODO: Use ze_memory_access_capabilities_t
      Supported = PI_USM_ACCESS | PI_USM_ATOMIC_ACCESS |
                  PI_USM_CONCURRENT_ACCESS | PI_USM_CONCURRENT_ATOMIC_ACCESS;
    }
    return ReturnValue(Supported);
  }
  case PI_DEVICE_INFO_ADDRESS_BITS:
    return ReturnValue(
        pi_uint32{sizeof(void *) * std::numeric_limits<unsigned char>::digits});
  case PI_DEVICE_INFO_MAX_CLOCK_FREQUENCY:
    return ReturnValue(pi_uint32{1000});
  case PI_DEVICE_INFO_ENDIAN_LITTLE:
    return ReturnValue(pi_bool{true});
  case PI_DEVICE_INFO_AVAILABLE:
    return ReturnValue(pi_bool{true});
  case PI_DEVICE_INFO_MAX_READ_IMAGE_ARGS:
  case PI_DEVICE_INFO_MAX_WRITE_IMAGE_ARGS:
    /// TODO : Check
    return ReturnValue(pi_uint32{0});
  case PI_DEVICE_INFO_IMAGE_MAX_ARRAY_SIZE:
    /// TODO : Check
    return ReturnValue(size_t{0});
  case PI_DEVICE_INFO_MAX_PARAMETER_SIZE:
    /// TODO : Check
    return ReturnValue(size_t{32});
  case PI_DEVICE_INFO_GLOBAL_MEM_CACHE_TYPE:
    return ReturnValue(PI_DEVICE_MEM_CACHE_TYPE_READ_WRITE_CACHE);
  case PI_DEVICE_INFO_GLOBAL_MEM_CACHELINE_SIZE:
    // TODO : CHECK
    return ReturnValue(pi_uint32{64});
  case PI_DEVICE_INFO_GLOBAL_MEM_CACHE_SIZE:
    // TODO : CHECK
    return ReturnValue(pi_uint64{0});
  case PI_DEVICE_INFO_GLOBAL_MEM_SIZE:
    // TODO : CHECK
    return ReturnValue(pi_uint64{0});
  case PI_DEVICE_INFO_MAX_CONSTANT_BUFFER_SIZE:
    // TODO : CHECK
    return ReturnValue(pi_uint64{0});
  case PI_DEVICE_INFO_MAX_CONSTANT_ARGS:
    // TODO : CHECK
    return ReturnValue(pi_uint32{64});
  case PI_DEVICE_INFO_LOCAL_MEM_TYPE:
    // TODO : CHECK
    return ReturnValue(PI_DEVICE_LOCAL_MEM_TYPE_LOCAL);
  case PI_DEVICE_INFO_ERROR_CORRECTION_SUPPORT:
    return ReturnValue(pi_bool{false});
  case PI_DEVICE_INFO_PROFILING_TIMER_RESOLUTION:
    // TODO : CHECK
    return ReturnValue(size_t{0});
  case PI_DEVICE_INFO_BUILT_IN_KERNELS:
    // TODO : CHECK
    return ReturnValue("");
  case PI_DEVICE_INFO_PRINTF_BUFFER_SIZE:
    // TODO : CHECK
    return ReturnValue(size_t{1024});
  case PI_DEVICE_INFO_PREFERRED_INTEROP_USER_SYNC:
    return ReturnValue(pi_bool{false});
  case PI_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN:
    return ReturnValue(pi_device_affinity_domain{0});
  case PI_DEVICE_INFO_MAX_MEM_ALLOC_SIZE:
    // TODO : CHECK
    return ReturnValue(pi_uint64{0});
  case PI_DEVICE_INFO_EXECUTION_CAPABILITIES:
    // TODO : CHECK
    return ReturnValue(
        pi_device_exec_capabilities{PI_DEVICE_EXEC_CAPABILITIES_KERNEL});
  case PI_DEVICE_INFO_PROFILE:
    return ReturnValue("FULL_PROFILE");
  case PI_DEVICE_INFO_REFERENCE_COUNT:
    // TODO : CHECK
    return ReturnValue(pi_uint32{0});
  case PI_DEVICE_INFO_BUILD_ON_SUBDEVICE:
    return ReturnValue(pi_bool{0});
  case PI_DEVICE_INFO_ATOMIC_64:
    return ReturnValue(pi_bool{0});
  case PI_EXT_ONEAPI_DEVICE_INFO_BFLOAT16_MATH_FUNCTIONS:
    return ReturnValue(pi_bool{0});
  case PI_EXT_INTEL_DEVICE_INFO_MEM_CHANNEL_SUPPORT:
    return ReturnValue(pi_bool{0});
  case PI_DEVICE_INFO_IMAGE_SRGB:
    return ReturnValue(pi_bool{0});
  case PI_DEVICE_INFO_SUB_GROUP_SIZES_INTEL:
    return ReturnValue(pi_uint32{1});
  case PI_DEVICE_INFO_GPU_EU_COUNT:
  case PI_DEVICE_INFO_PCI_ADDRESS:
  case PI_DEVICE_INFO_GPU_SLICES:
  case PI_DEVICE_INFO_GPU_EU_COUNT_PER_SUBSLICE:
  case PI_DEVICE_INFO_GPU_SUBSLICES_PER_SLICE:
  case PI_DEVICE_INFO_GPU_EU_SIMD_WIDTH:
  case PI_EXT_ONEAPI_DEVICE_INFO_MAX_WORK_GROUPS_1D:
  case PI_EXT_ONEAPI_DEVICE_INFO_MAX_WORK_GROUPS_2D:
  case PI_EXT_ONEAPI_DEVICE_INFO_MAX_WORK_GROUPS_3D:
  case PI_DEVICE_INFO_GPU_HW_THREADS_PER_EU:
  case PI_EXT_ONEAPI_DEVICE_INFO_CUDA_ASYNC_BARRIER:
  case PI_EXT_INTEL_DEVICE_INFO_FREE_MEMORY:
  case PI_DEVICE_INFO_UUID:
  case PI_DEVICE_INFO_DEVICE_ID:
  case PI_EXT_INTEL_DEVICE_INFO_MEMORY_CLOCK_RATE:
  case PI_EXT_INTEL_DEVICE_INFO_MEMORY_BUS_WIDTH:
  case PI_DEVICE_INFO_MAX_NUM_SUB_GROUPS:
  case PI_DEVICE_INFO_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS:
  case PI_DEVICE_INFO_IL_VERSION:
    return PI_ERROR_INVALID_VALUE;

    // Intel-specific extensions
    CASE_PI_UNSUPPORTED(PI_DEVICE_INFO_MAX_MEM_BANDWIDTH)
    CASE_PI_UNSUPPORTED(PI_EXT_ONEAPI_DEVICE_INFO_MAX_GLOBAL_WORK_GROUPS)

  default:
    DIE_NO_IMPLEMENTATION;
  }
  return PI_SUCCESS;
}

pi_result piDevicePartition(pi_device, const pi_device_partition_property *,
                            pi_uint32, pi_device *, pi_uint32 *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextDeviceGetNativeHandle(pi_device, pi_native_handle *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextDeviceCreateWithNativeHandle(pi_native_handle, pi_platform,
                                            pi_device *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piContextCreate(const pi_context_properties *Properties,
                          pi_uint32 NumDevices, const pi_device *Devices,
                          void (*PFnNotify)(const char *ErrInfo,
                                            const void *PrivateInfo, size_t CB,
                                            void *UserData),
                          void *UserData, pi_context *RetContext) {

  // Todo: proper error checking
  assert(NumDevices == 1 && Devices != nullptr);
  _pi_context *ctx = new _pi_context();
  ctx->_device = Devices[0];
  *RetContext = ctx;
  return PI_SUCCESS;
}

pi_result piContextGetInfo(pi_context, pi_context_info param_name,
                           size_t param_value_size, void *param_value,
                           size_t *param_value_size_ret) {
  switch (param_name) {
  case PI_CONTEXT_INFO_NUM_DEVICES:
    return getInfo(param_value_size, param_value, param_value_size_ret, 1);
  case PI_CONTEXT_INFO_DEVICES:
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   nullptr);
  case PI_CONTEXT_INFO_REFERENCE_COUNT:
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   nullptr);
  case PI_EXT_ONEAPI_CONTEXT_INFO_USM_MEMCPY2D_SUPPORT:
    return getInfo<pi_bool>(param_value_size, param_value, param_value_size_ret,
                            true);
  case PI_EXT_ONEAPI_CONTEXT_INFO_USM_FILL2D_SUPPORT:
  case PI_EXT_ONEAPI_CONTEXT_INFO_USM_MEMSET2D_SUPPORT:
    // 2D USM operations currently not supported.
    return getInfo<pi_bool>(param_value_size, param_value, param_value_size_ret,
                            false);
  case PI_EXT_CONTEXT_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES:
  case PI_EXT_CONTEXT_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES:
  case PI_EXT_CONTEXT_INFO_ATOMIC_FENCE_ORDER_CAPABILITIES:
  case PI_EXT_CONTEXT_INFO_ATOMIC_FENCE_SCOPE_CAPABILITIES: {
    // These queries should be dealt with in context_impl.cpp by calling the
    // queries of each device separately and building the intersection set.
    // setErrorMessage("These queries should have never come here.",
    //               PI_ERROR_INVALID_ARG_VALUE);
    return PI_ERROR_PLUGIN_SPECIFIC_ERROR;
  }
  default:
    __SYCL_PI_HANDLE_UNKNOWN_PARAM_NAME(param_name);
  }

  return PI_ERROR_OUT_OF_RESOURCES;
}

pi_result piextContextSetExtendedDeleter(pi_context,
                                         pi_context_extended_deleter, void *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextContextGetNativeHandle(pi_context, pi_native_handle *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextContextCreateWithNativeHandle(pi_native_handle, pi_uint32,
                                             const pi_device *, bool,
                                             pi_context *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piContextRetain(pi_context Context) { DIE_NO_IMPLEMENTATION; }

pi_result piContextRelease(pi_context Context) {
  // Todo: is it fine as a no-op?
  return PI_SUCCESS;
}

pi_result piQueueCreate(pi_context Context, pi_device Device,
                        pi_queue_properties Properties, pi_queue *Queue) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piQueueGetInfo(pi_queue, pi_queue_info, size_t, void *, size_t *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piQueueRetain(pi_queue Queue) { DIE_NO_IMPLEMENTATION; }

pi_result piQueueRelease(pi_queue Queue) {
  // Todo: is it fine as a no-op?
  return PI_SUCCESS;
}

pi_result piQueueFinish(pi_queue) {
  // Todo: is it fine as a no-op?
  return PI_SUCCESS;
}

pi_result piQueueFlush(pi_queue) { DIE_NO_IMPLEMENTATION; }

pi_result piextQueueGetNativeHandle(pi_queue, pi_native_handle *, int32_t *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextQueueCreateWithNativeHandle(pi_native_handle, int32_t,
                                           pi_context, pi_device, bool,
                                           pi_queue_properties *, pi_queue *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piMemBufferCreate(pi_context Context, pi_mem_flags Flags, size_t Size,
                            void *HostPtr, pi_mem *RetMem,
                            const pi_mem_properties *properties) {
  // TODO: add proper error checking and double check flag semantics
  // TODO: support PI_MEM_FLAGS_HOST_PTR_ALLOC flag
  const bool UseHostPtr = Flags & PI_MEM_FLAGS_HOST_PTR_USE;
  const bool CopyHostPtr = Flags & PI_MEM_FLAGS_HOST_PTR_COPY;
  const bool HostPtrNotNull = HostPtr != nullptr;
  if (UseHostPtr && HostPtrNotNull) {
    *RetMem = new _pi_mem(HostPtr);
  } else if (CopyHostPtr && HostPtrNotNull) {
    *RetMem = new _pi_mem(HostPtr, Size);
  } else {
    *RetMem = new _pi_mem(Size);
  }
  return PI_SUCCESS;
}

pi_result piMemGetInfo(pi_mem, pi_mem_info, size_t, void *, size_t *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piMemRetain(pi_mem Mem) { DIE_NO_IMPLEMENTATION; }

pi_result piMemRelease(pi_mem Mem) {
  uint32_t OldRefCount = Mem->RefCount.fetch_sub(1, std::memory_order_acq_rel);
  if (OldRefCount == 1) {
    delete Mem;
  }

  return PI_SUCCESS;
}

pi_result piMemImageCreate(pi_context Context, pi_mem_flags Flags,
                           const pi_image_format *ImageFormat,
                           const pi_image_desc *ImageDesc, void *HostPtr,
                           pi_mem *RetImage) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextMemGetNativeHandle(pi_mem, pi_native_handle *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextMemCreateWithNativeHandle(pi_native_handle, pi_context, bool,
                                         pi_mem *) {

  DIE_NO_IMPLEMENTATION;
}

pi_result piProgramCreate(pi_context, const void *, size_t, pi_program *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piProgramCreateWithBinary(pi_context context, pi_uint32,
                                    const pi_device *, const size_t *,
                                    const unsigned char **binaries, size_t,
                                    const pi_device_binary_property *,
                                    pi_int32 *, pi_program *program) {
  // Todo: proper error checking
  assert(binaries);
  auto p = new _pi_program();
  p->_ptr = binaries[0];
  p->_ctx = context;
  *program = p;
  return PI_SUCCESS;
}

pi_result piclProgramCreateWithBinary(pi_context, pi_uint32, const pi_device *,
                                      const size_t *, const unsigned char **,
                                      pi_int32 *, pi_program *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piclProgramCreateWithSource(pi_context, pi_uint32, const char **,
                                      const size_t *, pi_program *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piProgramGetInfo(pi_program program, pi_program_info param_name,
                           size_t param_value_size, void *param_value,
                           size_t *param_value_size_ret) {
  assert(program != nullptr);

  switch (param_name) {
  case PI_PROGRAM_INFO_REFERENCE_COUNT:
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   &program->RefCount);
  case PI_PROGRAM_INFO_CONTEXT:
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   nullptr);
  case PI_PROGRAM_INFO_NUM_DEVICES:
    return getInfo(param_value_size, param_value, param_value_size_ret, 1u);
  case PI_PROGRAM_INFO_DEVICES:
    return getInfoArray(1, param_value_size, param_value, param_value_size_ret,
                        program->_ctx->_device);
  case PI_PROGRAM_INFO_SOURCE:
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   nullptr);
  case PI_PROGRAM_INFO_BINARY_SIZES:
    return getInfoArray(1, param_value_size, param_value, param_value_size_ret,
                        "foo");
  case PI_PROGRAM_INFO_BINARIES:
    return getInfoArray(1, param_value_size, param_value, param_value_size_ret,
                        "foo");
  case PI_PROGRAM_INFO_KERNEL_NAMES: {
    return getInfo(param_value_size, param_value, param_value_size_ret, "foo");
  }
  default:
    __SYCL_PI_HANDLE_UNKNOWN_PARAM_NAME(param_name);
  }
  sycl::detail::pi::die("Program info request not implemented");
  return {};
}

pi_result piProgramLink(pi_context, pi_uint32, const pi_device *, const char *,
                        pi_uint32, const pi_program *,
                        void (*)(pi_program, void *), void *, pi_program *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piProgramCompile(pi_program, pi_uint32, const pi_device *,
                           const char *, pi_uint32, const pi_program *,
                           const char **, void (*)(pi_program, void *),
                           void *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piProgramBuild(pi_program, pi_uint32, const pi_device *, const char *,
                         void (*)(pi_program, void *), void *) {
  return PI_SUCCESS;
}

pi_result piProgramGetBuildInfo(pi_program, pi_device,
                                pi_program_build_info param_name, size_t,
                                void *, size_t *) {
  CONTINUE_NO_IMPLEMENTATION;
}

pi_result piProgramRetain(pi_program) { DIE_NO_IMPLEMENTATION; }

pi_result piProgramRelease(pi_program program) {
  delete program;
  return PI_SUCCESS;
}

pi_result piextProgramGetNativeHandle(pi_program, pi_native_handle *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextProgramCreateWithNativeHandle(pi_native_handle, pi_context, bool,
                                             pi_program *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piKernelCreate(pi_program program, const char *name,
                         pi_kernel *kernel) {
  // Todo: error checking
  auto ker = new _pi_kernel();
  auto f = reinterpret_cast<nativecpu_ptr_t>(program->_ptr);
  ker->_subhandler = *f;
  ker->_name = name;
  *kernel = ker;
  return PI_SUCCESS;
}

pi_result piKernelSetArg(pi_kernel kernel, pi_uint32, size_t, const void *arg) {
  // Todo: error checking
  // Todo: I think that the opencl spec (and therefore the pi spec mandates that
  // arg is copied (this is why it is defined as const void*, I guess we should
  // do it
  kernel->_args.emplace_back(const_cast<void *>(arg));
  return PI_SUCCESS;
}

pi_result piextKernelSetArgMemObj(pi_kernel kernel, pi_uint32,
                                  const pi_mem_obj_property *,
                                  const pi_mem *memObj) {
  // Todo: error checking
  _pi_mem *memPtr = *memObj;
  kernel->_args.emplace_back(memPtr->_mem);
  return PI_SUCCESS;
}

pi_result piextKernelSetArgSampler(pi_kernel, pi_uint32, const pi_sampler *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piKernelGetInfo(pi_kernel, pi_kernel_info, size_t, void *, size_t *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piKernelGetGroupInfo(pi_kernel kernel, pi_device,
                               pi_kernel_group_info param_name,
                               size_t param_value_size, void *param_value,
                               size_t *param_value_size_ret) {
  // Todo: return something meaningful here, we could emit info in our
  // integration header and read them here?

  if (kernel != nullptr) {

    switch (param_name) {
    case PI_KERNEL_GROUP_INFO_GLOBAL_WORK_SIZE: {
      size_t global_work_size[3] = {0, 0, 0};

      return getInfoArray(3, param_value_size, param_value,
                          param_value_size_ret, global_work_size);
    }
    case PI_KERNEL_GROUP_INFO_WORK_GROUP_SIZE: {
      size_t max_threads = 0;
      return getInfo(param_value_size, param_value, param_value_size_ret,
                     size_t(max_threads));
    }
    case PI_KERNEL_GROUP_INFO_COMPILE_WORK_GROUP_SIZE: {
      size_t group_size[3] = {1, 1, 1};
      return getInfoArray(3, param_value_size, param_value,
                          param_value_size_ret, group_size);
    }
    case PI_KERNEL_GROUP_INFO_LOCAL_MEM_SIZE: {
      int bytes = 0;
      return getInfo(param_value_size, param_value, param_value_size_ret,
                     pi_uint64(bytes));
    }
    case PI_KERNEL_GROUP_INFO_PREFERRED_WORK_GROUP_SIZE_MULTIPLE: {
      int warpSize = 0;
      return getInfo(param_value_size, param_value, param_value_size_ret,
                     static_cast<size_t>(warpSize));
    }
    case PI_KERNEL_GROUP_INFO_PRIVATE_MEM_SIZE: {
      int bytes = 0;
      return getInfo(param_value_size, param_value, param_value_size_ret,
                     pi_uint64(bytes));
    }
    case PI_KERNEL_GROUP_INFO_NUM_REGS: {
      sycl::detail::pi::die("PI_KERNEL_GROUP_INFO_NUM_REGS in "
                            "piKernelGetGroupInfo not implemented\n");
      return {};
    }

    default:
      __SYCL_PI_HANDLE_UNKNOWN_PARAM_NAME(param_name);
    }
  }

  return PI_ERROR_INVALID_KERNEL;
}

pi_result piKernelGetSubGroupInfo(pi_kernel, pi_device,
                                  pi_kernel_sub_group_info, size_t,
                                  const void *, size_t, void *, size_t *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piKernelRetain(pi_kernel) { DIE_NO_IMPLEMENTATION; }

pi_result piKernelRelease(pi_kernel kernel) {
  delete kernel;
  return PI_SUCCESS;
}

pi_result piEventCreate(pi_context, pi_event *) { DIE_NO_IMPLEMENTATION; }

pi_result piEventGetInfo(pi_event, pi_event_info, size_t, void *, size_t *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piEventGetProfilingInfo(pi_event Event, pi_profiling_info ParamName,
                                  size_t ParamValueSize, void *ParamValue,
                                  size_t *ParamValueSizeRet) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piEventsWait(pi_uint32 NumEvents, const pi_event *EventList) {
  // Todo: currently we do everything synchronously so this is a no-op
  return PI_SUCCESS;
}

pi_result piEventSetCallback(pi_event, pi_int32,
                             void (*)(pi_event, pi_int32, void *), void *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piEventSetStatus(pi_event, pi_int32) { DIE_NO_IMPLEMENTATION; }

pi_result piEventRetain(pi_event Event) { DIE_NO_IMPLEMENTATION; }

pi_result piEventRelease(pi_event Event) { DIE_NO_IMPLEMENTATION; }

pi_result piextEventGetNativeHandle(pi_event, pi_native_handle *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextEventCreateWithNativeHandle(pi_native_handle, pi_context, bool,
                                           pi_event *) {
  DIE_NO_IMPLEMENTATION;
}
pi_result piSamplerCreate(pi_context, const pi_sampler_properties *,
                          pi_sampler *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piSamplerGetInfo(pi_sampler, pi_sampler_info, size_t, void *,
                           size_t *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piSamplerRetain(pi_sampler) { DIE_NO_IMPLEMENTATION; }

pi_result piSamplerRelease(pi_sampler) { DIE_NO_IMPLEMENTATION; }

pi_result piEnqueueEventsWait(pi_queue, pi_uint32, const pi_event *,
                              pi_event *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piEnqueueEventsWaitWithBarrier(pi_queue, pi_uint32, const pi_event *,
                                         pi_event *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piEnqueueMemBufferRead(pi_queue Queue, pi_mem Src,
                                 pi_bool BlockingRead, size_t Offset,
                                 size_t Size, void *Dst,
                                 pi_uint32 NumEventsInWaitList,
                                 const pi_event *EventWaitList,
                                 pi_event *Event) {
  // TODO: is it ok to have this as no-op?
  return PI_SUCCESS;
}

pi_result piEnqueueMemBufferReadRect(pi_queue, pi_mem, pi_bool,
                                     pi_buff_rect_offset, pi_buff_rect_offset,
                                     pi_buff_rect_region, size_t, size_t,
                                     size_t, size_t, void *, pi_uint32,
                                     const pi_event *, pi_event *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piEnqueueMemBufferWrite(pi_queue, pi_mem, pi_bool, size_t, size_t,
                                  const void *, pi_uint32, const pi_event *,
                                  pi_event *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piEnqueueMemBufferWriteRect(pi_queue, pi_mem, pi_bool,
                                      pi_buff_rect_offset, pi_buff_rect_offset,
                                      pi_buff_rect_region, size_t, size_t,
                                      size_t, size_t, const void *, pi_uint32,
                                      const pi_event *, pi_event *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piEnqueueMemBufferCopy(pi_queue, pi_mem, pi_mem, size_t, size_t,
                                 size_t, pi_uint32, const pi_event *,
                                 pi_event *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piEnqueueMemBufferCopyRect(pi_queue, pi_mem, pi_mem,
                                     pi_buff_rect_offset, pi_buff_rect_offset,
                                     pi_buff_rect_region, size_t, size_t,
                                     size_t, size_t, pi_uint32,
                                     const pi_event *, pi_event *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piEnqueueMemBufferFill(pi_queue, pi_mem buffer, const void *pattern,
                                 size_t pattern_size, size_t offset,
                                 size_t size, pi_uint32, const pi_event *,
                                 pi_event *) {
  // Todo: error checking
  // Todo: handle async
  void *startingPtr = buffer->_mem + offset;
  unsigned steps = size / pattern_size;
  for (unsigned i = 0; i < steps; i++) {
    memcpy(static_cast<int8_t *>(startingPtr) + i * pattern_size, pattern,
           pattern_size);
  }
  return PI_SUCCESS;
}

pi_result piEnqueueMemBufferMap(pi_queue, pi_mem buffer, pi_bool, pi_map_flags,
                                size_t offset, size_t size, pi_uint32,
                                const pi_event *, pi_event *, void **ret_map) {
  // Todo: add proper error checking
  *ret_map = buffer->_mem + offset;
  return PI_SUCCESS;
}

pi_result piEnqueueMemUnmap(pi_queue, pi_mem mem_obj, void *mapped_ptr,
                            pi_uint32, const pi_event *, pi_event *) {
  // Todo: no-op?
  return PI_SUCCESS;
}

pi_result piMemImageGetInfo(pi_mem, pi_image_info, size_t, void *, size_t *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piEnqueueMemImageRead(pi_queue CommandQueue, pi_mem Image,
                                pi_bool BlockingRead, pi_image_offset Origin,
                                pi_image_region Region, size_t RowPitch,
                                size_t SlicePitch, void *Ptr,
                                pi_uint32 NumEventsInWaitList,
                                const pi_event *EventWaitList,
                                pi_event *Event) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piEnqueueMemImageWrite(pi_queue, pi_mem, pi_bool, pi_image_offset,
                                 pi_image_region, size_t, size_t, const void *,
                                 pi_uint32, const pi_event *, pi_event *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piEnqueueMemImageCopy(pi_queue, pi_mem, pi_mem, pi_image_offset,
                                pi_image_offset, pi_image_region, pi_uint32,
                                const pi_event *, pi_event *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piEnqueueMemImageFill(pi_queue, pi_mem, const void *, const size_t *,
                                const size_t *, pi_uint32, const pi_event *,
                                pi_event *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piMemBufferPartition(pi_mem, pi_mem_flags, pi_buffer_create_type,
                               void *, pi_mem *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result
piEnqueueKernelLaunch(pi_queue Queue, pi_kernel Kernel, pi_uint32 WorkDim,
                      const size_t *GlobalWorkOffset,
                      const size_t *GlobalWorkSize, const size_t *LocalWorkSize,
                      pi_uint32 NumEventsInWaitList,
                      const pi_event *EventWaitList, pi_event *Event) {
  // TODO: add proper error checking
  // TODO: add proper event dep management
  sycl::detail::NDRDescT ndr =
      getNDRDesc(WorkDim, GlobalWorkOffset, GlobalWorkSize, LocalWorkSize);
  __nativecpu_state state(ndr.GlobalSize[0], ndr.GlobalSize[1],
                          ndr.GlobalSize[2], ndr.LocalSize[0], ndr.LocalSize[1],
                          ndr.LocalSize[2], ndr.GlobalOffset[0],
                          ndr.GlobalOffset[1], ndr.GlobalOffset[2]);
  auto numWG0 = ndr.GlobalSize[0] / ndr.LocalSize[0];
  auto numWG1 = ndr.GlobalSize[1] / ndr.LocalSize[1];
  auto numWG2 = ndr.GlobalSize[2] / ndr.LocalSize[2];
  for (unsigned g2 = 0; g2 < numWG2; g2++) {
    for (unsigned g1 = 0; g1 < numWG1; g1++) {
      for (unsigned g0 = 0; g0 < numWG0; g0++) {
        for (unsigned local2 = 0; local2 < ndr.LocalSize[2]; local2++) {
          for (unsigned local1 = 0; local1 < ndr.LocalSize[1]; local1++) {
            for (unsigned local0 = 0; local0 < ndr.LocalSize[0]; local0++) {
              state.update(g0, g1, g2, local0, local1, local2);
              Kernel->_subhandler(Kernel->_args.data(), &state);
            }
          }
        }
      }
    }
  }
  // Todo: we should avoid calling clear here by avoiding using push_back
  // in setKernelArgs.
  Kernel->_args.clear();
  return PI_SUCCESS;
}

pi_result piextKernelCreateWithNativeHandle(pi_native_handle, pi_context,
                                            pi_program, bool, pi_kernel *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextKernelGetNativeHandle(pi_kernel, pi_native_handle *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piEnqueueNativeKernel(pi_queue, void (*)(void *), void *, size_t,
                                pi_uint32, const pi_mem *, const void **,
                                pi_uint32, const pi_event *, pi_event *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextGetDeviceFunctionPointer(pi_device, pi_program, const char *,
                                        pi_uint64 *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextUSMHostAlloc(void **result_ptr, pi_context,
                            pi_usm_mem_properties *, size_t size, pi_uint32) {
  // Todo: check properties and alignment.
  // Todo: error checking.
  *result_ptr = malloc(size);
  return PI_SUCCESS;
}

pi_result piextUSMDeviceAlloc(void **ResultPtr, pi_context, pi_device,
                              pi_usm_mem_properties *, size_t Size, pi_uint32) {
  // Todo: check properties and alignment.
  // Todo: error checking.
  *ResultPtr = malloc(Size);
  return PI_SUCCESS;
}

pi_result piextUSMSharedAlloc(void **ResultPtr, pi_context Context,
                              pi_device Device,
                              pi_usm_mem_properties *Properties, size_t Size,
                              pi_uint32 Alignment) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextUSMFree(pi_context, void *Ptr) {
  // Todo: error checking
  free(Ptr);
  return PI_SUCCESS;
}

pi_result piextKernelSetArgPointer(pi_kernel kernel, pi_uint32, size_t,
                                   const void *ptr) {
  // Todo: error checking
  auto PtrToPtr = reinterpret_cast<const intptr_t *>(ptr);
  auto DerefPtr = reinterpret_cast<void *>(*PtrToPtr);
  kernel->_args.push_back(DerefPtr);
  return PI_SUCCESS;
}

pi_result piextUSMEnqueueMemset(pi_queue, void *ptr, pi_int32 value,
                                size_t count, pi_uint32, const pi_event *,
                                pi_event *) {
  // Todo: event dependency
  // Todo: error checking.
  memset(ptr, value, count);
  return PI_SUCCESS;
}

pi_result piextUSMEnqueueMemcpy(pi_queue, pi_bool, void *dest, const void *src,
                                size_t len, pi_uint32, const pi_event *,
                                pi_event *) {
  // Todo: event dependency
  // Todo: error checking
  memcpy(dest, src, len);
  return PI_SUCCESS;
}

pi_result piextUSMEnqueueMemAdvise(pi_queue, const void *, size_t,
                                   pi_mem_advice, pi_event *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextUSMGetMemAllocInfo(pi_context, const void *, pi_mem_alloc_info,
                                  size_t, void *, size_t *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piKernelSetExecInfo(pi_kernel, pi_kernel_exec_info, size_t,
                              const void *) {
  return PI_SUCCESS;
}

pi_result piextProgramSetSpecializationConstant(pi_program, pi_uint32, size_t,
                                                const void *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextDeviceSelectBinary(pi_device, pi_device_binary *,
                                  pi_uint32 RawImgSize, pi_uint32 *ImgInd) {
  CONTINUE_NO_IMPLEMENTATION;
}

pi_result piextUSMEnqueuePrefetch(pi_queue, const void *, size_t,
                                  pi_usm_migration_flags, pi_uint32,
                                  const pi_event *, pi_event *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextPluginGetOpaqueData(void *, void **OpaqueDataReturn) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextQueueCreate(pi_context context, pi_device device,
                           pi_queue_properties *properties, pi_queue *queue) {
  CONTINUE_NO_IMPLEMENTATION;
}
pi_result piextMemImageCreateWithNativeHandle(
    pi_native_handle nativeHandle, pi_context context, bool ownNativeHandle,
    const pi_image_format *ImageFormat, const pi_image_desc *ImageDesc,
    pi_mem *img) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextEnqueueWriteHostPipe(pi_queue queue, pi_program program,
                                    const char *pipe_symbol, pi_bool blocking,
                                    void *ptr, size_t size,
                                    pi_uint32 num_events_in_waitlist,
                                    const pi_event *events_waitlist,
                                    pi_event *event) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextEnqueueReadHostPipe(pi_queue queue, pi_program program,
                                   const char *pipe_symbol, pi_bool blocking,
                                   void *ptr, size_t size,
                                   pi_uint32 num_events_in_waitlist,
                                   const pi_event *events_waitlist,
                                   pi_event *event) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piPluginGetLastError(char **message) { DIE_NO_IMPLEMENTATION; }

pi_result piextUSMEnqueueMemcpy2D(pi_queue queue, pi_bool blocking,
                                  void *dst_ptr, size_t dst_pitch,
                                  const void *src_ptr, size_t src_pitch,
                                  size_t width, size_t height,
                                  pi_uint32 num_events_in_waitlist,
                                  const pi_event *events_waitlist,
                                  pi_event *event) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextEnqueueDeviceGlobalVariableWrite(
    pi_queue queue, pi_program program, const char *name,
    pi_bool blocking_write, size_t count, size_t offset, const void *src,
    pi_uint32 num_events_in_wait_list, const pi_event *event_wait_list,
    pi_event *event) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextEnqueueDeviceGlobalVariableRead(
    pi_queue queue, pi_program program, const char *name, pi_bool blocking_read,
    size_t count, size_t offset, void *dst, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piPluginGetBackendOption(pi_platform platform,
                                   const char *frontend_option,
                                   const char **backend_option) {
  CONTINUE_NO_IMPLEMENTATION;
}

pi_result piextUSMEnqueueMemset2D(pi_queue queue, void *ptr, size_t pitch,
                                  int value, size_t width, size_t height,
                                  pi_uint32 num_events_in_waitlist,
                                  const pi_event *events_waitlist,
                                  pi_event *event) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piGetDeviceAndHostTimer(pi_device Device, uint64_t *DeviceTime,
                                  uint64_t *HostTime) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextQueueGetNativeHandle2(pi_queue queue,
                                     pi_native_handle *nativeHandle,
                                     int32_t *nativeHandleDesc) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextQueueCreate2(pi_context context, pi_device device,
                            pi_queue_properties *properties, pi_queue *queue) {
  // Todo: is it fine as a no-op?
  return PI_SUCCESS;
}

pi_result piextQueueCreateWithNativeHandle2(
    pi_native_handle nativeHandle, int32_t nativeHandleDesc, pi_context context,
    pi_device device, bool pluginOwnsNativeHandle,
    pi_queue_properties *Properties, pi_queue *queue) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextUSMEnqueueFill2D(pi_queue queue, void *ptr, size_t pitch,
                                size_t pattern_size, const void *pattern,
                                size_t width, size_t height,
                                pi_uint32 num_events_in_waitlist,
                                const pi_event *events_waitlist,
                                pi_event *event) {
  DIE_NO_IMPLEMENTATION;
}
pi_result piextCommandBufferCreate(pi_context, pi_device,
                                   const pi_ext_command_buffer_desc *,
                                   pi_ext_command_buffer *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextCommandBufferRetain(pi_ext_command_buffer) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextCommandBufferRelease(pi_ext_command_buffer) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextCommandBufferFinalize(pi_ext_command_buffer) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextCommandBufferNDRangeKernel(pi_ext_command_buffer, pi_kernel,
                                          pi_uint32, const size_t *,
                                          const size_t *, const size_t *,
                                          pi_uint32, const pi_ext_sync_point *,
                                          pi_ext_sync_point *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextCommandBufferMemcpyUSM(pi_ext_command_buffer, void *,
                                      const void *, size_t, pi_uint32,
                                      const pi_ext_sync_point *,
                                      pi_ext_sync_point *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextCommandBufferMemBufferCopy(pi_ext_command_buffer, pi_mem, pi_mem,
                                          size_t, size_t, size_t, pi_uint32,
                                          const pi_ext_sync_point *,
                                          pi_ext_sync_point *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextCommandBufferMemBufferCopyRect(
    pi_ext_command_buffer, pi_mem, pi_mem, pi_buff_rect_offset,
    pi_buff_rect_offset, pi_buff_rect_region, size_t, size_t, size_t, size_t,
    pi_uint32, const pi_ext_sync_point *, pi_ext_sync_point *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextCommandBufferMemBufferRead(pi_ext_command_buffer, pi_mem, size_t,
                                          size_t, void *, pi_uint32,
                                          const pi_ext_sync_point *,
                                          pi_ext_sync_point *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextCommandBufferMemBufferReadRect(
    pi_ext_command_buffer, pi_mem, pi_buff_rect_offset, pi_buff_rect_offset,
    pi_buff_rect_region, size_t, size_t, size_t, size_t, void *, pi_uint32,
    const pi_ext_sync_point *, pi_ext_sync_point *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextCommandBufferMemBufferWrite(pi_ext_command_buffer, pi_mem,
                                           size_t, size_t, const void *,
                                           pi_uint32, const pi_ext_sync_point *,
                                           pi_ext_sync_point *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextCommandBufferMemBufferWriteRect(
    pi_ext_command_buffer, pi_mem, pi_buff_rect_offset, pi_buff_rect_offset,
    pi_buff_rect_region, size_t, size_t, size_t, size_t, const void *,
    pi_uint32, const pi_ext_sync_point *, pi_ext_sync_point *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextEnqueueCommandBuffer(pi_ext_command_buffer, pi_queue, pi_uint32,
                                    const pi_event *, pi_event *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextEnablePeerAccess(pi_device, pi_device) { DIE_NO_IMPLEMENTATION; }

pi_result piextDisablePeerAccess(pi_device, pi_device) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextPeerAccessGetInfo(pi_device, pi_device, pi_peer_attr, size_t,
                                 void *, size_t *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piTearDown(void *) {
  // Todo: is it fine as a no-op?
  return PI_SUCCESS;
}

pi_result piPluginInit(pi_plugin *PluginInit) {
#define _PI_API(api)                                                           \
  (PluginInit->PiFunctionTable).api = (decltype(&::api))(&api);
#include <sycl/detail/pi.def>

  return PI_SUCCESS;
}
}
