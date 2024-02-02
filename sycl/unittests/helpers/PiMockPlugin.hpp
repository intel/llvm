//==---------- PiMockPlugin.hpp --- Mock unit testing PI plugin ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A simple implementation of a PI plugin to be used for device-independent
// mock unit-testing.
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/pi.hpp>

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstring>

// Helpers for dummy handles

struct DummyHandleT {
  DummyHandleT(size_t DataSize = 0)
      : MStorage(DataSize), MData(MStorage.data()) {}
  DummyHandleT(unsigned char *Data) : MData(Data) {}
  std::atomic<size_t> MRefCounter = 1;
  std::vector<unsigned char> MStorage;
  unsigned char *MData = nullptr;
};

using DummyHandlePtrT = DummyHandleT *;

// Allocates a dummy handle of type T with support of reference counting.
// Takes optional 'Size' parameter which can be used to allocate additional
// memory. The handle has to be deallocated using 'releaseDummyHandle'.
template <class T> inline T createDummyHandle(size_t Size = 0) {
  DummyHandlePtrT DummyHandlePtr = new DummyHandleT(Size);
  return reinterpret_cast<T>(DummyHandlePtr);
}

// Allocates a dummy handle of type T with support of reference counting
// and associates it with the provided Data.
template <class T> inline T createDummyHandleWithData(unsigned char *Data) {
  DummyHandlePtrT DummyHandlePtr = new DummyHandleT(Data);
  return reinterpret_cast<T>(DummyHandlePtr);
}

// Decrement reference counter for the handle and deallocates it if the
// reference counter becomes zero
template <class T> inline void releaseDummyHandle(T Handle) {
  auto DummyHandlePtr = reinterpret_cast<DummyHandlePtrT>(Handle);
  const size_t NewValue = --DummyHandlePtr->MRefCounter;
  if (NewValue == 0)
    delete DummyHandlePtr;
}

// Increment reference counter for the handle
template <class T> inline void retainDummyHandle(T Handle) {
  auto DummyHandlePtr = reinterpret_cast<DummyHandlePtrT>(Handle);
  ++DummyHandlePtr->MRefCounter;
}

//
// Platform
//
inline pi_result mock_piPlatformsGet(pi_uint32 num_entries,
                                     pi_platform *platforms,
                                     pi_uint32 *num_platforms) {
  if (num_platforms)
    *num_platforms = 1;

  if (platforms && num_entries > 0)
    platforms[0] = reinterpret_cast<pi_platform>(1);

  return PI_SUCCESS;
}

inline pi_result mock_piPlatformGetInfo(pi_platform platform,
                                        pi_platform_info param_name,
                                        size_t param_value_size,
                                        void *param_value,
                                        size_t *param_value_size_ret) {
  constexpr char MockPlatformName[] = "Mock platform";
  constexpr char MockSupportedExtensions[] =
      "cl_khr_il_program cl_khr_subgroups cl_intel_subgroups "
      "cl_intel_subgroups_short cl_intel_required_subgroup_size ";
  switch (param_name) {
  case PI_PLATFORM_INFO_NAME: {
    if (param_value) {
      assert(param_value_size == sizeof(MockPlatformName));
      std::memcpy(param_value, MockPlatformName, sizeof(MockPlatformName));
    }
    if (param_value_size_ret)
      *param_value_size_ret = sizeof(MockPlatformName);
    return PI_SUCCESS;
  }
  case PI_PLATFORM_INFO_EXTENSIONS: {
    if (param_value) {
      assert(param_value_size == sizeof(MockSupportedExtensions));
      std::memcpy(param_value, MockSupportedExtensions,
                  sizeof(MockSupportedExtensions));
    }
    if (param_value_size_ret)
      *param_value_size_ret = sizeof(MockSupportedExtensions);
    return PI_SUCCESS;
  }
  case PI_EXT_PLATFORM_INFO_BACKEND: {
    constexpr auto MockPlatformBackend = PI_EXT_PLATFORM_BACKEND_OPENCL;
    if (param_value) {
      std::memcpy(param_value, &MockPlatformBackend,
                  sizeof(MockPlatformBackend));
    }
    if (param_value_size_ret)
      *param_value_size_ret = sizeof(MockPlatformBackend);
    return PI_SUCCESS;
  }
  default: {
    constexpr const char FallbackValue[] = "str";
    constexpr size_t FallbackValueSize = sizeof(FallbackValue);
    if (param_value_size_ret)
      *param_value_size_ret = FallbackValueSize;

    if (param_value && param_value_size >= FallbackValueSize)
      std::memcpy(param_value, FallbackValue, FallbackValueSize);

    return PI_SUCCESS;
  }
  }
}

inline pi_result
mock_piextPlatformGetNativeHandle(pi_platform platform,
                                  pi_native_handle *nativeHandle) {
  *nativeHandle = reinterpret_cast<pi_native_handle>(platform);
  return PI_SUCCESS;
}

inline pi_result
mock_piextPlatformCreateWithNativeHandle(pi_native_handle nativeHandle,
                                         pi_platform *platform) {
  *platform = reinterpret_cast<pi_platform>(nativeHandle);
  retainDummyHandle(*platform);
  return PI_SUCCESS;
}

inline pi_result mock_piDevicesGet(pi_platform platform,
                                   pi_device_type device_type,
                                   pi_uint32 num_entries, pi_device *devices,
                                   pi_uint32 *num_devices) {
  if (num_devices)
    *num_devices = 1;

  if (devices && num_entries > 0)
    devices[0] = reinterpret_cast<pi_device>(1);

  return PI_SUCCESS;
}

inline pi_result mock_piDeviceGetInfo(pi_device device,
                                      pi_device_info param_name,
                                      size_t param_value_size,
                                      void *param_value,
                                      size_t *param_value_size_ret) {
  constexpr char MockDeviceName[] = "Mock device";
  constexpr char MockSupportedExtensions[] =
      "cl_khr_fp64 cl_khr_fp16 cl_khr_il_program ur_exp_command_buffer";
  switch (param_name) {
  case PI_DEVICE_INFO_TYPE: {
    // Act like any device is a GPU.
    // TODO: Should we mock more device types?
    if (param_value)
      *static_cast<_pi_device_type *>(param_value) = PI_DEVICE_TYPE_GPU;
    if (param_value_size_ret)
      *param_value_size_ret = sizeof(PI_DEVICE_TYPE_GPU);
    return PI_SUCCESS;
  }
  case PI_DEVICE_INFO_NAME: {
    if (param_value) {
      assert(param_value_size == sizeof(MockDeviceName));
      std::memcpy(param_value, MockDeviceName, sizeof(MockDeviceName));
    }
    if (param_value_size_ret)
      *param_value_size_ret = sizeof(MockDeviceName);
    return PI_SUCCESS;
  }
  case PI_DEVICE_INFO_PARENT_DEVICE: {
    if (param_value)
      *static_cast<pi_device *>(param_value) = nullptr;
    if (param_value_size_ret)
      *param_value_size_ret = sizeof(pi_device *);
    return PI_SUCCESS;
  }
  case PI_DEVICE_INFO_EXTENSIONS: {
    if (param_value) {
      assert(param_value_size >= sizeof(MockSupportedExtensions));
      std::memcpy(param_value, MockSupportedExtensions,
                  sizeof(MockSupportedExtensions));
    }
    if (param_value_size_ret)
      *param_value_size_ret = sizeof(MockSupportedExtensions);
    return PI_SUCCESS;
  }
  case PI_DEVICE_INFO_USM_HOST_SUPPORT:
  case PI_DEVICE_INFO_USM_DEVICE_SUPPORT:
  case PI_DEVICE_INFO_HOST_UNIFIED_MEMORY:
  case PI_DEVICE_INFO_AVAILABLE:
  case PI_DEVICE_INFO_LINKER_AVAILABLE:
  case PI_DEVICE_INFO_COMPILER_AVAILABLE: {
    if (param_value)
      *static_cast<pi_bool *>(param_value) = PI_TRUE;
    if (param_value_size_ret)
      *param_value_size_ret = sizeof(PI_TRUE);
    return PI_SUCCESS;
  }
  // This mock GPU device has no sub-devices
  case PI_DEVICE_INFO_PARTITION_PROPERTIES: {
    if (param_value_size_ret) {
      *param_value_size_ret = 0;
    }
    return PI_SUCCESS;
  }
  case PI_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN: {
    assert(param_value_size == sizeof(pi_device_affinity_domain));
    if (param_value) {
      *static_cast<pi_device_affinity_domain *>(param_value) = 0;
    }
    return PI_SUCCESS;
  }
  case PI_DEVICE_INFO_QUEUE_PROPERTIES: {
    assert(param_value_size == sizeof(pi_queue_properties));
    if (param_value) {
      *static_cast<pi_queue_properties *>(param_value) =
          PI_QUEUE_FLAG_PROFILING_ENABLE;
    }
    return PI_SUCCESS;
  }
  default:
    return PI_SUCCESS;
  }
}

inline pi_result mock_piDeviceRetain(pi_device device) { return PI_SUCCESS; }

inline pi_result mock_piDeviceRelease(pi_device device) { return PI_SUCCESS; }

inline pi_result mock_piDevicePartition(
    pi_device device, const pi_device_partition_property *properties,
    pi_uint32 num_devices, pi_device *out_devices, pi_uint32 *out_num_devices) {
  return PI_SUCCESS;
}

inline pi_result
mock_piextDeviceGetNativeHandle(pi_device device,
                                pi_native_handle *nativeHandle) {
  *nativeHandle = reinterpret_cast<pi_native_handle>(device);
  return PI_SUCCESS;
}

inline pi_result mock_piextDeviceCreateWithNativeHandle(
    pi_native_handle nativeHandle, pi_platform platform, pi_device *device) {
  *device = reinterpret_cast<pi_device>(nativeHandle);
  retainDummyHandle(*device);
  return PI_SUCCESS;
}

inline pi_result mock_piextDeviceSelectBinary(pi_device device,
                                              pi_device_binary *binaries,
                                              pi_uint32 num_binaries,
                                              pi_uint32 *selected_binary_ind) {
  *selected_binary_ind = 0;
  return PI_SUCCESS;
}

inline pi_result
mock_piextGetDeviceFunctionPointer(pi_device device, pi_program program,
                                   const char *function_name,
                                   pi_uint64 *function_pointer_ret) {
  return PI_SUCCESS;
}

//
// Context
//
inline pi_result mock_piContextCreate(
    const pi_context_properties *properties, pi_uint32 num_devices,
    const pi_device *devices,
    void (*pfn_notify)(const char *errinfo, const void *private_info, size_t cb,
                       void *user_data),
    void *user_data, pi_context *ret_context) {
  *ret_context = createDummyHandle<pi_context>();
  return PI_SUCCESS;
}

inline pi_result mock_piContextGetInfo(pi_context context,
                                       pi_context_info param_name,
                                       size_t param_value_size,
                                       void *param_value,
                                       size_t *param_value_size_ret) {
  switch (param_name) {
  case PI_CONTEXT_INFO_NUM_DEVICES: {
    if (param_value)
      *static_cast<pi_uint32 *>(param_value) = 1;
    if (param_value_size_ret)
      *param_value_size_ret = sizeof(pi_uint32);
    return PI_SUCCESS;
  }
  default:
    return PI_SUCCESS;
  }
}

inline pi_result mock_piContextRetain(pi_context context) {
  retainDummyHandle(context);
  return PI_SUCCESS;
}

inline pi_result mock_piContextRelease(pi_context context) {
  releaseDummyHandle(context);
  return PI_SUCCESS;
}

inline pi_result mock_piextContextSetExtendedDeleter(
    pi_context context, pi_context_extended_deleter func, void *user_data) {
  return PI_SUCCESS;
}

inline pi_result
mock_piextContextGetNativeHandle(pi_context context,
                                 pi_native_handle *nativeHandle) {
  *nativeHandle = reinterpret_cast<pi_native_handle>(context);
  return PI_SUCCESS;
}

inline pi_result mock_piextContextCreateWithNativeHandle(
    pi_native_handle nativeHandle, pi_uint32 numDevices,
    const pi_device *devices, bool pluginOwnsNativeHandle,
    pi_context *context) {
  *context = reinterpret_cast<pi_context>(nativeHandle);
  retainDummyHandle(*context);
  return PI_SUCCESS;
}

//
// Queue
//
inline pi_result mock_piQueueCreate(pi_context context, pi_device device,
                                    pi_queue_properties properties,
                                    pi_queue *queue) {
  *queue = createDummyHandle<pi_queue>();
  return PI_SUCCESS;
}
inline pi_result mock_piextQueueCreate(pi_context context, pi_device device,
                                       pi_queue_properties *properties,
                                       pi_queue *queue) {
  *queue = createDummyHandle<pi_queue>();
  return PI_SUCCESS;
}

inline pi_result mock_piQueueGetInfo(pi_queue command_queue,
                                     pi_queue_info param_name,
                                     size_t param_value_size, void *param_value,
                                     size_t *param_value_size_ret) {
  switch (param_name) {
  case PI_QUEUE_INFO_DEVICE: {
    if (param_value)
      *static_cast<pi_device *>(param_value) = reinterpret_cast<pi_device>(1);
    if (param_value_size_ret)
      *param_value_size_ret = sizeof(pi_device);
    return PI_SUCCESS;
  }
  default:
    return PI_SUCCESS;
  }
}

inline pi_result mock_piQueueRetain(pi_queue command_queue) {
  retainDummyHandle(command_queue);
  return PI_SUCCESS;
}

inline pi_result mock_piQueueRelease(pi_queue command_queue) {
  releaseDummyHandle(command_queue);
  return PI_SUCCESS;
}

inline pi_result mock_piQueueFinish(pi_queue command_queue) {
  return PI_SUCCESS;
}

inline pi_result mock_piQueueFlush(pi_queue command_queue) {
  return PI_SUCCESS;
}

inline pi_result mock_piextQueueGetNativeHandle(pi_queue queue,
                                                pi_native_handle *nativeHandle,
                                                int32_t *nativeHandleDesc) {
  *nativeHandle = reinterpret_cast<pi_native_handle>(queue);
  return PI_SUCCESS;
}

inline pi_result mock_piextQueueCreateWithNativeHandle(
    pi_native_handle nativeHandle, int32_t nativeHandleDesc, pi_context context,
    pi_device device, bool pluginOwnsNativeHandle,
    pi_queue_properties *Properties, pi_queue *queue) {
  *queue = reinterpret_cast<pi_queue>(nativeHandle);
  retainDummyHandle(*queue);
  return PI_SUCCESS;
}

//
// Memory
//
inline pi_result
mock_piMemBufferCreate(pi_context context, pi_mem_flags flags, size_t size,
                       void *host_ptr, pi_mem *ret_mem,
                       const pi_mem_properties *properties = nullptr) {
  if (host_ptr && flags & PI_MEM_FLAGS_HOST_PTR_USE)
    *ret_mem = createDummyHandleWithData<pi_mem>(
        reinterpret_cast<unsigned char *>(host_ptr));
  else
    *ret_mem = createDummyHandle<pi_mem>(size);
  return PI_SUCCESS;
}

inline pi_result mock_piMemImageCreate(pi_context context, pi_mem_flags flags,
                                       const pi_image_format *image_format,
                                       const pi_image_desc *image_desc,
                                       void *host_ptr, pi_mem *ret_mem) {
  assert(false &&
         "TODO: mock_piMemImageCreate handle allocation size correctly");
  *ret_mem = createDummyHandle<pi_mem>(/*size=*/1024 * 16);
  return PI_SUCCESS;
}

inline pi_result
mock_piextMemUnsampledImageHandleDestroy(pi_context context, pi_device device,
                                         pi_image_handle handle) {
  return PI_SUCCESS;
}

inline pi_result
mock_piextMemSampledImageHandleDestroy(pi_context context, pi_device device,
                                       pi_image_handle handle) {
  return PI_SUCCESS;
}

inline pi_result mock_piextMemImageAllocate(pi_context context,
                                            pi_device device,
                                            pi_image_format *image_format,
                                            pi_image_desc *image_desc,
                                            pi_image_mem_handle *ret_mem) {
  return PI_SUCCESS;
}

inline pi_result mock_piextMemMipmapGetLevel(pi_context context,
                                             pi_device device,
                                             pi_image_mem_handle mip_mem,
                                             unsigned int level,
                                             pi_image_mem_handle *ret_mem) {
  return PI_SUCCESS;
}

inline pi_result mock_piextMemImageFree(pi_context context, pi_device device,
                                        pi_image_mem_handle memory_handle) {
  return PI_SUCCESS;
}

inline pi_result mock_piextMemMipmapFree(pi_context context, pi_device device,
                                         pi_image_mem_handle memory_handle) {
  return PI_SUCCESS;
}

inline pi_result mock_piextMemUnsampledImageCreate(
    pi_context context, pi_device device, pi_image_mem_handle img_mem,
    pi_image_format *image_format, pi_image_desc *desc, pi_mem *ret_mem,
    pi_image_handle *ret_handle) {
  return PI_SUCCESS;
}

inline pi_result
mock_piextMemImportOpaqueFD(pi_context context, pi_device device, size_t size,
                            int file_descriptor,
                            pi_interop_mem_handle *ret_handle) {
  return PI_SUCCESS;
}

inline pi_result mock_piextMemMapExternalArray(pi_context context,
                                               pi_device device,
                                               pi_image_format *image_format,
                                               pi_image_desc *image_desc,
                                               pi_interop_mem_handle mem_handle,
                                               pi_image_mem_handle *ret_mem) {
  return PI_SUCCESS;
}

inline pi_result mock_piextMemReleaseInterop(pi_context context,
                                             pi_device device,
                                             pi_interop_mem_handle ext_mem) {
  return PI_SUCCESS;
}

inline pi_result mock_piextImportExternalSemaphoreOpaqueFD(
    pi_context context, pi_device device, int file_descriptor,
    pi_interop_semaphore_handle *ret_handle) {
  return PI_SUCCESS;
}

inline pi_result
mock_piextDestroyExternalSemaphore(pi_context context, pi_device device,
                                   pi_interop_semaphore_handle sem_handle) {
  return PI_SUCCESS;
}

inline pi_result mock_piextWaitExternalSemaphore(
    pi_queue command_queue, pi_interop_semaphore_handle sem_handle,
    pi_uint32 num_events_in_wait_list, const pi_event *event_wait_list,
    pi_event *event) {
  return PI_SUCCESS;
}

inline pi_result mock_piextSignalExternalSemaphore(
    pi_queue command_queue, pi_interop_semaphore_handle sem_handle,
    pi_uint32 num_events_in_wait_list, const pi_event *event_wait_list,
    pi_event *event) {
  return PI_SUCCESS;
}

inline pi_result mock_piextMemUnsampledImageCreateInterop(
    pi_context context, pi_device device, pi_image_format *image_format,
    pi_image_desc *desc, pi_interop_mem_handle ext_mem_handle,
    pi_image_handle *ret_img_handle) {
  return PI_SUCCESS;
}

inline pi_result mock_piextMemSampledImageCreateInterop(
    pi_context context, pi_device device, pi_image_format *image_format,
    pi_image_desc *desc, pi_sampler sampler,
    pi_interop_mem_handle ext_mem_handle, pi_image_handle *ret_img_handle) {
  return PI_SUCCESS;
}

inline pi_result mock_piextMemSampledImageCreate(
    pi_context context, pi_device device, pi_image_mem_handle img_mem,
    pi_image_format *image_format, pi_image_desc *desc, pi_sampler sampler,
    pi_mem *ret_mem, pi_image_handle *ret_handle) {
  return PI_SUCCESS;
}

inline pi_result mock_piextBindlessImageSamplerCreate(
    pi_context context, const pi_sampler_properties *sampler_properties,
    const float minMipmapLevelClamp, const float maxMipmapLevelClamp,
    const float maxAnisotropy, pi_sampler *result_sampler) {
  *result_sampler = createDummyHandle<pi_sampler>();
  return PI_SUCCESS;
}

inline pi_result mock_piextMemImageCopy(
    pi_queue command_queue, void *dst_ptr, void *src_ptr,
    const pi_image_format *image_format, const pi_image_desc *image_desc,
    const pi_image_copy_flags flags, pi_image_offset src_offset,
    pi_image_offset dst_offset, pi_image_region copy_extent,
    pi_image_region host_extent, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event) {
  return PI_SUCCESS;
}

inline pi_result mock_piextMemImageGetInfo(const pi_image_mem_handle mem_handle,
                                           pi_image_info param_name,
                                           void *param_value,
                                           size_t *param_value_size_ret) {
  return PI_SUCCESS;
}

inline pi_result mock_piMemGetInfo(pi_mem mem, pi_mem_info param_name,
                                   size_t param_value_size, void *param_value,
                                   size_t *param_value_size_ret) {
  return PI_SUCCESS;
}

inline pi_result mock_piMemImageGetInfo(pi_mem image, pi_image_info param_name,
                                        size_t param_value_size,
                                        void *param_value,
                                        size_t *param_value_size_ret) {
  return PI_SUCCESS;
}

inline pi_result mock_piMemRetain(pi_mem mem) {
  retainDummyHandle(mem);
  return PI_SUCCESS;
}

inline pi_result mock_piMemRelease(pi_mem mem) {
  releaseDummyHandle(mem);
  return PI_SUCCESS;
}

inline pi_result
mock_piMemBufferPartition(pi_mem buffer, pi_mem_flags flags,
                          pi_buffer_create_type buffer_create_type,
                          void *buffer_create_info, pi_mem *ret_mem) {
  // Create a sub buf without memory as we will reuse parent's one
  *ret_mem = createDummyHandle<pi_mem>(/*size=*/0);

  auto parentDummyHandle = reinterpret_cast<DummyHandlePtrT>(buffer);
  auto childDummyHandle = reinterpret_cast<DummyHandlePtrT>(*ret_mem);

  auto region = reinterpret_cast<pi_buffer_region>(buffer_create_info);

  // Point the sub buf to the original buf memory
  childDummyHandle->MData = parentDummyHandle->MData + region->origin;

  return PI_SUCCESS;
}

inline pi_result mock_piextMemGetNativeHandle(pi_mem mem, pi_device dev,
                                              pi_native_handle *nativeHandle) {
  *nativeHandle = reinterpret_cast<pi_native_handle>(mem);
  return PI_SUCCESS;
}

inline pi_result
mock_piextMemCreateWithNativeHandle(pi_native_handle nativeHandle,
                                    pi_context context, bool ownNativeHandle,
                                    pi_mem *mem) {
  *mem = reinterpret_cast<pi_mem>(nativeHandle);
  retainDummyHandle(*mem);
  return PI_SUCCESS;
}

inline pi_result mock_piextMemImageCreateWithNativeHandle(
    pi_native_handle NativeHandle, pi_context Context, bool OwnNativeHandle,
    const pi_image_format *ImageFormat, const pi_image_desc *ImageDesc,
    pi_mem *RetImage) {
  *RetImage = reinterpret_cast<pi_mem>(NativeHandle);
  retainDummyHandle(*RetImage);
  return PI_SUCCESS;
}

//
// Program
//

inline pi_result mock_piProgramCreate(pi_context context, const void *il,
                                      size_t length, pi_program *res_program) {
  *res_program = createDummyHandle<pi_program>();
  return PI_SUCCESS;
}

inline pi_result mock_piProgramCreateWithBinary(
    pi_context context, pi_uint32 num_devices, const pi_device *device_list,
    const size_t *lengths, const unsigned char **binaries,
    size_t num_metadata_entries, const pi_device_binary_property *metadata,
    pi_int32 *binary_status, pi_program *ret_program) {
  *ret_program = createDummyHandle<pi_program>();
  return PI_SUCCESS;
}

inline pi_result mock_piProgramGetInfo(pi_program program,
                                       pi_program_info param_name,
                                       size_t param_value_size,
                                       void *param_value,
                                       size_t *param_value_size_ret) {

  switch (param_name) {
  case PI_PROGRAM_INFO_NUM_DEVICES: {
    if (param_value)
      *static_cast<unsigned int *>(param_value) = 1;
    if (param_value_size_ret)
      *param_value_size_ret = sizeof(size_t);
    return PI_SUCCESS;
  }
  case PI_PROGRAM_INFO_BINARY_SIZES: {
    if (param_value)
      *static_cast<size_t *>(param_value) = 1;
    if (param_value_size_ret)
      *param_value_size_ret = sizeof(size_t);
    return PI_SUCCESS;
  }
  case PI_PROGRAM_INFO_BINARIES: {
    if (param_value)
      **static_cast<unsigned char **>(param_value) = 1;
    if (param_value_size_ret)
      *param_value_size_ret = sizeof(unsigned char);
    return PI_SUCCESS;
  }
  default: {
    // TODO: Buildlog requires this but not any actual data afterwards.
    //       This should be investigated. Should this be moved to that test?
    if (param_value_size_ret)
      *param_value_size_ret = sizeof(size_t);
    return PI_SUCCESS;
  }
  }
}

inline pi_result
mock_piProgramLink(pi_context context, pi_uint32 num_devices,
                   const pi_device *device_list, const char *options,
                   pi_uint32 num_input_programs,
                   const pi_program *input_programs,
                   void (*pfn_notify)(pi_program program, void *user_data),
                   void *user_data, pi_program *ret_program) {
  *ret_program = createDummyHandle<pi_program>();
  return PI_SUCCESS;
}

inline pi_result mock_piProgramCompile(
    pi_program program, pi_uint32 num_devices, const pi_device *device_list,
    const char *options, pi_uint32 num_input_headers,
    const pi_program *input_headers, const char **header_include_names,
    void (*pfn_notify)(pi_program program, void *user_data), void *user_data) {
  return PI_SUCCESS;
}

inline pi_result
mock_piProgramBuild(pi_program program, pi_uint32 num_devices,
                    const pi_device *device_list, const char *options,
                    void (*pfn_notify)(pi_program program, void *user_data),
                    void *user_data) {
  return PI_SUCCESS;
}

inline pi_result mock_piProgramGetBuildInfo(
    pi_program program, pi_device device, _pi_program_build_info param_name,
    size_t param_value_size, void *param_value, size_t *param_value_size_ret) {
  return PI_SUCCESS;
}

inline pi_result mock_piProgramRetain(pi_program program) {
  retainDummyHandle(program);
  return PI_SUCCESS;
}

inline pi_result mock_piProgramRelease(pi_program program) {
  releaseDummyHandle(program);
  return PI_SUCCESS;
}

inline pi_result
mock_piextProgramSetSpecializationConstant(pi_program prog, pi_uint32 spec_id,
                                           size_t spec_size,
                                           const void *spec_value) {
  return PI_SUCCESS;
}

inline pi_result
mock_piextProgramGetNativeHandle(pi_program program,
                                 pi_native_handle *nativeHandle) {
  *nativeHandle = reinterpret_cast<pi_native_handle>(program);
  return PI_SUCCESS;
}

inline pi_result mock_piextProgramCreateWithNativeHandle(
    pi_native_handle nativeHandle, pi_context context,
    bool pluginOwnsNativeHandle, pi_program *program) {
  *program = reinterpret_cast<pi_program>(nativeHandle);
  retainDummyHandle(*program);
  return PI_SUCCESS;
}

//
// Kernel
//

inline pi_result mock_piKernelCreate(pi_program program,
                                     const char *kernel_name,
                                     pi_kernel *ret_kernel) {
  *ret_kernel = createDummyHandle<pi_kernel>();
  return PI_SUCCESS;
}

inline pi_result mock_piKernelSetArg(pi_kernel kernel, pi_uint32 arg_index,
                                     size_t arg_size, const void *arg_value) {
  return PI_SUCCESS;
}

inline pi_result mock_piKernelGetInfo(pi_kernel kernel,
                                      pi_kernel_info param_name,
                                      size_t param_value_size,
                                      void *param_value,
                                      size_t *param_value_size_ret) {
  return PI_SUCCESS;
}

inline pi_result mock_piKernelGetGroupInfo(pi_kernel kernel, pi_device device,
                                           pi_kernel_group_info param_name,
                                           size_t param_value_size,
                                           void *param_value,
                                           size_t *param_value_size_ret) {
  switch (param_name) {
  case PI_KERNEL_GROUP_INFO_WORK_GROUP_SIZE: {
    if (param_value) {
      auto RealVal = reinterpret_cast<size_t *>(param_value);
      RealVal[0] = 0;
      RealVal[1] = 0;
      RealVal[2] = 0;
    }
    if (param_value_size_ret)
      *param_value_size_ret = 3 * sizeof(size_t);
    return PI_SUCCESS;
  }
  default: {
    return PI_SUCCESS;
  }
  }
}

inline pi_result mock_piKernelGetSubGroupInfo(
    pi_kernel kernel, pi_device device, pi_kernel_sub_group_info param_name,
    size_t input_value_size, const void *input_value, size_t param_value_size,
    void *param_value, size_t *param_value_size_ret) {
  return PI_SUCCESS;
}

inline pi_result mock_piKernelRetain(pi_kernel kernel) {
  retainDummyHandle(kernel);
  return PI_SUCCESS;
}

inline pi_result mock_piKernelRelease(pi_kernel kernel) {
  releaseDummyHandle(kernel);
  return PI_SUCCESS;
}

inline pi_result mock_piextKernelSetArgPointer(pi_kernel kernel,
                                               pi_uint32 arg_index,
                                               size_t arg_size,
                                               const void *arg_value) {
  return PI_SUCCESS;
}

inline pi_result mock_piKernelSetExecInfo(pi_kernel kernel,
                                          pi_kernel_exec_info value_name,
                                          size_t param_value_size,
                                          const void *param_value) {
  return PI_SUCCESS;
}

inline pi_result mock_piextKernelCreateWithNativeHandle(
    pi_native_handle nativeHandle, pi_context context, pi_program program,
    bool pluginOwnsNativeHandle, pi_kernel *kernel) {

  *kernel = reinterpret_cast<pi_kernel>(nativeHandle);
  retainDummyHandle(*kernel);
  return PI_SUCCESS;
}

inline pi_result
mock_piextKernelGetNativeHandle(pi_kernel kernel,
                                pi_native_handle *nativeHandle) {
  *nativeHandle = reinterpret_cast<pi_native_handle>(kernel);
  return PI_SUCCESS;
}

//
// Events
//
inline pi_result mock_piEventCreate(pi_context context, pi_event *ret_event) {
  *ret_event = createDummyHandle<pi_event>();
  return PI_SUCCESS;
}

inline pi_result mock_piEventGetInfo(pi_event event, pi_event_info param_name,
                                     size_t param_value_size, void *param_value,
                                     size_t *param_value_size_ret) {
  switch (param_name) {
  case PI_EVENT_INFO_COMMAND_EXECUTION_STATUS: {
    if (param_value)
      *static_cast<pi_event_status *>(param_value) = PI_EVENT_SUBMITTED;
    if (param_value_size_ret)
      *param_value_size_ret = sizeof(pi_event_status);
    return PI_SUCCESS;
  }
  default: {
    return PI_SUCCESS;
  }
  }
}

inline pi_result mock_piEventGetProfilingInfo(pi_event event,
                                              pi_profiling_info param_name,
                                              size_t param_value_size,
                                              void *param_value,
                                              size_t *param_value_size_ret) {
  return PI_SUCCESS;
}

inline pi_result mock_piEventsWait(pi_uint32 num_events,
                                   const pi_event *event_list) {
  return PI_SUCCESS;
}

inline pi_result mock_piEventSetCallback(
    pi_event event, pi_int32 command_exec_callback_type,
    void (*pfn_notify)(pi_event event, pi_int32 event_command_status,
                       void *user_data),
    void *user_data) {
  return PI_SUCCESS;
}

inline pi_result mock_piEventSetStatus(pi_event event,
                                       pi_int32 execution_status) {
  return PI_SUCCESS;
}

inline pi_result mock_piEventRetain(pi_event event) {
  retainDummyHandle(event);
  return PI_SUCCESS;
}

inline pi_result mock_piEventRelease(pi_event event) {
  releaseDummyHandle(event);
  return PI_SUCCESS;
}

inline pi_result
mock_piextEventGetNativeHandle(pi_event event, pi_native_handle *nativeHandle) {
  *nativeHandle = reinterpret_cast<pi_native_handle>(event);
  return PI_SUCCESS;
}

inline pi_result
mock_piextEventCreateWithNativeHandle(pi_native_handle nativeHandle,
                                      pi_context context, bool ownNativeHandle,
                                      pi_event *event) {
  *event = reinterpret_cast<pi_event>(nativeHandle);
  retainDummyHandle(*event);
  return PI_SUCCESS;
}

//
// Sampler
//
inline pi_result
mock_piSamplerCreate(pi_context context,
                     const pi_sampler_properties *sampler_properties,
                     pi_sampler *result_sampler) {
  *result_sampler = createDummyHandle<pi_sampler>();
  return PI_SUCCESS;
}

inline pi_result mock_piSamplerGetInfo(pi_sampler sampler,
                                       pi_sampler_info param_name,
                                       size_t param_value_size,
                                       void *param_value,
                                       size_t *param_value_size_ret) {
  return PI_SUCCESS;
}

inline pi_result mock_piSamplerRetain(pi_sampler sampler) {
  retainDummyHandle(sampler);
  return PI_SUCCESS;
}

inline pi_result mock_piSamplerRelease(pi_sampler sampler) {
  releaseDummyHandle(sampler);
  return PI_SUCCESS;
}

//
// Queue Commands
//
inline pi_result mock_piEnqueueKernelLaunch(
    pi_queue queue, pi_kernel kernel, pi_uint32 work_dim,
    const size_t *global_work_offset, const size_t *global_work_size,
    const size_t *local_work_size, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event) {
  *event = createDummyHandle<pi_event>();
  return PI_SUCCESS;
}

inline pi_result mock_piEnqueueEventsWait(pi_queue command_queue,
                                          pi_uint32 num_events_in_wait_list,
                                          const pi_event *event_wait_list,
                                          pi_event *event) {
  *event = createDummyHandle<pi_event>();
  return PI_SUCCESS;
}

inline pi_result mock_piEnqueueEventsWaitWithBarrier(
    pi_queue command_queue, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event) {
  *event = createDummyHandle<pi_event>();
  return PI_SUCCESS;
}

inline pi_result
mock_piEnqueueMemBufferRead(pi_queue queue, pi_mem buffer,
                            pi_bool blocking_read, size_t offset, size_t size,
                            void *ptr, pi_uint32 num_events_in_wait_list,
                            const pi_event *event_wait_list, pi_event *event) {
  *event = createDummyHandle<pi_event>();
  return PI_SUCCESS;
}

inline pi_result mock_piEnqueueMemBufferReadRect(
    pi_queue command_queue, pi_mem buffer, pi_bool blocking_read,
    pi_buff_rect_offset buffer_offset, pi_buff_rect_offset host_offset,
    pi_buff_rect_region region, size_t buffer_row_pitch,
    size_t buffer_slice_pitch, size_t host_row_pitch, size_t host_slice_pitch,
    void *ptr, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event) {
  *event = createDummyHandle<pi_event>();
  return PI_SUCCESS;
}

inline pi_result
mock_piEnqueueMemBufferWrite(pi_queue command_queue, pi_mem buffer,
                             pi_bool blocking_write, size_t offset, size_t size,
                             const void *ptr, pi_uint32 num_events_in_wait_list,
                             const pi_event *event_wait_list, pi_event *event) {
  *event = createDummyHandle<pi_event>();
  return PI_SUCCESS;
}

inline pi_result mock_piEnqueueMemBufferWriteRect(
    pi_queue command_queue, pi_mem buffer, pi_bool blocking_write,
    pi_buff_rect_offset buffer_offset, pi_buff_rect_offset host_offset,
    pi_buff_rect_region region, size_t buffer_row_pitch,
    size_t buffer_slice_pitch, size_t host_row_pitch, size_t host_slice_pitch,
    const void *ptr, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event) {
  *event = createDummyHandle<pi_event>();
  return PI_SUCCESS;
}

inline pi_result
mock_piEnqueueMemBufferCopy(pi_queue command_queue, pi_mem src_buffer,
                            pi_mem dst_buffer, size_t src_offset,
                            size_t dst_offset, size_t size,
                            pi_uint32 num_events_in_wait_list,
                            const pi_event *event_wait_list, pi_event *event) {
  *event = createDummyHandle<pi_event>();
  return PI_SUCCESS;
}

inline pi_result mock_piEnqueueMemBufferCopyRect(
    pi_queue command_queue, pi_mem src_buffer, pi_mem dst_buffer,
    pi_buff_rect_offset src_origin, pi_buff_rect_offset dst_origin,
    pi_buff_rect_region region, size_t src_row_pitch, size_t src_slice_pitch,
    size_t dst_row_pitch, size_t dst_slice_pitch,
    pi_uint32 num_events_in_wait_list, const pi_event *event_wait_list,
    pi_event *event) {
  *event = createDummyHandle<pi_event>();
  return PI_SUCCESS;
}

inline pi_result mock_piEnqueueMemBufferFill(pi_queue command_queue,
                                             pi_mem buffer, const void *pattern,
                                             size_t pattern_size, size_t offset,
                                             size_t size,
                                             pi_uint32 num_events_in_wait_list,
                                             const pi_event *event_wait_list,
                                             pi_event *event) {
  *event = createDummyHandle<pi_event>();
  return PI_SUCCESS;
}

inline pi_result mock_piEnqueueMemImageRead(
    pi_queue command_queue, pi_mem image, pi_bool blocking_read,
    pi_image_offset origin, pi_image_region region, size_t row_pitch,
    size_t slice_pitch, void *ptr, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event) {
  *event = createDummyHandle<pi_event>();
  return PI_SUCCESS;
}

inline pi_result
mock_piEnqueueMemImageWrite(pi_queue command_queue, pi_mem image,
                            pi_bool blocking_write, pi_image_offset origin,
                            pi_image_region region, size_t input_row_pitch,
                            size_t input_slice_pitch, const void *ptr,
                            pi_uint32 num_events_in_wait_list,
                            const pi_event *event_wait_list, pi_event *event) {
  *event = createDummyHandle<pi_event>();
  return PI_SUCCESS;
}

inline pi_result
mock_piEnqueueMemImageCopy(pi_queue command_queue, pi_mem src_image,
                           pi_mem dst_image, pi_image_offset src_origin,
                           pi_image_offset dst_origin, pi_image_region region,
                           pi_uint32 num_events_in_wait_list,
                           const pi_event *event_wait_list, pi_event *event) {
  *event = createDummyHandle<pi_event>();
  return PI_SUCCESS;
}

inline pi_result
mock_piEnqueueMemImageFill(pi_queue command_queue, pi_mem image,
                           const void *fill_color, const size_t *origin,
                           const size_t *region,
                           pi_uint32 num_events_in_wait_list,
                           const pi_event *event_wait_list, pi_event *event) {
  *event = createDummyHandle<pi_event>();
  return PI_SUCCESS;
}

inline pi_result mock_piEnqueueMemBufferMap(pi_queue command_queue,
                                            pi_mem buffer, pi_bool blocking_map,
                                            pi_map_flags map_flags,
                                            size_t offset, size_t size,
                                            pi_uint32 num_events_in_wait_list,
                                            const pi_event *event_wait_list,
                                            pi_event *event, void **ret_map) {
  *event = createDummyHandle<pi_event>();

  auto parentDummyHandle = reinterpret_cast<DummyHandlePtrT>(buffer);
  *ret_map = (void *)(parentDummyHandle->MData);
  return PI_SUCCESS;
}

inline pi_result mock_piEnqueueMemUnmap(pi_queue command_queue, pi_mem memobj,
                                        void *mapped_ptr,
                                        pi_uint32 num_events_in_wait_list,
                                        const pi_event *event_wait_list,
                                        pi_event *event) {
  *event = createDummyHandle<pi_event>();
  return PI_SUCCESS;
}

inline pi_result
mock_piextKernelSetArgMemObj(pi_kernel kernel, pi_uint32 arg_index,
                             const pi_mem_obj_property *arg_properties,
                             const pi_mem *arg_value) {
  return PI_SUCCESS;
}

inline pi_result mock_piextKernelSetArgSampler(pi_kernel kernel,
                                               pi_uint32 arg_index,
                                               const pi_sampler *arg_value) {
  return PI_SUCCESS;
}

///
// USM
///
inline pi_result mock_piextUSMHostAlloc(void **result_ptr, pi_context context,
                                        pi_usm_mem_properties *properties,
                                        size_t size, pi_uint32 alignment) {
  assert(alignment < 16 && "TODO: mock_piextUSMHostAlloc handle alignment");
  *result_ptr = createDummyHandle<void *>(size);
  return PI_SUCCESS;
}

inline pi_result mock_piextUSMDeviceAlloc(void **result_ptr, pi_context context,
                                          pi_device device,
                                          pi_usm_mem_properties *properties,
                                          size_t size, pi_uint32 alignment) {
  assert(alignment < 16 && "TODO: mock_piextUSMHostAlloc handle alignment");
  *result_ptr = createDummyHandle<void *>(size);
  return PI_SUCCESS;
}

inline pi_result mock_piextUSMSharedAlloc(void **result_ptr, pi_context context,
                                          pi_device device,
                                          pi_usm_mem_properties *properties,
                                          size_t size, pi_uint32 alignment) {
  assert(alignment < 16 && "TODO: mock_piextUSMHostAlloc handle alignment");
  *result_ptr = createDummyHandle<void *>(size);
  return PI_SUCCESS;
}

inline pi_result mock_piextUSMPitchedAlloc(void **result_ptr,
                                           size_t *result_pitch,
                                           pi_context context, pi_device device,
                                           pi_usm_mem_properties *properties,
                                           size_t width_in_bytes, size_t height,
                                           unsigned int element_size_bytes) {
  *result_ptr = createDummyHandle<void *>(width_in_bytes * height);
  return PI_SUCCESS;
}

inline pi_result mock_piextUSMFree(pi_context context, void *ptr) {
  return PI_SUCCESS;
}

inline pi_result mock_piextUSMEnqueueMemset(pi_queue queue, void *ptr,
                                            pi_int32 value, size_t count,
                                            pi_uint32 num_events_in_waitlist,
                                            const pi_event *events_waitlist,
                                            pi_event *event) {
  *event = createDummyHandle<pi_event>();
  return PI_SUCCESS;
}

inline pi_result mock_piextUSMEnqueueMemcpy(pi_queue queue, pi_bool blocking,
                                            void *dst_ptr, const void *src_ptr,
                                            size_t size,
                                            pi_uint32 num_events_in_waitlist,
                                            const pi_event *events_waitlist,
                                            pi_event *event) {
  *event = createDummyHandle<pi_event>();
  return PI_SUCCESS;
}

inline pi_result mock_piextUSMEnqueuePrefetch(pi_queue queue, const void *ptr,
                                              size_t size,
                                              pi_usm_migration_flags flags,
                                              pi_uint32 num_events_in_waitlist,
                                              const pi_event *events_waitlist,
                                              pi_event *event) {
  *event = createDummyHandle<pi_event>();
  return PI_SUCCESS;
}

inline pi_result mock_piextUSMEnqueueMemAdvise(pi_queue queue, const void *ptr,
                                               size_t length,
                                               pi_mem_advice advice,
                                               pi_event *event) {
  *event = createDummyHandle<pi_event>();
  return PI_SUCCESS;
}

inline pi_result mock_piextUSMGetMemAllocInfo(
    pi_context context, const void *ptr, pi_mem_alloc_info param_name,
    size_t param_value_size, void *param_value, size_t *param_value_size_ret) {
  return PI_SUCCESS;
}

inline pi_result mock_piextUSMEnqueueFill2D(pi_queue queue, void *ptr,
                                            size_t pitch, size_t pattern_size,
                                            const void *pattern, size_t width,
                                            size_t height,
                                            pi_uint32 num_events_in_waitlist,
                                            const pi_event *events_waitlist,
                                            pi_event *event) {
  return PI_SUCCESS;
}

inline pi_result mock_piextUSMEnqueueMemset2D(pi_queue queue, void *ptr,
                                              size_t pitch, int value,
                                              size_t width, size_t height,
                                              pi_uint32 num_events_in_waitlist,
                                              const pi_event *events_waitlist,
                                              pi_event *event) {
  return PI_SUCCESS;
}

inline pi_result
mock_piextUSMEnqueueMemcpy2D(pi_queue queue, pi_bool blocking, void *dst_ptr,
                             size_t dst_pitch, const void *src_ptr,
                             size_t src_pitch, size_t width, size_t height,
                             pi_uint32 num_events_in_waitlist,
                             const pi_event *events_waitlist, pi_event *event) {
  return PI_SUCCESS;
}

inline pi_result mock_piextEnqueueDeviceGlobalVariableWrite(
    pi_queue queue, pi_program program, const char *name,
    pi_bool blocking_write, size_t count, size_t offset, const void *src,
    pi_uint32 num_events_in_wait_list, const pi_event *event_wait_list,
    pi_event *event) {
  *event = createDummyHandle<pi_event>();
  return PI_SUCCESS;
}

inline pi_result mock_piextEnqueueDeviceGlobalVariableRead(
    pi_queue queue, pi_program program, const char *name, pi_bool blocking_read,
    size_t count, size_t offset, void *dst, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event) {
  *event = createDummyHandle<pi_event>();
  return PI_SUCCESS;
}

inline pi_result mock_piextPluginGetOpaqueData(void *opaque_data_param,
                                               void **opaque_data_return) {
  return PI_SUCCESS;
}

inline pi_result
mock_piextCommandBufferCreate(pi_context context, pi_device device,
                              const pi_ext_command_buffer_desc *desc,
                              pi_ext_command_buffer *ret_command_buffer) {
  *ret_command_buffer = createDummyHandle<pi_ext_command_buffer>();
  return PI_SUCCESS;
}

inline pi_result
mock_piextCommandBufferRetain(pi_ext_command_buffer command_buffer) {
  retainDummyHandle(command_buffer);
  return PI_SUCCESS;
}

inline pi_result
mock_piextCommandBufferRelease(pi_ext_command_buffer command_buffer) {
  releaseDummyHandle(command_buffer);
  return PI_SUCCESS;
}

inline pi_result
mock_piextCommandBufferFinalize(pi_ext_command_buffer command_buffer) {
  return PI_SUCCESS;
}

inline pi_result mock_piextCommandBufferNDRangeKernel(
    pi_ext_command_buffer command_buffer, pi_kernel kernel, pi_uint32 work_dim,
    const size_t *global_work_offset, const size_t *global_work_size,
    const size_t *local_work_size, pi_uint32 num_sync_points_in_wait_list,
    const pi_ext_sync_point *sync_point_wait_list,
    pi_ext_sync_point *sync_point) {
  return PI_SUCCESS;
}

inline pi_result mock_piextCommandBufferMemcpyUSM(
    pi_ext_command_buffer command_buffer, void *dst_ptr, const void *src_ptr,
    size_t size, pi_uint32 num_sync_points_in_wait_list,
    const pi_ext_sync_point *sync_point_wait_list,
    pi_ext_sync_point *sync_point) {
  return PI_SUCCESS;
}

inline pi_result mock_piextCommandBufferMemBufferRead(
    pi_ext_command_buffer command_buffer, pi_mem buffer, size_t offset,
    size_t size, void *dst, pi_uint32 num_sync_points_in_wait_list,
    const pi_ext_sync_point *sync_point_wait_list,
    pi_ext_sync_point *sync_point) {
  return PI_SUCCESS;
}

inline pi_result mock_piextCommandBufferMemBufferReadRect(
    pi_ext_command_buffer command_buffer, pi_mem buffer,
    pi_buff_rect_offset buffer_offset, pi_buff_rect_offset host_offset,
    pi_buff_rect_region region, size_t buffer_row_pitch,
    size_t buffer_slice_pitch, size_t host_row_pitch, size_t host_slice_pitch,
    void *ptr, pi_uint32 num_sync_points_in_wait_list,
    const pi_ext_sync_point *sync_point_wait_list,
    pi_ext_sync_point *sync_point) {
  return PI_SUCCESS;
}

inline pi_result mock_piextCommandBufferMemBufferWrite(
    pi_ext_command_buffer command_buffer, pi_mem buffer, size_t offset,
    size_t size, const void *ptr, pi_uint32 num_sync_points_in_wait_list,
    const pi_ext_sync_point *sync_point_wait_list,
    pi_ext_sync_point *sync_point) {
  return PI_SUCCESS;
}

inline pi_result mock_piextCommandBufferMemBufferWriteRect(
    pi_ext_command_buffer command_buffer, pi_mem buffer,
    pi_buff_rect_offset buffer_offset, pi_buff_rect_offset host_offset,
    pi_buff_rect_region region, size_t BufferRowPitch,
    size_t buffer_slice_pitch, size_t host_row_pitch, size_t host_slice_pitch,
    const void *ptr, pi_uint32 num_sync_points_in_wait_list,
    const pi_ext_sync_point *sync_point_wait_list,
    pi_ext_sync_point *sync_point) {
  return PI_SUCCESS;
}

inline pi_result mock_piextEnqueueCommandBuffer(
    pi_ext_command_buffer command_buffer, pi_queue queue,
    pi_uint32 num_events_in_wait_list, const pi_event *event_wait_list,
    pi_event *event) {
  return PI_SUCCESS;
}

inline pi_result mock_piextCommandBufferMemBufferCopy(
    pi_ext_command_buffer command_buffer, pi_mem src_buffer, pi_mem dst_buffer,
    size_t src_offset, size_t dst_offset, size_t size,
    pi_uint32 num_sync_points_in_wait_list,
    const pi_ext_sync_point *sync_point_wait_list,
    pi_ext_sync_point *sync_point) {
  return PI_SUCCESS;
}

inline pi_result mock_piextCommandBufferMemBufferCopyRect(
    pi_ext_command_buffer command_buffer, pi_mem src_buffer, pi_mem dst_buffer,
    pi_buff_rect_offset src_origin, pi_buff_rect_offset dst_origin,
    pi_buff_rect_region region, size_t src_row_pitch, size_t src_slice_pitch,
    size_t dst_row_pitch, size_t dst_slice_pitch,
    pi_uint32 num_sync_points_in_wait_list,
    const pi_ext_sync_point *sync_point_wait_list,
    pi_ext_sync_point *sync_point) {
  return PI_SUCCESS;
}

inline pi_result mock_piextCommandBufferMemBufferFill(
    pi_ext_command_buffer command_buffer, pi_mem buffer, const void *pattern,
    size_t pattern_size, size_t offset, size_t size,
    pi_uint32 num_sync_points_in_wait_list,
    const pi_ext_sync_point *sync_point_wait_list,
    pi_ext_sync_point *sync_point) {
  return PI_SUCCESS;
}

inline pi_result mock_piextCommandBufferFillUSM(
    pi_ext_command_buffer command_buffer, void *ptr, const void *pattern,
    size_t pattern_size, size_t size, pi_uint32 num_sync_points_in_wait_list,
    const pi_ext_sync_point *sync_point_wait_list,
    pi_ext_sync_point *sync_point) {
  return PI_SUCCESS;
}

inline pi_result mock_piextCommandBufferPrefetchUSM(
    pi_ext_command_buffer command_buffer, const void *ptr, size_t size,
    pi_usm_migration_flags flags, pi_uint32 num_sync_points_in_wait_list,
    const pi_ext_sync_point *sync_point_wait_list,
    pi_ext_sync_point *sync_point) {
  return PI_SUCCESS;
}

inline pi_result mock_piextCommandBufferAdviseUSM(
    pi_ext_command_buffer command_buffer, const void *ptr, size_t length,
    pi_mem_advice advice, pi_uint32 num_sync_points_in_wait_list,
    const pi_ext_sync_point *sync_point_wait_list,
    pi_ext_sync_point *sync_point) {
  return PI_SUCCESS;
}

inline pi_result mock_piextSyncPointGetProfilingInfo(
    pi_event event, pi_ext_sync_point sync_point, pi_profiling_info param_name,
    size_t param_value_size, void *param_value, size_t *param_value_size_ret) {
  return PI_SUCCESS;
}

inline pi_result mock_piTearDown(void *PluginParameter) { return PI_SUCCESS; }

inline pi_result mock_piPluginGetLastError(char **message) {
  return PI_SUCCESS;
}

inline pi_result mock_piPluginGetBackendOption(pi_platform platform,
                                               const char *frontend_option,
                                               const char **backend_option) {
  *backend_option = "";
  return PI_SUCCESS;
}

// Returns the wall-clock timestamp of host for deviceTime and hostTime
inline pi_result mock_piGetDeviceAndHostTimer(pi_device device,
                                              uint64_t *deviceTime,
                                              uint64_t *hostTime) {

  using namespace std::chrono;
  auto timeNanoseconds =
      duration_cast<nanoseconds>(steady_clock::now().time_since_epoch())
          .count();
  if (deviceTime) {
    *deviceTime = timeNanoseconds;
  }
  if (hostTime) {
    *hostTime = timeNanoseconds;
  }
  return PI_SUCCESS;
}

inline pi_result mock_piextEnqueueReadHostPipe(
    pi_queue queue, pi_program program, const char *pipe_symbol,
    pi_bool blocking, void *ptr, size_t size, pi_uint32 num_events_in_waitlist,
    const pi_event *events_waitlist, pi_event *event) {
  *event = createDummyHandle<pi_event>();
  return PI_SUCCESS;
}

inline pi_result mock_piextEnqueueWriteHostPipe(
    pi_queue queue, pi_program program, const char *pipe_symbol,
    pi_bool blocking, void *ptr, size_t size, pi_uint32 num_events_in_waitlist,
    const pi_event *events_waitlist, pi_event *event) {
  *event = createDummyHandle<pi_event>();
  return PI_SUCCESS;
}

inline pi_result mock_piextEnablePeerAccess(pi_device command_device,
                                            pi_device peer_device) {
  return PI_SUCCESS;
}

inline pi_result mock_piextDisablePeerAccess(pi_device command_device,
                                             pi_device peer_device) {
  return PI_SUCCESS;
}

inline pi_result
mock_piextPeerAccessGetInfo(pi_device command_device, pi_device peer_device,
                            pi_peer_attr attr, size_t param_value_size,
                            void *param_value, size_t *param_value_size_ret) {
  if (param_value)
    *static_cast<pi_int32 *>(param_value) = 1;
  if (param_value_size_ret)
    *param_value_size_ret = sizeof(pi_int32);

  return PI_SUCCESS;
}

inline pi_result mock_piextUSMImport(const void *HostPtr, size_t Size,
                                     pi_context Context) {
  return PI_SUCCESS;
}

inline pi_result mock_piextUSMRelease(const void *HostPtr, pi_context Context) {
  return PI_SUCCESS;
}
