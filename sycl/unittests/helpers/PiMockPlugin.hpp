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

#include <cstring>
#include <sycl/detail/pi.hpp>

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
      "cl_khr_fp64 cl_khr_fp16 cl_khr_il_program";
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
      assert(param_value_size == sizeof(MockSupportedExtensions));
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
  static uintptr_t NextContext = 0;
  *ret_context = reinterpret_cast<pi_context>(++NextContext);
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

inline pi_result mock_piContextRetain(pi_context context) { return PI_SUCCESS; }

inline pi_result mock_piContextRelease(pi_context context) {
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
  return PI_SUCCESS;
}

//
// Queue
//
inline pi_result mock_piQueueCreate(pi_context context, pi_device device,
                                    pi_queue_properties properties,
                                    pi_queue *queue) {
  static uintptr_t NextQueue = 0;
  *queue = reinterpret_cast<pi_queue>(++NextQueue);
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
  return PI_SUCCESS;
}

inline pi_result mock_piQueueRelease(pi_queue command_queue) {
  return PI_SUCCESS;
}

inline pi_result mock_piQueueFinish(pi_queue command_queue) {
  return PI_SUCCESS;
}

inline pi_result mock_piQueueFlush(pi_queue command_queue) {
  return PI_SUCCESS;
}

inline pi_result
mock_piextQueueGetNativeHandle(pi_queue queue, pi_native_handle *nativeHandle) {
  *nativeHandle = reinterpret_cast<pi_native_handle>(queue);
  return PI_SUCCESS;
}

inline pi_result mock_piextQueueCreateWithNativeHandle(
    pi_native_handle nativeHandle, pi_context context, pi_device device,
    bool pluginOwnsNativeHandle, pi_queue *queue) {
  return PI_SUCCESS;
}

//
// Memory
//
inline pi_result
mock_piMemBufferCreate(pi_context context, pi_mem_flags flags, size_t size,
                       void *host_ptr, pi_mem *ret_mem,
                       const pi_mem_properties *properties = nullptr) {
  static uintptr_t NextMem = 0;
  *ret_mem = reinterpret_cast<pi_mem>(++NextMem);
  return PI_SUCCESS;
}

inline pi_result mock_piMemImageCreate(pi_context context, pi_mem_flags flags,
                                       const pi_image_format *image_format,
                                       const pi_image_desc *image_desc,
                                       void *host_ptr, pi_mem *ret_mem) {
  static uintptr_t NextMem = 0;
  *ret_mem = reinterpret_cast<pi_mem>(++NextMem);
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

inline pi_result mock_piMemRetain(pi_mem mem) { return PI_SUCCESS; }

inline pi_result mock_piMemRelease(pi_mem mem) { return PI_SUCCESS; }

inline pi_result
mock_piMemBufferPartition(pi_mem buffer, pi_mem_flags flags,
                          pi_buffer_create_type buffer_create_type,
                          void *buffer_create_info, pi_mem *ret_mem) {
  return PI_SUCCESS;
}

inline pi_result mock_piextMemGetNativeHandle(pi_mem mem,
                                              pi_native_handle *nativeHandle) {
  *nativeHandle = reinterpret_cast<pi_native_handle>(mem);
  return PI_SUCCESS;
}

inline pi_result
mock_piextMemCreateWithNativeHandle(pi_native_handle nativeHandle,
                                    pi_context context, bool ownNativeHandle,
                                    pi_mem *mem) {
  return PI_SUCCESS;
}

//
// Program
//

inline pi_result mock_piProgramCreate(pi_context context, const void *il,
                                      size_t length, pi_program *res_program) {
  static uintptr_t NextProgram = 0;
  *res_program = reinterpret_cast<pi_program>(++NextProgram);
  return PI_SUCCESS;
}

inline pi_result mock_piclProgramCreateWithSource(pi_context context,
                                                  pi_uint32 count,
                                                  const char **strings,
                                                  const size_t *lengths,
                                                  pi_program *ret_program) {
  static uintptr_t NextProgram = 100;
  *ret_program = reinterpret_cast<pi_program>(++NextProgram);
  return PI_SUCCESS;
}

inline pi_result mock_piProgramCreateWithBinary(
    pi_context context, pi_uint32 num_devices, const pi_device *device_list,
    const size_t *lengths, const unsigned char **binaries,
    size_t num_metadata_entries, const pi_device_binary_property *metadata,
    pi_int32 *binary_status, pi_program *ret_program) {
  static uintptr_t NextProgram = 200;
  *ret_program = reinterpret_cast<pi_program>(++NextProgram);
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
      *static_cast<unsigned char *>(param_value) = 1;
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
  static uintptr_t NextProgram = 300;
  *ret_program = reinterpret_cast<pi_program>(++NextProgram);
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

inline pi_result mock_piProgramRetain(pi_program program) { return PI_SUCCESS; }

inline pi_result mock_piProgramRelease(pi_program program) {
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
  return PI_SUCCESS;
}

//
// Kernel
//

inline pi_result mock_piKernelCreate(pi_program program,
                                     const char *kernel_name,
                                     pi_kernel *ret_kernel) {
  static uintptr_t NextKernel = 0;
  *ret_kernel = reinterpret_cast<pi_kernel>(++NextKernel);
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

inline pi_result mock_piKernelRetain(pi_kernel kernel) { return PI_SUCCESS; }

inline pi_result mock_piKernelRelease(pi_kernel kernel) { return PI_SUCCESS; }

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
  static uintptr_t NextEvent = 0;
  *ret_event = reinterpret_cast<pi_event>(++NextEvent);
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

inline pi_result mock_piEventRetain(pi_event event) { return PI_SUCCESS; }

inline pi_result mock_piEventRelease(pi_event event) { return PI_SUCCESS; }

inline pi_result
mock_piextEventGetNativeHandle(pi_event event, pi_native_handle *nativeHandle) {
  *nativeHandle = reinterpret_cast<pi_native_handle>(event);
  return PI_SUCCESS;
}

inline pi_result
mock_piextEventCreateWithNativeHandle(pi_native_handle nativeHandle,
                                      pi_context context, bool ownNativeHandle,
                                      pi_event *event) {
  return PI_SUCCESS;
}

//
// Sampler
//
inline pi_result
mock_piSamplerCreate(pi_context context,
                     const pi_sampler_properties *sampler_properties,
                     pi_sampler *result_sampler) {
  static uintptr_t NextSampler = 0;
  *result_sampler = reinterpret_cast<pi_sampler>(++NextSampler);
  return PI_SUCCESS;
}

inline pi_result mock_piSamplerGetInfo(pi_sampler sampler,
                                       pi_sampler_info param_name,
                                       size_t param_value_size,
                                       void *param_value,
                                       size_t *param_value_size_ret) {
  return PI_SUCCESS;
}

inline pi_result mock_piSamplerRetain(pi_sampler sampler) { return PI_SUCCESS; }

inline pi_result mock_piSamplerRelease(pi_sampler sampler) {
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
  static uintptr_t NextEvent = 1000;
  *event = reinterpret_cast<pi_event>(++NextEvent);
  return PI_SUCCESS;
}

inline pi_result mock_piEnqueueNativeKernel(
    pi_queue queue, void (*user_func)(void *), void *args, size_t cb_args,
    pi_uint32 num_mem_objects, const pi_mem *mem_list,
    const void **args_mem_loc, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event) {
  return PI_SUCCESS;
}

inline pi_result mock_piEnqueueEventsWait(pi_queue command_queue,
                                          pi_uint32 num_events_in_wait_list,
                                          const pi_event *event_wait_list,
                                          pi_event *event) {
  return PI_SUCCESS;
}

inline pi_result mock_piEnqueueEventsWaitWithBarrier(
    pi_queue command_queue, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event) {
  return PI_SUCCESS;
}

inline pi_result
mock_piEnqueueMemBufferRead(pi_queue queue, pi_mem buffer,
                            pi_bool blocking_read, size_t offset, size_t size,
                            void *ptr, pi_uint32 num_events_in_wait_list,
                            const pi_event *event_wait_list, pi_event *event) {
  return PI_SUCCESS;
}

inline pi_result mock_piEnqueueMemBufferReadRect(
    pi_queue command_queue, pi_mem buffer, pi_bool blocking_read,
    pi_buff_rect_offset buffer_offset, pi_buff_rect_offset host_offset,
    pi_buff_rect_region region, size_t buffer_row_pitch,
    size_t buffer_slice_pitch, size_t host_row_pitch, size_t host_slice_pitch,
    void *ptr, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event) {
  return PI_SUCCESS;
}

inline pi_result
mock_piEnqueueMemBufferWrite(pi_queue command_queue, pi_mem buffer,
                             pi_bool blocking_write, size_t offset, size_t size,
                             const void *ptr, pi_uint32 num_events_in_wait_list,
                             const pi_event *event_wait_list, pi_event *event) {
  return PI_SUCCESS;
}

inline pi_result mock_piEnqueueMemBufferWriteRect(
    pi_queue command_queue, pi_mem buffer, pi_bool blocking_write,
    pi_buff_rect_offset buffer_offset, pi_buff_rect_offset host_offset,
    pi_buff_rect_region region, size_t buffer_row_pitch,
    size_t buffer_slice_pitch, size_t host_row_pitch, size_t host_slice_pitch,
    const void *ptr, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event) {
  return PI_SUCCESS;
}

inline pi_result
mock_piEnqueueMemBufferCopy(pi_queue command_queue, pi_mem src_buffer,
                            pi_mem dst_buffer, size_t src_offset,
                            size_t dst_offset, size_t size,
                            pi_uint32 num_events_in_wait_list,
                            const pi_event *event_wait_list, pi_event *event) {
  return PI_SUCCESS;
}

inline pi_result mock_piEnqueueMemBufferCopyRect(
    pi_queue command_queue, pi_mem src_buffer, pi_mem dst_buffer,
    pi_buff_rect_offset src_origin, pi_buff_rect_offset dst_origin,
    pi_buff_rect_region region, size_t src_row_pitch, size_t src_slice_pitch,
    size_t dst_row_pitch, size_t dst_slice_pitch,
    pi_uint32 num_events_in_wait_list, const pi_event *event_wait_list,
    pi_event *event) {
  return PI_SUCCESS;
}

inline pi_result mock_piEnqueueMemBufferFill(pi_queue command_queue,
                                             pi_mem buffer, const void *pattern,
                                             size_t pattern_size, size_t offset,
                                             size_t size,
                                             pi_uint32 num_events_in_wait_list,
                                             const pi_event *event_wait_list,
                                             pi_event *event) {
  return PI_SUCCESS;
}

inline pi_result mock_piEnqueueMemImageRead(
    pi_queue command_queue, pi_mem image, pi_bool blocking_read,
    pi_image_offset origin, pi_image_region region, size_t row_pitch,
    size_t slice_pitch, void *ptr, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event) {
  return PI_SUCCESS;
}

inline pi_result
mock_piEnqueueMemImageWrite(pi_queue command_queue, pi_mem image,
                            pi_bool blocking_write, pi_image_offset origin,
                            pi_image_region region, size_t input_row_pitch,
                            size_t input_slice_pitch, const void *ptr,
                            pi_uint32 num_events_in_wait_list,
                            const pi_event *event_wait_list, pi_event *event) {
  return PI_SUCCESS;
}

inline pi_result
mock_piEnqueueMemImageCopy(pi_queue command_queue, pi_mem src_image,
                           pi_mem dst_image, pi_image_offset src_origin,
                           pi_image_offset dst_origin, pi_image_region region,
                           pi_uint32 num_events_in_wait_list,
                           const pi_event *event_wait_list, pi_event *event) {
  return PI_SUCCESS;
}

inline pi_result
mock_piEnqueueMemImageFill(pi_queue command_queue, pi_mem image,
                           const void *fill_color, const size_t *origin,
                           const size_t *region,
                           pi_uint32 num_events_in_wait_list,
                           const pi_event *event_wait_list, pi_event *event) {
  return PI_SUCCESS;
}

inline pi_result mock_piEnqueueMemBufferMap(pi_queue command_queue,
                                            pi_mem buffer, pi_bool blocking_map,
                                            pi_map_flags map_flags,
                                            size_t offset, size_t size,
                                            pi_uint32 num_events_in_wait_list,
                                            const pi_event *event_wait_list,
                                            pi_event *event, void **ret_map) {
  return PI_SUCCESS;
}

inline pi_result mock_piEnqueueMemUnmap(pi_queue command_queue, pi_mem memobj,
                                        void *mapped_ptr,
                                        pi_uint32 num_events_in_wait_list,
                                        const pi_event *event_wait_list,
                                        pi_event *event) {
  return PI_SUCCESS;
}

inline pi_result mock_piextKernelSetArgMemObj(pi_kernel kernel,
                                              pi_uint32 arg_index,
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
  *result_ptr = (void *)0x1;
  return PI_SUCCESS;
}

inline pi_result mock_piextUSMDeviceAlloc(void **result_ptr, pi_context context,
                                          pi_device device,
                                          pi_usm_mem_properties *properties,
                                          size_t size, pi_uint32 alignment) {
  *result_ptr = (void *)0x1;
  return PI_SUCCESS;
}

inline pi_result mock_piextUSMSharedAlloc(void **result_ptr, pi_context context,
                                          pi_device device,
                                          pi_usm_mem_properties *properties,
                                          size_t size, pi_uint32 alignment) {
  *result_ptr = (void *)0x1;
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
  return PI_SUCCESS;
}

inline pi_result mock_piextUSMEnqueueMemcpy(pi_queue queue, pi_bool blocking,
                                            void *dst_ptr, const void *src_ptr,
                                            size_t size,
                                            pi_uint32 num_events_in_waitlist,
                                            const pi_event *events_waitlist,
                                            pi_event *event) {
  return PI_SUCCESS;
}

inline pi_result mock_piextUSMEnqueuePrefetch(pi_queue queue, const void *ptr,
                                              size_t size,
                                              pi_usm_migration_flags flags,
                                              pi_uint32 num_events_in_waitlist,
                                              const pi_event *events_waitlist,
                                              pi_event *event) {
  return PI_SUCCESS;
}

inline pi_result mock_piextUSMEnqueueMemAdvise(pi_queue queue, const void *ptr,
                                               size_t length,
                                               pi_mem_advice advice,
                                               pi_event *event) {
  return PI_SUCCESS;
}

inline pi_result mock_piextUSMGetMemAllocInfo(
    pi_context context, const void *ptr, pi_mem_alloc_info param_name,
    size_t param_value_size, void *param_value, size_t *param_value_size_ret) {
  return PI_SUCCESS;
}

inline pi_result mock_piextPluginGetOpaqueData(void *opaque_data_param,
                                               void **opaque_data_return) {
  return PI_SUCCESS;
}

inline pi_result mock_piTearDown(void *PluginParameter) { return PI_SUCCESS; }

inline pi_result mock_piPluginGetLastError(char **message) {
  return PI_SUCCESS;
}

#define _PI_MOCK_PLUGIN_CONCAT(A, B) A##B
#define PI_MOCK_PLUGIN_CONCAT(A, B) _PI_MOCK_PLUGIN_CONCAT(A, B)

inline pi_plugin::FunctionPointers getMockedFunctionPointers() {
  return {
#define _PI_API(api) PI_MOCK_PLUGIN_CONCAT(mock_, api),
#include <sycl/detail/pi.def>
  };
}

#undef PI_MOCK_PLUGIN_CONCAT
#undef _PI_MOCK_PLUGIN_CONCAT
