//===-------------- pi_structs.hpp - PI Trace Structs ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// clang-format off
// This file is auto-generated! Do not modify!
#pragma once
struct __attribute__((packed)) piPlatformsGet_args {
pi_uint32 num_entries;
pi_platform *platforms;
pi_uint32 *num_platforms;
};
struct __attribute__((packed)) piPlatformGetInfo_args {
pi_platform platform;
pi_platform_info param_name;
size_t param_value_size;
void *param_value;
size_t *param_value_size_ret;
};
struct __attribute__((packed)) piextPlatformGetNativeHandle_args {
pi_platform platform;
pi_native_handle *nativeHandle;
};
struct __attribute__((packed)) piextPlatformCreateWithNativeHandle_args {
pi_native_handle nativeHandle;
pi_platform *platform;
};
struct __attribute__((packed)) piDevicesGet_args {
pi_platform platform;
pi_device_type device_type;
pi_uint32 num_entries;
pi_device *devices;
pi_uint32 *num_devices;
};
struct __attribute__((packed)) piDeviceGetInfo_args {
pi_device device;
pi_device_info param_name;
size_t param_value_size;
void *param_value;
size_t *param_value_size_ret;
};
struct __attribute__((packed)) piDeviceRetain_args {
pi_device device;
};
struct __attribute__((packed)) piDeviceRelease_args {
pi_device device;
};
struct __attribute__((packed)) piDevicePartition_args {
pi_device device;
const pi_device_partition_property *properties;
pi_uint32 num_devices;
pi_device *out_devices;
pi_uint32 *out_num_devices;
};
struct __attribute__((packed)) piextDeviceGetNativeHandle_args {
pi_device device;
pi_native_handle *nativeHandle;
};
struct __attribute__((packed)) piextDeviceCreateWithNativeHandle_args {
pi_native_handle nativeHandle;
pi_platform platform;
pi_device *device;
};
struct __attribute__((packed)) piextDeviceSelectBinary_args {
pi_device device;
pi_device_binary *binaries;
pi_uint32 num_binaries;
pi_uint32 *selected_binary_ind;
};
struct __attribute__((packed)) piextGetDeviceFunctionPointer_args {
pi_device device;
pi_program program;
const char *function_name;
pi_uint64 *function_pointer_ret;
};
struct __attribute__((packed)) piContextGetInfo_args {
pi_context context;
pi_context_info param_name;
size_t param_value_size;
void *param_value;
size_t *param_value_size_ret;
};
struct __attribute__((packed)) piContextRetain_args {
pi_context context;
};
struct __attribute__((packed)) piContextRelease_args {
pi_context context;
};
struct __attribute__((packed)) piextContextSetExtendedDeleter_args {
pi_context context;
pi_context_extended_deleter func;
void *user_data;
};
struct __attribute__((packed)) piextContextGetNativeHandle_args {
pi_context context;
pi_native_handle *nativeHandle;
};
struct __attribute__((packed)) piextContextCreateWithNativeHandle_args {
pi_native_handle nativeHandle;
pi_uint32 numDevices;
const pi_device *devices;
bool pluginOwnsNativeHandle;
pi_context *context;
};
struct __attribute__((packed)) piQueueCreate_args {
pi_context context;
pi_device device;
pi_queue_properties properties;
pi_queue *queue;
};
struct __attribute__((packed)) piQueueGetInfo_args {
pi_queue command_queue;
pi_queue_info param_name;
size_t param_value_size;
void *param_value;
size_t *param_value_size_ret;
};
struct __attribute__((packed)) piQueueRetain_args {
pi_queue command_queue;
};
struct __attribute__((packed)) piQueueRelease_args {
pi_queue command_queue;
};
struct __attribute__((packed)) piQueueFinish_args {
pi_queue command_queue;
};
struct __attribute__((packed)) piQueueFlush_args {
pi_queue command_queue;
};
struct __attribute__((packed)) piextQueueGetNativeHandle_args {
pi_queue queue;
pi_native_handle *nativeHandle;
};
struct __attribute__((packed)) piextQueueCreateWithNativeHandle_args {
pi_native_handle nativeHandle;
pi_context context;
pi_queue *queue;
bool pluginOwnsNativeHandle;
};
struct __attribute__((packed)) piMemImageCreate_args {
pi_context context;
pi_mem_flags flags;
const pi_image_format *image_format;
const pi_image_desc *image_desc;
void *host_ptr;
pi_mem *ret_mem;
};
struct __attribute__((packed)) piMemImageGetInfo_args {
pi_mem image;
pi_image_info param_name;
size_t param_value_size;
void *param_value;
size_t *param_value_size_ret;
};
struct __attribute__((packed)) piMemRetain_args {
pi_mem mem;
};
struct __attribute__((packed)) piMemRelease_args {
pi_mem mem;
};
struct __attribute__((packed)) piMemBufferPartition_args {
pi_mem buffer;
pi_mem_flags flags;
pi_buffer_create_type buffer_create_type;
void *buffer_create_info;
pi_mem *ret_mem;
};
struct __attribute__((packed)) piextMemGetNativeHandle_args {
pi_mem mem;
pi_native_handle *nativeHandle;
};
struct __attribute__((packed)) piextMemCreateWithNativeHandle_args {
pi_native_handle nativeHandle;
pi_mem *mem;
};
struct __attribute__((packed)) piProgramCreate_args {
pi_context context;
const void *il;
size_t length;
pi_program *res_program;
};
struct __attribute__((packed)) piclProgramCreateWithSource_args {
pi_context context;
pi_uint32 count;
const char **strings;
const size_t *lengths;
pi_program *ret_program;
};
struct __attribute__((packed)) piProgramCreateWithBinary_args {
pi_context context;
pi_uint32 num_devices;
const pi_device *device_list;
const size_t *lengths;
const unsigned char **binaries;
size_t num_metadata_entries;
const pi_device_binary_property *metadata;
pi_int32 *binary_status;
pi_program *ret_program;
};
struct __attribute__((packed)) piProgramGetInfo_args {
pi_program program;
pi_program_info param_name;
size_t param_value_size;
void *param_value;
size_t *param_value_size_ret;
};
struct __attribute__((packed)) piProgramRetain_args {
pi_program program;
};
struct __attribute__((packed)) piProgramRelease_args {
pi_program program;
};
struct __attribute__((packed)) piextProgramSetSpecializationConstant_args {
pi_program prog;
pi_uint32 spec_id;
size_t spec_size;
const void *spec_value;
};
struct __attribute__((packed)) piextProgramGetNativeHandle_args {
pi_program program;
pi_native_handle *nativeHandle;
};
struct __attribute__((packed)) piextProgramCreateWithNativeHandle_args {
pi_native_handle nativeHandle;
pi_context context;
bool pluginOwnsNativeHandle;
pi_program *program;
};
struct __attribute__((packed)) piKernelCreate_args {
pi_program program;
const char *kernel_name;
pi_kernel *ret_kernel;
};
struct __attribute__((packed)) piKernelSetArg_args {
pi_kernel kernel;
pi_uint32 arg_index;
size_t arg_size;
const void *arg_value;
};
struct __attribute__((packed)) piKernelGetInfo_args {
pi_kernel kernel;
pi_kernel_info param_name;
size_t param_value_size;
void *param_value;
size_t *param_value_size_ret;
};
struct __attribute__((packed)) piKernelGetGroupInfo_args {
pi_kernel kernel;
pi_device device;
pi_kernel_group_info param_name;
size_t param_value_size;
void *param_value;
size_t *param_value_size_ret;
};
struct __attribute__((packed)) piKernelGetSubGroupInfo_args {
pi_kernel kernel;
pi_device device;
pi_kernel_sub_group_info param_name;
size_t input_value_size;
const void *input_value;
size_t param_value_size;
void *param_value;
size_t *param_value_size_ret;
};
struct __attribute__((packed)) piKernelRetain_args {
pi_kernel kernel;
};
struct __attribute__((packed)) piKernelRelease_args {
pi_kernel kernel;
};
struct __attribute__((packed)) piextKernelSetArgPointer_args {
pi_kernel kernel;
pi_uint32 arg_index;
size_t arg_size;
const void *arg_value;
};
struct __attribute__((packed)) piKernelSetExecInfo_args {
pi_kernel kernel;
pi_kernel_exec_info value_name;
size_t param_value_size;
const void *param_value;
};
struct __attribute__((packed)) piextKernelCreateWithNativeHandle_args {
pi_native_handle nativeHandle;
pi_context context;
pi_program program;
bool pluginOwnsNativeHandle;
pi_kernel *kernel;
};
struct __attribute__((packed)) piextKernelGetNativeHandle_args {
pi_kernel kernel;
pi_native_handle *nativeHandle;
};
struct __attribute__((packed)) piEventCreate_args {
pi_context context;
pi_event *ret_event;
};
struct __attribute__((packed)) piEventGetInfo_args {
pi_event event;
pi_event_info param_name;
size_t param_value_size;
void *param_value;
size_t *param_value_size_ret;
};
struct __attribute__((packed)) piEventGetProfilingInfo_args {
pi_event event;
pi_profiling_info param_name;
size_t param_value_size;
void *param_value;
size_t *param_value_size_ret;
};
struct __attribute__((packed)) piEventsWait_args {
pi_uint32 num_events;
const pi_event *event_list;
};
struct __attribute__((packed)) piEventSetStatus_args {
pi_event event;
pi_int32 execution_status;
};
struct __attribute__((packed)) piEventRetain_args {
pi_event event;
};
struct __attribute__((packed)) piEventRelease_args {
pi_event event;
};
struct __attribute__((packed)) piextEventGetNativeHandle_args {
pi_event event;
pi_native_handle *nativeHandle;
};
struct __attribute__((packed)) piextEventCreateWithNativeHandle_args {
pi_native_handle nativeHandle;
pi_context context;
bool ownNativeHandle;
pi_event *event;
};
struct __attribute__((packed)) piSamplerCreate_args {
pi_context context;
const pi_sampler_properties *sampler_properties;
pi_sampler *result_sampler;
};
struct __attribute__((packed)) piSamplerGetInfo_args {
pi_sampler sampler;
pi_sampler_info param_name;
size_t param_value_size;
void *param_value;
size_t *param_value_size_ret;
};
struct __attribute__((packed)) piSamplerRetain_args {
pi_sampler sampler;
};
struct __attribute__((packed)) piSamplerRelease_args {
pi_sampler sampler;
};
struct __attribute__((packed)) piEnqueueKernelLaunch_args {
pi_queue queue;
pi_kernel kernel;
pi_uint32 work_dim;
const size_t *global_work_offset;
const size_t *global_work_size;
const size_t *local_work_size;
pi_uint32 num_events_in_wait_list;
const pi_event *event_wait_list;
pi_event *event;
};
struct __attribute__((packed)) piEnqueueEventsWait_args {
pi_queue command_queue;
pi_uint32 num_events_in_wait_list;
const pi_event *event_wait_list;
pi_event *event;
};
struct __attribute__((packed)) piEnqueueEventsWaitWithBarrier_args {
pi_queue command_queue;
pi_uint32 num_events_in_wait_list;
const pi_event *event_wait_list;
pi_event *event;
};
struct __attribute__((packed)) piEnqueueMemBufferRead_args {
pi_queue queue;
pi_mem buffer;
pi_bool blocking_read;
size_t offset;
size_t size;
void *ptr;
pi_uint32 num_events_in_wait_list;
const pi_event *event_wait_list;
pi_event *event;
};
struct __attribute__((packed)) piEnqueueMemBufferReadRect_args {
pi_queue command_queue;
pi_mem buffer;
pi_bool blocking_read;
pi_buff_rect_offset buffer_offset;
pi_buff_rect_offset host_offset;
pi_buff_rect_region region;
size_t buffer_row_pitch;
size_t buffer_slice_pitch;
size_t host_row_pitch;
size_t host_slice_pitch;
void *ptr;
pi_uint32 num_events_in_wait_list;
const pi_event *event_wait_list;
pi_event *event;
};
struct __attribute__((packed)) piEnqueueMemBufferWrite_args {
pi_queue command_queue;
pi_mem buffer;
pi_bool blocking_write;
size_t offset;
size_t size;
const void *ptr;
pi_uint32 num_events_in_wait_list;
const pi_event *event_wait_list;
pi_event *event;
};
struct __attribute__((packed)) piEnqueueMemBufferWriteRect_args {
pi_queue command_queue;
pi_mem buffer;
pi_bool blocking_write;
pi_buff_rect_offset buffer_offset;
pi_buff_rect_offset host_offset;
pi_buff_rect_region region;
size_t buffer_row_pitch;
size_t buffer_slice_pitch;
size_t host_row_pitch;
size_t host_slice_pitch;
const void *ptr;
pi_uint32 num_events_in_wait_list;
const pi_event *event_wait_list;
pi_event *event;
};
struct __attribute__((packed)) piEnqueueMemBufferCopy_args {
pi_queue command_queue;
pi_mem src_buffer;
pi_mem dst_buffer;
size_t src_offset;
size_t dst_offset;
size_t size;
pi_uint32 num_events_in_wait_list;
const pi_event *event_wait_list;
pi_event *event;
};
struct __attribute__((packed)) piEnqueueMemBufferCopyRect_args {
pi_queue command_queue;
pi_mem src_buffer;
pi_mem dst_buffer;
pi_buff_rect_offset src_origin;
pi_buff_rect_offset dst_origin;
pi_buff_rect_region region;
size_t src_row_pitch;
size_t src_slice_pitch;
size_t dst_row_pitch;
size_t dst_slice_pitch;
pi_uint32 num_events_in_wait_list;
const pi_event *event_wait_list;
pi_event *event;
};
struct __attribute__((packed)) piEnqueueMemBufferFill_args {
pi_queue command_queue;
pi_mem buffer;
const void *pattern;
size_t pattern_size;
size_t offset;
size_t size;
pi_uint32 num_events_in_wait_list;
const pi_event *event_wait_list;
pi_event *event;
};
struct __attribute__((packed)) piEnqueueMemImageRead_args {
pi_queue command_queue;
pi_mem image;
pi_bool blocking_read;
pi_image_offset origin;
pi_image_region region;
size_t row_pitch;
size_t slice_pitch;
void *ptr;
pi_uint32 num_events_in_wait_list;
const pi_event *event_wait_list;
pi_event *event;
};
struct __attribute__((packed)) piEnqueueMemImageWrite_args {
pi_queue command_queue;
pi_mem image;
pi_bool blocking_write;
pi_image_offset origin;
pi_image_region region;
size_t input_row_pitch;
size_t input_slice_pitch;
const void *ptr;
pi_uint32 num_events_in_wait_list;
const pi_event *event_wait_list;
pi_event *event;
};
struct __attribute__((packed)) piEnqueueMemImageCopy_args {
pi_queue command_queue;
pi_mem src_image;
pi_mem dst_image;
pi_image_offset src_origin;
pi_image_offset dst_origin;
pi_image_region region;
pi_uint32 num_events_in_wait_list;
const pi_event *event_wait_list;
pi_event *event;
};
struct __attribute__((packed)) piEnqueueMemImageFill_args {
pi_queue command_queue;
pi_mem image;
const void *fill_color;
const size_t *origin;
const size_t *region;
pi_uint32 num_events_in_wait_list;
const pi_event *event_wait_list;
pi_event *event;
};
struct __attribute__((packed)) piEnqueueMemBufferMap_args {
pi_queue command_queue;
pi_mem buffer;
pi_bool blocking_map;
pi_map_flags map_flags;
size_t offset;
size_t size;
pi_uint32 num_events_in_wait_list;
const pi_event *event_wait_list;
pi_event *event;
void **ret_map;
};
struct __attribute__((packed)) piEnqueueMemUnmap_args {
pi_queue command_queue;
pi_mem memobj;
void *mapped_ptr;
pi_uint32 num_events_in_wait_list;
const pi_event *event_wait_list;
pi_event *event;
};
struct __attribute__((packed)) piextKernelSetArgMemObj_args {
pi_kernel kernel;
pi_uint32 arg_index;
const pi_mem *arg_value;
};
struct __attribute__((packed)) piextKernelSetArgSampler_args {
pi_kernel kernel;
pi_uint32 arg_index;
const pi_sampler *arg_value;
};
struct __attribute__((packed)) piextUSMHostAlloc_args {
void **result_ptr;
pi_context context;
pi_usm_mem_properties *properties;
size_t size;
pi_uint32 alignment;
};
struct __attribute__((packed)) piextUSMDeviceAlloc_args {
void **result_ptr;
pi_context context;
pi_device device;
pi_usm_mem_properties *properties;
size_t size;
pi_uint32 alignment;
};
struct __attribute__((packed)) piextUSMSharedAlloc_args {
void **result_ptr;
pi_context context;
pi_device device;
pi_usm_mem_properties *properties;
size_t size;
pi_uint32 alignment;
};
struct __attribute__((packed)) piextUSMFree_args {
pi_context context;
void *ptr;
};
struct __attribute__((packed)) piextUSMEnqueueMemset_args {
pi_queue queue;
void *ptr;
pi_int32 value;
size_t count;
pi_uint32 num_events_in_waitlist;
const pi_event *events_waitlist;
pi_event *event;
};
struct __attribute__((packed)) piextUSMEnqueueMemcpy_args {
pi_queue queue;
pi_bool blocking;
void *dst_ptr;
const void *src_ptr;
size_t size;
pi_uint32 num_events_in_waitlist;
const pi_event *events_waitlist;
pi_event *event;
};
struct __attribute__((packed)) piextUSMEnqueuePrefetch_args {
pi_queue queue;
const void *ptr;
size_t size;
pi_usm_migration_flags flags;
pi_uint32 num_events_in_waitlist;
const pi_event *events_waitlist;
pi_event *event;
};
struct __attribute__((packed)) piextUSMEnqueueMemAdvise_args {
pi_queue queue;
const void *ptr;
size_t length;
pi_mem_advice advice;
pi_event *event;
};
struct __attribute__((packed)) piextUSMGetMemAllocInfo_args {
pi_context context;
const void *ptr;
pi_mem_info param_name;
size_t param_value_size;
void *param_value;
size_t *param_value_size_ret;
};
struct __attribute__((packed)) piextPluginGetOpaqueData_args {
void *opaque_data_param;
void **opaque_data_return;
};
struct __attribute__((packed)) piTearDown_args {
void *PluginParameter;
};
