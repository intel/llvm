//===-- pi.cpp - SYCL PI interface impl -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CL/sycl/detail/pi.h"
#include <CL/sycl/detail/common.hpp>

#include <assert.h>
#include <cstdint>

cl_int piSelectDeviceImage(pi_context ctx, pi_device_image **images,
                           cl_uint num_images,
                           pi_device_image **selected_image) {
  // TODO dummy implementation.
  // Real implementaion will use the same mechanism OpenCL ICD dispatcher
  // uses. Somthing like:
  //   PI_VALIDATE_HANDLE_RETURN_HANDLE(ctx, PI_INVALID_CONTEXT);
  //     return context->dispatch->piSelectDeviceImage(
  //       ctx, images, num_images, selected_image);
  // where context->dispatch is set to the dispatch table provided by PI
  // plugin for platform/device the ctx was created for.

  *selected_image = num_images > 0 ? images[0] : nullptr;
  return PI_SUCCESS;
}

pi_int piGetPlatformInfo(pi_platform_id platform, pi_platform_info param_name,
                         size_t param_value_size, void *param_value,
                         size_t *param_value_size_ret) {
  return clGetPlatformInfo(platform, param_name, param_value_size, param_value,
                           param_value_size_ret);
}

pi_command_queue piCreateCommandQueue(pi_context context, pi_device_id device,
                                      pi_command_queue_properties properties,
                                      pi_int *errcode_ret) {

#ifdef CL_VERSION_2_0
  pi_queue_properties CreationFlagProperties[] = {CL_QUEUE_PROPERTIES,
                                                  properties, 0};
  return clCreateCommandQueueWithProperties(
      context, device, CreationFlagProperties, errcode_ret);
#else
  return clCreateCommandQueue(context, device, properties, errcode_ret);
#endif
}

pi_int piGetDeviceInfo(pi_device_id device, pi_device_info param_name,
                       size_t param_value_size, void *param_value,
                       size_t *param_value_size_ret) {
  return clGetDeviceInfo(device, param_name, param_value_size, param_value,
                         param_value_size_ret);
}

pi_int piSetKernelArg(pi_kernel kernel, pi_uint arg_index, size_t arg_size,
                      const void *arg_value) {
  return clSetKernelArg(kernel, arg_index, arg_size, arg_value);
}

pi_int piEnqueueNDRangeKernel(
    pi_command_queue command_queue, pi_kernel kernel, pi_uint work_dim,
    const size_t *global_work_offset, const size_t *global_work_size,
    const size_t *local_work_size, pi_uint num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event) {
  return clEnqueueNDRangeKernel(
      command_queue, kernel, work_dim, global_work_offset, global_work_size,
      local_work_size, num_events_in_wait_list, event_wait_list, event);
}

pi_program piLinkProgram(pi_context context, pi_uint num_devices,
                         const pi_device_id *device_list, const char *options,
                         pi_uint num_input_programs,
                         const pi_program *input_programs,
                         void(PI_CALLBACK *pfn_notify)(pi_program program,
                                                       void *user_data),
                         void *user_data, pi_int *errcode_ret) {
  return clLinkProgram(context, num_devices, device_list, options,
                       num_input_programs, input_programs, pfn_notify,
                       user_data, errcode_ret);
}

pi_program piCreateProgramWithSource(pi_context context, pi_uint count,
                                     const char **strings,
                                     const size_t *lengths,
                                     pi_int *errcode_ret) {
  return clCreateProgramWithSource(context, count, strings, lengths,
                                   errcode_ret);
}

pi_kernel piCreateKernel(pi_program program, const char *kernel_name,
                         pi_int *errcode_ret) {
  return clCreateKernel(program, kernel_name, errcode_ret);
}

pi_int piGetPlatformIDs(pi_uint num_entries, pi_platform_id *platforms,
                        pi_uint *num_platforms) {
  return clGetPlatformIDs(num_entries, platforms, num_platforms);
}

pi_int piGetDeviceIDs(pi_platform_id platform, pi_device_type device_type,
                      pi_uint num_entries, pi_device_id *devices,
                      pi_uint *num_devices) {
  return clGetDeviceIDs(platform, device_type, num_entries, devices,
                        num_devices);
}

pi_int piWaitForEvents(pi_uint num_events, const pi_event *event_list) {
  return clWaitForEvents(num_events, event_list);
}

pi_mem piCreateBuffer(pi_context context, pi_mem_flags flags, size_t size,
                      void *host_ptr, pi_int *errcode_ret) {
  return clCreateBuffer(context, flags, size, host_ptr, errcode_ret);
}

pi_int piGetProgramInfo(pi_program program, pi_program_info param_name,
                        size_t param_value_size, void *param_value,
                        size_t *param_value_size_ret) {
  return clGetProgramInfo(program, param_name, param_value_size, param_value,
                          param_value_size_ret);
}

pi_context piCreateContext(
    const pi_context_properties *properties, pi_uint num_devices,
    const pi_device_id *devices,
    void(PI_CALLBACK *pfn_notify)(const char *errinfo, const void *private_info,
                                  size_t cb, void *user_data),
    void *user_data, pi_int *errcode_ret) {
  return clCreateContext(properties, num_devices, devices, pfn_notify,
                         user_data, errcode_ret);
}

pi_int piGetContextInfo(pi_context context, pi_context_info param_name,
                        size_t param_value_size, void *param_value,
                        size_t *param_value_size_ret) {
  return clGetContextInfo(context, param_name, param_value_size, param_value,
                          param_value_size_ret);
}

pi_program piCreateProgramWithBinary(pi_context context, pi_uint num_devices,
                                     const pi_device_id *device_list,
                                     const size_t *lengths,
                                     const unsigned char **binaries,
                                     pi_int *binary_status,
                                     pi_int *errcode_ret) {
  return clCreateProgramWithBinary(context, num_devices, device_list, lengths,
                                   binaries, binary_status, errcode_ret);
}

pi_program piCreateProgramWithIL(pi_context context, const void *il,
                                 size_t length, pi_int *errcode_ret) {
  return clCreateProgramWithIL(context, il, length, errcode_ret);
}

pi_int piGetCommandQueueInfo(pi_command_queue command_queue,
                             pi_command_queue_info param_name,
                             size_t param_value_size, void *param_value,
                             size_t *param_value_size_ret) {
  return clGetCommandQueueInfo(command_queue, param_name, param_value_size,
                               param_value, param_value_size_ret);
}

pi_int piEnqueueWriteBuffer(pi_command_queue command_queue, pi_mem buffer,
                            pi_bool blocking_write, size_t offset, size_t size,
                            const void *ptr, pi_uint num_events_in_wait_list,
                            const pi_event *event_wait_list, pi_event *event) {

  return clEnqueueWriteBuffer(command_queue, buffer, blocking_write, offset,
                              size, ptr, num_events_in_wait_list,
                              event_wait_list, event);
}

pi_int
piEnqueueWriteBufferRect(pi_command_queue command_queue, pi_mem buffer,
                         pi_bool blocking_write, const size_t *buffer_offset,
                         const size_t *host_offset, const size_t *region,
                         size_t buffer_row_pitch, size_t buffer_slice_pitch,
                         size_t host_row_pitch, size_t host_slice_pitch,
                         const void *ptr, pi_uint num_events_in_wait_list,
                         const pi_event *event_wait_list, pi_event *event) {
  return clEnqueueWriteBufferRect(
      command_queue, buffer, blocking_write, buffer_offset, host_offset, region,
      buffer_row_pitch, buffer_slice_pitch, host_row_pitch, host_slice_pitch,
      ptr, num_events_in_wait_list, event_wait_list, event);
}

pi_int piEnqueueReadBuffer(pi_command_queue command_queue, pi_mem buffer,
                           pi_bool blocking_read, size_t offset, size_t size,
                           void *ptr, pi_uint num_events_in_wait_list,
                           const pi_event *event_wait_list, pi_event *event) {
  return clEnqueueReadBuffer(command_queue, buffer, blocking_read, offset, size,
                             ptr, num_events_in_wait_list, event_wait_list,
                             event);
}

pi_int
piEnqueueReadBufferRect(pi_command_queue command_queue, pi_mem buffer,
                        pi_bool blocking_read, const size_t *buffer_offset,
                        const size_t *host_offset, const size_t *region,
                        size_t buffer_row_pitch, size_t buffer_slice_pitch,
                        size_t host_row_pitch, size_t host_slice_pitch,
                        void *ptr, pi_uint num_events_in_wait_list,
                        const pi_event *event_wait_list, pi_event *event) {
  return clEnqueueReadBufferRect(
      command_queue, buffer, blocking_read, buffer_offset, host_offset, region,
      buffer_row_pitch, buffer_slice_pitch, host_row_pitch, host_slice_pitch,
      ptr, num_events_in_wait_list, event_wait_list, event);
}

pi_int piEnqueueFillBuffer(pi_command_queue command_queue, pi_mem buffer,
                           const void *pattern, size_t pattern_size,
                           size_t offset, size_t size,
                           pi_uint num_events_in_wait_list,
                           const pi_event *event_wait_list, pi_event *event) {
  return clEnqueueFillBuffer(command_queue, buffer, pattern, pattern_size,
                             offset, size, num_events_in_wait_list,
                             event_wait_list, event);
}

pi_int piRetainContext(pi_context context) { return clRetainContext(context); }

pi_int piReleaseContext(pi_context context) {
  return clReleaseContext(context);
}

pi_int piGetKernelInfo(pi_kernel kernel, pi_kernel_info param_name,
                       size_t param_value_size, void *param_value,
                       size_t *param_value_size_ret) {
  return clGetKernelInfo(kernel, param_name, param_value_size, param_value,
                         param_value_size_ret);
}

pi_int piBuildProgram(pi_program program, pi_uint num_devices,
                      const pi_device_id *device_list, const char *options,
                      void(PI_CALLBACK *pfn_notify)(pi_program program,
                                                    void *user_data),
                      void *user_data) {
  return clBuildProgram(program, num_devices, device_list, options, pfn_notify,
                        user_data);
}

pi_int piRetainCommandQueue(pi_command_queue command_queue) {
  return clRetainCommandQueue(command_queue);
}

pi_int piReleaseCommandQueue(pi_command_queue command_queue) {
  return clReleaseCommandQueue(command_queue);
}

pi_int piFinish(pi_command_queue command_queue) {
  return clFinish(command_queue);
}

pi_int piGetKernelWorkGroupInfo(pi_kernel kernel, pi_device_id device,
                                pi_kernel_work_group_info param_name,
                                size_t param_value_size, void *param_value,
                                size_t *param_value_size_ret) {
  return clGetKernelWorkGroupInfo(kernel, device, param_name, param_value_size,
                                  param_value, param_value_size_ret);
}

pi_int piGetKernelSubGroupInfo(pi_kernel kernel, pi_device_id device,
                               pi_kernel_sub_group_info param_name,
                               size_t input_value_size, const void *input_value,
                               size_t param_value_size, void *param_value,
                               size_t *param_value_size_ret) {
  return clGetKernelSubGroupInfo(kernel, device, param_name, input_value_size,
                                 input_value, param_value_size, param_value,
                                 param_value_size_ret);
}

pi_int piGetMemObjectInfo(pi_mem memobj, pi_mem_info param_name,
                          size_t param_value_size, void *param_value,
                          size_t *param_value_size_ret) {
  return clGetMemObjectInfo(memobj, param_name, param_value_size, param_value,
                            param_value_size_ret);
}

pi_int piRetainMemObject(pi_mem memobj) { return clRetainMemObject(memobj); }

pi_int piReleaseMemObject(pi_mem memobj) { return clReleaseMemObject(memobj); }

pi_int piCreateSubDevices(pi_device_id in_device,
                          const pi_device_partition_property *properties,
                          pi_uint num_devices, pi_device_id *out_devices,
                          pi_uint *num_devices_ret) {
  return clCreateSubDevices(in_device, properties, num_devices, out_devices,
                            num_devices_ret);
}

pi_int piRetainDevice(pi_device_id device) { return clRetainDevice(device); }

pi_int piReleaseDevice(pi_device_id device) { return clReleaseDevice(device); }

pi_int piRetainProgram(pi_program program) { return clRetainProgram(program); }

pi_int piReleaseProgram(pi_program program) {
  return clReleaseProgram(program);
}

pi_int piCompileProgram(
    pi_program program, pi_uint num_devices, const pi_device_id *device_list,
    const char *options, pi_uint num_input_headers,
    const pi_program *input_headers, const char **header_inpiude_names,
    void(PI_CALLBACK *pfn_notify)(pi_program program, void *user_data),
    void *user_data) {
  return clCompileProgram(program, num_devices, device_list, options,
                          num_input_headers, input_headers,
                          header_inpiude_names, pfn_notify, user_data);
}

pi_int piGetEventInfo(pi_event event, pi_event_info param_name,
                      size_t param_value_size, void *param_value,
                      size_t *param_value_size_ret) {
  return clGetEventInfo(event, param_name, param_value_size, param_value,
                        param_value_size_ret);
}

pi_event piCreateUserEvent(pi_context context, pi_int *errcode_ret) {
  return clCreateUserEvent(context, errcode_ret);
}

pi_int piRetainEvent(pi_event event) { return clRetainEvent(event); }

pi_int piReleaseEvent(pi_event event) { return clReleaseEvent(event); }

pi_int piSetUserEventStatus(pi_event event, pi_int execution_status) {
  return clSetUserEventStatus(event, execution_status);
}

pi_int piSetEventCallback(
    pi_event event, pi_int command_exec_callback_type,
    void(PI_CALLBACK *pfn_notify)(pi_event event, pi_int event_command_status,
                                  void *user_data),
    void *user_data) {
  return clSetEventCallback(event, command_exec_callback_type, pfn_notify,
                            user_data);
}

pi_int piGetEventProfilingInfo(pi_event event, pi_profiling_info param_name,
                               size_t param_value_size, void *param_value,
                               size_t *param_value_size_ret) {
  return clGetEventProfilingInfo(event, param_name, param_value_size,
                                 param_value, param_value_size_ret);
}

pi_int piFlush(pi_command_queue command_queue) {
  return clFlush(command_queue);
}

pi_int piEnqueueCopyBuffer(pi_command_queue command_queue, pi_mem src_buffer,
                           pi_mem dst_buffer, size_t src_offset,
                           size_t dst_offset, size_t size,
                           pi_uint num_events_in_wait_list,
                           const pi_event *event_wait_list, pi_event *event) {
  return clEnqueueCopyBuffer(command_queue, src_buffer, dst_buffer, src_offset,
                             dst_offset, size, num_events_in_wait_list,
                             event_wait_list, event);
}

pi_int piEnqueueCopyBufferRect(
    pi_command_queue command_queue, pi_mem src_buffer, pi_mem dst_buffer,
    const size_t *src_origin, const size_t *dst_origin, const size_t *region,
    size_t src_row_pitch, size_t src_slice_pitch, size_t dst_row_pitch,
    size_t dst_slice_pitch, pi_uint num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event) {
  return clEnqueueCopyBufferRect(
      command_queue, src_buffer, dst_buffer, src_origin, dst_origin, region,
      src_row_pitch, src_slice_pitch, dst_row_pitch, dst_slice_pitch,
      num_events_in_wait_list, event_wait_list, event);
}
