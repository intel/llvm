#if INTEL_COLLAB
//===--- Target RTLs Implementation ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Code for tracing RTL
//
//===----------------------------------------------------------------------===//
#ifndef RTL_TRACE_H
#define RTL_TRACE_H

#include <CL/cl.h>
#include <CL/cl_ext_intel.h>
#include <inttypes.h>
#include <string>
#include "omptarget.h"

#define STR(x) #x
#define TO_STRING(x) STR(x)

extern int DebugLevel;

#define IDPLEVEL(Level, ...)                                                   \
  do {                                                                         \
    if (DebugLevel > Level) {                                                  \
      fprintf(stderr, "Target OPENCL RTL --> ");                               \
      fprintf(stderr, __VA_ARGS__);                                            \
    }                                                                          \
  } while (0)

#define IDP(...) IDPLEVEL(0, __VA_ARGS__)
#define IDP1(...) IDPLEVEL(1, __VA_ARGS__)

typedef cl_int (CL_API_CALL *clSetProgramSpecializationConstant_fn)(
    cl_program, cl_uint, size_t, const void *);

#define FOR_EACH_EXTENSION_FN(M)                                               \
  M(clGetMemAllocInfoINTEL)                                                    \
  M(clHostMemAllocINTEL)                                                       \
  M(clDeviceMemAllocINTEL)                                                     \
  M(clSharedMemAllocINTEL)                                                     \
  M(clMemFreeINTEL)                                                            \
  M(clSetKernelArgMemPointerINTEL)                                             \
  M(clEnqueueMemcpyINTEL)                                                      \
  M(clSetProgramSpecializationConstant)

enum ExtensionIdTy {
#define EXTENSION_FN_ID(Fn) Fn##Id,
  FOR_EACH_EXTENSION_FN(EXTENSION_FN_ID)
  ExtensionIdLast
};

#define FOREACH_CL_ERROR_CODE(FN)                                              \
  FN(CL_SUCCESS)                                                               \
  FN(CL_DEVICE_NOT_FOUND)                                                      \
  FN(CL_DEVICE_NOT_AVAILABLE)                                                  \
  FN(CL_COMPILER_NOT_AVAILABLE)                                                \
  FN(CL_MEM_OBJECT_ALLOCATION_FAILURE)                                         \
  FN(CL_OUT_OF_RESOURCES)                                                      \
  FN(CL_OUT_OF_HOST_MEMORY)                                                    \
  FN(CL_PROFILING_INFO_NOT_AVAILABLE)                                          \
  FN(CL_MEM_COPY_OVERLAP)                                                      \
  FN(CL_IMAGE_FORMAT_MISMATCH)                                                 \
  FN(CL_IMAGE_FORMAT_NOT_SUPPORTED)                                            \
  FN(CL_BUILD_PROGRAM_FAILURE)                                                 \
  FN(CL_MAP_FAILURE)                                                           \
  FN(CL_MISALIGNED_SUB_BUFFER_OFFSET)                                          \
  FN(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST)                             \
  FN(CL_COMPILE_PROGRAM_FAILURE)                                               \
  FN(CL_LINKER_NOT_AVAILABLE)                                                  \
  FN(CL_LINK_PROGRAM_FAILURE)                                                  \
  FN(CL_DEVICE_PARTITION_FAILED)                                               \
  FN(CL_KERNEL_ARG_INFO_NOT_AVAILABLE)                                         \
  FN(CL_INVALID_VALUE)                                                         \
  FN(CL_INVALID_DEVICE_TYPE)                                                   \
  FN(CL_INVALID_PLATFORM)                                                      \
  FN(CL_INVALID_DEVICE)                                                        \
  FN(CL_INVALID_CONTEXT)                                                       \
  FN(CL_INVALID_QUEUE_PROPERTIES)                                              \
  FN(CL_INVALID_COMMAND_QUEUE)                                                 \
  FN(CL_INVALID_HOST_PTR)                                                      \
  FN(CL_INVALID_MEM_OBJECT)                                                    \
  FN(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR)                                       \
  FN(CL_INVALID_IMAGE_SIZE)                                                    \
  FN(CL_INVALID_SAMPLER)                                                       \
  FN(CL_INVALID_BINARY)                                                        \
  FN(CL_INVALID_BUILD_OPTIONS)                                                 \
  FN(CL_INVALID_PROGRAM)                                                       \
  FN(CL_INVALID_PROGRAM_EXECUTABLE)                                            \
  FN(CL_INVALID_KERNEL_NAME)                                                   \
  FN(CL_INVALID_KERNEL_DEFINITION)                                             \
  FN(CL_INVALID_KERNEL)                                                        \
  FN(CL_INVALID_ARG_INDEX)                                                     \
  FN(CL_INVALID_ARG_VALUE)                                                     \
  FN(CL_INVALID_ARG_SIZE)                                                      \
  FN(CL_INVALID_KERNEL_ARGS)                                                   \
  FN(CL_INVALID_WORK_DIMENSION)                                                \
  FN(CL_INVALID_WORK_GROUP_SIZE)                                               \
  FN(CL_INVALID_WORK_ITEM_SIZE)                                                \
  FN(CL_INVALID_GLOBAL_OFFSET)                                                 \
  FN(CL_INVALID_EVENT_WAIT_LIST)                                               \
  FN(CL_INVALID_EVENT)                                                         \
  FN(CL_INVALID_OPERATION)                                                     \
  FN(CL_INVALID_GL_OBJECT)                                                     \
  FN(CL_INVALID_BUFFER_SIZE)                                                   \
  FN(CL_INVALID_MIP_LEVEL)                                                     \
  FN(CL_INVALID_GLOBAL_WORK_SIZE)                                              \
  FN(CL_INVALID_PROPERTY)                                                      \
  FN(CL_INVALID_IMAGE_DESCRIPTOR)                                              \
  FN(CL_INVALID_COMPILER_OPTIONS)                                              \
  FN(CL_INVALID_LINKER_OPTIONS)                                                \
  FN(CL_INVALID_DEVICE_PARTITION_COUNT)                                        \
  FN(CL_INVALID_PIPE_SIZE)                                                     \
  FN(CL_INVALID_DEVICE_QUEUE)

#define CASE_TO_STRING(s) case s: return #s;

static const char *getCLErrorName(int error) {
  switch (error) {
    FOREACH_CL_ERROR_CODE(CASE_TO_STRING)
  default:
    return "Unknown Error";
  }
}

#define FATAL_ERROR(msg)                                                       \
  do {                                                                         \
    IDPLEVEL(-1, "Error: %s failed (%s) -- exiting...\n", __func__, msg);      \
    exit(EXIT_FAILURE);                                                        \
  } while (0)

#define WARNING(...) IDPLEVEL(-1, "Warning: " __VA_ARGS__)

#define TRACE_FN(Name) CLTR##Name
#define TRACE_FN_ARG_BEGIN()                                                   \
  do {                                                                         \
    std::string fn(__func__);                                                  \
    IDP1("CL_CALLEE: %s (\n", fn.substr(4).c_str());                           \
  } while (0)
#define TRACE_FN_ARG_END() IDP1(")\n")
#define TRACE_FN_ARG(Arg, Fmt) IDP1("    %s = " Fmt "\n", TO_STRING(Arg), Arg)
#define TRACE_FN_ARG_PTR(Arg)                                                  \
  IDP1("    %s = " DPxMOD "\n", TO_STRING(Arg), DPxPTR(Arg))
#define TRACE_FN_ARG_INT(Arg) TRACE_FN_ARG(Arg, "%" PRId32)
#define TRACE_FN_ARG_SIZE(Arg) TRACE_FN_ARG(Arg, "%zu")
#define TRACE_FN_ARG_UINT(Arg) TRACE_FN_ARG(Arg, "%" PRIu32)
#define TRACE_FN_ARG_ULONG(Arg) TRACE_FN_ARG(Arg, "%" PRIu64)

cl_int TRACE_FN(clCompileProgram)(
    cl_program program,
    cl_uint num_devices,
    const cl_device_id *device_list,
    const char *options,
    cl_uint num_input_headers,
    const cl_program *input_headers,
    const char **header_include_names,
    void (CL_CALLBACK *pfn_notify)(cl_program, void *),
    void *user_data) {
  auto rc = clCompileProgram(program, num_devices, device_list, options,
      num_input_headers, input_headers, header_include_names, pfn_notify,
      user_data);
  TRACE_FN_ARG_BEGIN();
  TRACE_FN_ARG_PTR(program);
  TRACE_FN_ARG_UINT(num_devices);
  TRACE_FN_ARG_PTR(device_list);
  TRACE_FN_ARG_PTR(options);
  TRACE_FN_ARG_UINT(num_input_headers);
  TRACE_FN_ARG_PTR(input_headers);
  TRACE_FN_ARG_PTR(header_include_names);
  TRACE_FN_ARG_PTR(pfn_notify);
  TRACE_FN_ARG_PTR(user_data);
  TRACE_FN_ARG_END();
  return rc;
}

cl_int TRACE_FN(clBuildProgram)(
    cl_program program,
    cl_uint num_devices,
    const cl_device_id *device_list,
    const char *options,
    void (CL_CALLBACK *pfn_notify)(cl_program, void *),
    void *user_data) {
  auto rc = clBuildProgram(program, num_devices, device_list, options,
                           pfn_notify, user_data);
  TRACE_FN_ARG_BEGIN();
  TRACE_FN_ARG_PTR(program);
  TRACE_FN_ARG_UINT(num_devices);
  TRACE_FN_ARG_PTR(device_list);
  TRACE_FN_ARG_PTR(options);
  TRACE_FN_ARG_PTR(pfn_notify);
  TRACE_FN_ARG_PTR(user_data);
  TRACE_FN_ARG_END();
  return rc;
}

cl_mem TRACE_FN(clCreateBuffer)(
    cl_context context,
    cl_mem_flags flags,
    size_t size,
    void *host_ptr,
    cl_int *errcode_ret) {
  auto ret = clCreateBuffer(context, flags, size, host_ptr, errcode_ret);
  TRACE_FN_ARG_BEGIN();
  TRACE_FN_ARG_PTR(context);
  TRACE_FN_ARG_ULONG(flags);
  TRACE_FN_ARG_SIZE(size);
  TRACE_FN_ARG_PTR(host_ptr);
  TRACE_FN_ARG_PTR(errcode_ret);
  TRACE_FN_ARG_END();
  return ret;
}

cl_command_queue TRACE_FN(clCreateCommandQueueWithProperties)(
    cl_context context,
    cl_device_id device,
    const cl_queue_properties *properties,
    cl_int *errcode_ret) {
  auto ret = clCreateCommandQueueWithProperties(context, device, properties,
                                                errcode_ret);
  TRACE_FN_ARG_BEGIN();
  TRACE_FN_ARG_PTR(context);
  TRACE_FN_ARG_PTR(device);
  TRACE_FN_ARG_PTR(properties);
  TRACE_FN_ARG_PTR(errcode_ret);
  TRACE_FN_ARG_END();
  return ret;
}

cl_context TRACE_FN(clCreateContext)(
    const cl_context_properties *properties,
    cl_uint num_devices,
    const cl_device_id *devices,
    void (CL_CALLBACK *pfn_notify)(const char *, const void *, size_t, void *),
    void *user_data,
    cl_int *errcode_ret) {
  auto ret = clCreateContext(properties, num_devices, devices, pfn_notify,
                             user_data, errcode_ret);
  TRACE_FN_ARG_BEGIN();
  TRACE_FN_ARG_PTR(properties);
  TRACE_FN_ARG_UINT(num_devices);
  TRACE_FN_ARG_PTR(devices);
  TRACE_FN_ARG_PTR(pfn_notify);
  TRACE_FN_ARG_PTR(user_data);
  TRACE_FN_ARG_PTR(errcode_ret);
  TRACE_FN_ARG_END();
  return ret;
}

cl_kernel TRACE_FN(clCreateKernel)(
    cl_program program,
    const char *kernel_name,
    cl_int *errcode_ret) {
  auto ret = clCreateKernel(program, kernel_name, errcode_ret);
  TRACE_FN_ARG_BEGIN();
  TRACE_FN_ARG_PTR(program);
  TRACE_FN_ARG_PTR(kernel_name);
  TRACE_FN_ARG_PTR(errcode_ret);
  TRACE_FN_ARG_END();
  return ret;
}

cl_program TRACE_FN(clCreateProgramWithIL)(
    cl_context context,
    const void *il,
    size_t length,
    cl_int *errcode_ret) {
  auto ret = clCreateProgramWithIL(context, il, length, errcode_ret);
  TRACE_FN_ARG_BEGIN();
  TRACE_FN_ARG_PTR(context);
  TRACE_FN_ARG_PTR(il);
  TRACE_FN_ARG_SIZE(length);
  TRACE_FN_ARG_PTR(errcode_ret);
  TRACE_FN_ARG_END();
  return ret;
}

cl_int TRACE_FN(clEnqueueBarrierWithWaitList)(
    cl_command_queue command_queue,
    cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list,
    cl_event *event) {
  auto rc = clEnqueueBarrierWithWaitList(command_queue, num_events_in_wait_list,
                                         event_wait_list, event);
  TRACE_FN_ARG_BEGIN();
  TRACE_FN_ARG_PTR(command_queue);
  TRACE_FN_ARG_UINT(num_events_in_wait_list);
  TRACE_FN_ARG_PTR(event_wait_list);
  TRACE_FN_ARG_PTR(event);
  TRACE_FN_ARG_END();
  return rc;
}

cl_int TRACE_FN(clEnqueueNDRangeKernel)(
    cl_command_queue command_queue,
    cl_kernel kernel,
    cl_uint work_dim,
    const size_t *global_work_offset,
    const size_t *global_work_size,
    const size_t *local_work_size,
    cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list,
    cl_event *event) {
  auto rc = clEnqueueNDRangeKernel(command_queue, kernel, work_dim,
      global_work_offset, global_work_size, local_work_size,
      num_events_in_wait_list, event_wait_list, event);
  TRACE_FN_ARG_BEGIN();
  TRACE_FN_ARG_PTR(command_queue);
  TRACE_FN_ARG_PTR(kernel);
  TRACE_FN_ARG_UINT(work_dim);
  TRACE_FN_ARG_PTR(global_work_offset);
  TRACE_FN_ARG_PTR(global_work_size);
  TRACE_FN_ARG_PTR(local_work_size);
  TRACE_FN_ARG_UINT(num_events_in_wait_list);
  TRACE_FN_ARG_PTR(event_wait_list);
  TRACE_FN_ARG_PTR(event);
  TRACE_FN_ARG_END();
  return rc;
}

cl_int TRACE_FN(clEnqueueReadBuffer)(
    cl_command_queue command_queue,
    cl_mem buffer,
    cl_bool blocking_read,
    size_t offset,
    size_t size,
    void *ptr,
    cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list,
    cl_event *event) {
  auto rc = clEnqueueReadBuffer(command_queue, buffer, blocking_read, offset,
      size, ptr, num_events_in_wait_list, event_wait_list, event);
  TRACE_FN_ARG_BEGIN();
  TRACE_FN_ARG_PTR(command_queue);
  TRACE_FN_ARG_PTR(buffer);
  TRACE_FN_ARG_UINT(blocking_read);
  TRACE_FN_ARG_SIZE(offset);
  TRACE_FN_ARG_SIZE(size);
  TRACE_FN_ARG_PTR(ptr);
  TRACE_FN_ARG_UINT(num_events_in_wait_list);
  TRACE_FN_ARG_PTR(event_wait_list);
  TRACE_FN_ARG_PTR(event);
  TRACE_FN_ARG_END();
  return rc;
}

cl_int TRACE_FN(clEnqueueSVMMap)(
    cl_command_queue command_queue,
    cl_bool blocking_map,
    cl_map_flags flags,
    void *svm_ptr,
    size_t size,
    cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list,
    cl_event *event) {
  auto rc = clEnqueueSVMMap(command_queue, blocking_map, flags, svm_ptr, size,
                            num_events_in_wait_list, event_wait_list, event);
  TRACE_FN_ARG_BEGIN();
  TRACE_FN_ARG_PTR(command_queue);
  TRACE_FN_ARG_UINT(blocking_map);
  TRACE_FN_ARG_ULONG(flags);
  TRACE_FN_ARG_PTR(svm_ptr);
  TRACE_FN_ARG_SIZE(size);
  TRACE_FN_ARG_UINT(num_events_in_wait_list);
  TRACE_FN_ARG_PTR(event_wait_list);
  TRACE_FN_ARG_PTR(event);
  TRACE_FN_ARG_END();
  return rc;
}

cl_int TRACE_FN(clEnqueueSVMMemcpy)(
    cl_command_queue command_queue,
    cl_bool blocking_copy,
    void *dst_ptr,
    const void *src_ptr,
    size_t size,
    cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list,
    cl_event *event) {
  auto rc = clEnqueueSVMMemcpy(command_queue, blocking_copy, dst_ptr, src_ptr,
      size, num_events_in_wait_list, event_wait_list, event);
  TRACE_FN_ARG_BEGIN();
  TRACE_FN_ARG_PTR(command_queue);
  TRACE_FN_ARG_UINT(blocking_copy);
  TRACE_FN_ARG_PTR(dst_ptr);
  TRACE_FN_ARG_PTR(src_ptr);
  TRACE_FN_ARG_SIZE(size);
  TRACE_FN_ARG_UINT(num_events_in_wait_list);
  TRACE_FN_ARG_PTR(event_wait_list);
  TRACE_FN_ARG_PTR(event);
  TRACE_FN_ARG_END();
  return rc;
}

cl_int TRACE_FN(clEnqueueMemcpyINTEL)(
    clEnqueueMemcpyINTEL_fn funcptr,
    cl_command_queue command_queue,
    cl_bool blocking,
    void *dst_ptr,
    const void *src_ptr,
    size_t size,
    cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list,
    cl_event *event) {
  auto rc = (*funcptr)(command_queue, blocking, dst_ptr, src_ptr, size,
                       num_events_in_wait_list, event_wait_list, event);
  TRACE_FN_ARG_BEGIN();
  TRACE_FN_ARG_PTR(command_queue);
  TRACE_FN_ARG_UINT(blocking);
  TRACE_FN_ARG_PTR(dst_ptr);
  TRACE_FN_ARG_PTR(src_ptr);
  TRACE_FN_ARG_SIZE(size);
  TRACE_FN_ARG_UINT(num_events_in_wait_list);
  TRACE_FN_ARG_PTR(event_wait_list);
  TRACE_FN_ARG_PTR(event);
  TRACE_FN_ARG_END();
  return rc;
}

cl_int TRACE_FN(clEnqueueSVMUnmap)(
    cl_command_queue command_queue,
    void *svm_ptr,
    cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list,
    cl_event *event) {
  auto rc = clEnqueueSVMUnmap(command_queue, svm_ptr, num_events_in_wait_list,
                              event_wait_list, event);
  TRACE_FN_ARG_BEGIN();
  TRACE_FN_ARG_PTR(command_queue);
  TRACE_FN_ARG_PTR(svm_ptr);
  TRACE_FN_ARG_UINT(num_events_in_wait_list);
  TRACE_FN_ARG_PTR(event_wait_list);
  TRACE_FN_ARG_PTR(event);
  TRACE_FN_ARG_END();
  return rc;
}

cl_int TRACE_FN(clEnqueueWriteBuffer)(
    cl_command_queue command_queue,
    cl_mem buffer,
    cl_bool blocking_write,
    size_t offset,
    size_t size,
    const void *ptr,
    cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list,
    cl_event *event) {
  auto rc = clEnqueueWriteBuffer(command_queue, buffer, blocking_write, offset,
      size, ptr, num_events_in_wait_list, event_wait_list, event);
  TRACE_FN_ARG_BEGIN();
  TRACE_FN_ARG_PTR(command_queue);
  TRACE_FN_ARG_PTR(buffer);
  TRACE_FN_ARG_UINT(blocking_write);
  TRACE_FN_ARG_SIZE(offset);
  TRACE_FN_ARG_SIZE(size);
  TRACE_FN_ARG_PTR(ptr);
  TRACE_FN_ARG_UINT(num_events_in_wait_list);
  TRACE_FN_ARG_PTR(event_wait_list);
  TRACE_FN_ARG_PTR(event);
  TRACE_FN_ARG_END();
  return rc;
}

cl_int TRACE_FN(clFinish)(
    cl_command_queue command_queue) {
  auto rc = clFinish(command_queue);
  TRACE_FN_ARG_BEGIN();
  TRACE_FN_ARG_PTR(command_queue);
  TRACE_FN_ARG_END();
  return rc;
}

cl_int TRACE_FN(clGetDeviceAndHostTimer)(
    cl_device_id device,
    cl_ulong *device_timestamp,
    cl_ulong *host_timestamp) {
  auto rc = clGetDeviceAndHostTimer(device, device_timestamp, host_timestamp);
  TRACE_FN_ARG_BEGIN();
  TRACE_FN_ARG_PTR(device);
  TRACE_FN_ARG_PTR(device_timestamp);
  TRACE_FN_ARG_PTR(host_timestamp);
  TRACE_FN_ARG_END();
  return rc;
}

cl_int TRACE_FN(clGetDeviceIDs)(
    cl_platform_id platform,
    cl_device_type device_type,
    cl_uint num_entries,
    cl_device_id *devices,
    cl_uint *num_devices) {
  auto rc = clGetDeviceIDs(platform, device_type, num_entries, devices,
                           num_devices);
  TRACE_FN_ARG_BEGIN();
  TRACE_FN_ARG_PTR(platform);
  TRACE_FN_ARG_ULONG(device_type);
  TRACE_FN_ARG_UINT(num_entries);
  TRACE_FN_ARG_PTR(devices);
  TRACE_FN_ARG_PTR(num_devices);
  TRACE_FN_ARG_END();
  return rc;
}

cl_int TRACE_FN(clGetDeviceInfo)(
    cl_device_id device,
    cl_device_info param_name,
    size_t param_value_size,
    void *param_value,
    size_t *param_value_size_ret) {
  auto rc = clGetDeviceInfo(device, param_name, param_value_size, param_value,
                            param_value_size_ret);
  TRACE_FN_ARG_BEGIN();
  TRACE_FN_ARG_PTR(device);
  TRACE_FN_ARG_UINT(param_name);
  TRACE_FN_ARG_SIZE(param_value_size);
  TRACE_FN_ARG_PTR(param_value);
  TRACE_FN_ARG_PTR(param_value_size_ret);
  TRACE_FN_ARG_END();
  return rc;
}

cl_int TRACE_FN(clGetEventInfo)(
    cl_event event,
    cl_event_info param_name,
    size_t param_value_size,
    void *param_value,
    size_t *param_value_size_ret) {
  auto rc = clGetEventInfo(event, param_name, param_value_size, param_value,
                           param_value_size_ret);
  TRACE_FN_ARG_BEGIN();
  TRACE_FN_ARG_PTR(event);
  TRACE_FN_ARG_UINT(param_name);
  TRACE_FN_ARG_SIZE(param_value_size);
  TRACE_FN_ARG_PTR(param_value);
  TRACE_FN_ARG_PTR(param_value_size_ret);
  TRACE_FN_ARG_END();
  return rc;
}

cl_int TRACE_FN(clGetEventProfilingInfo)(
    cl_event event,
    cl_profiling_info param_name,
    size_t param_value_size,
    void *param_value,
    size_t *param_value_size_ret) {
  auto rc = clGetEventProfilingInfo(event, param_name, param_value_size,
                                    param_value, param_value_size_ret);
  TRACE_FN_ARG_BEGIN();
  TRACE_FN_ARG_PTR(event);
  TRACE_FN_ARG_UINT(param_name);
  TRACE_FN_ARG_SIZE(param_value_size);
  TRACE_FN_ARG_PTR(param_value);
  TRACE_FN_ARG_PTR(param_value_size_ret);
  TRACE_FN_ARG_END();
  return rc;
}

void *TRACE_FN(clGetExtensionFunctionAddressForPlatform)(
    cl_platform_id platform,
    const char *funcname) {
  auto ret = clGetExtensionFunctionAddressForPlatform(platform, funcname);
  TRACE_FN_ARG_BEGIN();
  TRACE_FN_ARG_PTR(platform);
  TRACE_FN_ARG_PTR(funcname);
  TRACE_FN_ARG_END();
  return ret;
}

cl_int TRACE_FN(clGetKernelArgInfo)(
    cl_kernel kernel,
    cl_uint arg_index,
    cl_kernel_arg_info param_name,
    size_t param_value_size,
    void *param_value,
    size_t *param_value_size_ret) {
  auto rc = clGetKernelArgInfo(kernel, arg_index, param_name, param_value_size,
                               param_value, param_value_size_ret);
  TRACE_FN_ARG_BEGIN();
  TRACE_FN_ARG_PTR(kernel);
  TRACE_FN_ARG_UINT(arg_index);
  TRACE_FN_ARG_UINT(param_name);
  TRACE_FN_ARG_SIZE(param_value_size);
  TRACE_FN_ARG_PTR(param_value);
  TRACE_FN_ARG_PTR(param_value_size_ret);
  TRACE_FN_ARG_END();
  return rc;
}

cl_int TRACE_FN(clGetKernelInfo)(
    cl_kernel kernel,
    cl_kernel_info param_name,
    size_t param_value_size,
    void *param_value,
    size_t *param_value_size_ret) {
  auto rc = clGetKernelInfo(kernel, param_name, param_value_size, param_value,
                            param_value_size_ret);
  TRACE_FN_ARG_BEGIN();
  TRACE_FN_ARG_PTR(kernel);
  TRACE_FN_ARG_UINT(param_name);
  TRACE_FN_ARG_SIZE(param_value_size);
  TRACE_FN_ARG_PTR(param_value);
  TRACE_FN_ARG_PTR(param_value_size_ret);
  TRACE_FN_ARG_END();
  return rc;
}

cl_int TRACE_FN(clGetKernelWorkGroupInfo)(
    cl_kernel kernel,
    cl_device_id device,
    cl_kernel_work_group_info param_name,
    size_t param_value_size,
    void *param_value,
    size_t *param_value_size_ret) {
  auto rc = clGetKernelWorkGroupInfo(kernel, device, param_name,
      param_value_size, param_value, param_value_size_ret);
  TRACE_FN_ARG_BEGIN();
  TRACE_FN_ARG_PTR(kernel);
  TRACE_FN_ARG_PTR(device);
  TRACE_FN_ARG_UINT(param_name);
  TRACE_FN_ARG_SIZE(param_value_size);
  TRACE_FN_ARG_PTR(param_value);
  TRACE_FN_ARG_PTR(param_value_size_ret);
  TRACE_FN_ARG_END();
  return rc;
}

cl_int TRACE_FN(clGetMemAllocInfoINTEL)(
    clGetMemAllocInfoINTEL_fn funcptr,
    cl_context context,
    const void *ptr,
    cl_mem_info_intel param_name,
    size_t param_value_size,
    void *param_value,
    size_t *param_value_size_ret) {
  auto rc = (*funcptr)(context, ptr, param_name, param_value_size, param_value,
                       param_value_size_ret);
  TRACE_FN_ARG_BEGIN();
  TRACE_FN_ARG_PTR(context);
  TRACE_FN_ARG_PTR(ptr);
  TRACE_FN_ARG_UINT(param_name);
  TRACE_FN_ARG_SIZE(param_value_size);
  TRACE_FN_ARG_PTR(param_value);
  TRACE_FN_ARG_PTR(param_value_size_ret);
  TRACE_FN_ARG_END();
  return rc;
}

cl_int TRACE_FN(clGetPlatformIDs)(
    cl_uint num_entries,
    cl_platform_id *platforms,
    cl_uint *num_platforms) {
  auto rc = clGetPlatformIDs(num_entries, platforms, num_platforms);
  TRACE_FN_ARG_BEGIN();
  TRACE_FN_ARG_UINT(num_entries);
  TRACE_FN_ARG_PTR(platforms);
  TRACE_FN_ARG_PTR(num_platforms);
  TRACE_FN_ARG_END();
  return rc;
}

cl_int TRACE_FN(clGetPlatformInfo)(
    cl_platform_id platform,
    cl_platform_info param_name,
    size_t param_value_size,
    void *param_value,
    size_t *param_value_size_ret) {
  auto rc = clGetPlatformInfo(platform, param_name, param_value_size,
                              param_value, param_value_size_ret);
  TRACE_FN_ARG_BEGIN();
  TRACE_FN_ARG_PTR(platform);
  TRACE_FN_ARG_UINT(param_name);
  TRACE_FN_ARG_SIZE(param_value_size);
  TRACE_FN_ARG_PTR(param_value);
  TRACE_FN_ARG_PTR(param_value_size_ret);
  TRACE_FN_ARG_END();
  return rc;
}

cl_int TRACE_FN(clGetProgramBuildInfo)(
    cl_program program,
    cl_device_id device,
    cl_program_build_info param_name,
    size_t param_value_size,
    void *param_value,
    size_t *param_value_size_ret) {
  auto rc = clGetProgramBuildInfo(program, device, param_name, param_value_size,
                                  param_value, param_value_size_ret);
  TRACE_FN_ARG_BEGIN();
  TRACE_FN_ARG_PTR(program);
  TRACE_FN_ARG_PTR(device);
  TRACE_FN_ARG_UINT(param_name);
  TRACE_FN_ARG_SIZE(param_value_size);
  TRACE_FN_ARG_PTR(param_value);
  TRACE_FN_ARG_PTR(param_value_size_ret);
  TRACE_FN_ARG_END();
  return rc;
}

void *TRACE_FN(clHostMemAllocINTEL)(
    clHostMemAllocINTEL_fn funcptr,
    cl_context context,
    const cl_mem_properties_intel *properties,
    size_t size,
    cl_uint alignment,
    cl_int *errcode_ret) {
  auto ret = (*funcptr)(context, properties, size, alignment, errcode_ret);
  TRACE_FN_ARG_BEGIN();
  TRACE_FN_ARG_PTR(context);
  TRACE_FN_ARG_PTR(properties);
  TRACE_FN_ARG_SIZE(size);
  TRACE_FN_ARG_UINT(alignment);
  TRACE_FN_ARG_PTR(errcode_ret);
  TRACE_FN_ARG_END();
  return ret;
}

void *TRACE_FN(clDeviceMemAllocINTEL)(
    clDeviceMemAllocINTEL_fn funcptr,
    cl_context context,
    cl_device_id device,
    const cl_mem_properties_intel *properties,
    size_t size,
    cl_uint alignment,
    cl_int *errcode_ret) {
  auto ret = (*funcptr)(context, device, properties, size, alignment,
                        errcode_ret);
  TRACE_FN_ARG_BEGIN();
  TRACE_FN_ARG_PTR(context);
  TRACE_FN_ARG_PTR(device);
  TRACE_FN_ARG_PTR(properties);
  TRACE_FN_ARG_SIZE(size);
  TRACE_FN_ARG_UINT(alignment);
  TRACE_FN_ARG_PTR(errcode_ret);
  TRACE_FN_ARG_END();
  return ret;
}

cl_program TRACE_FN(clLinkProgram)(
    cl_context context,
    cl_uint num_devices,
    const cl_device_id *device_list,
    const char *options,
    cl_uint num_input_programs,
    const cl_program *input_programs,
    void (CL_CALLBACK *pfn_notify)(cl_program, void *),
    void *user_data,
    cl_int *errcode_ret) {
  auto ret = clLinkProgram(context, num_devices, device_list, options,
      num_input_programs, input_programs, pfn_notify, user_data, errcode_ret);
  TRACE_FN_ARG_BEGIN();
  TRACE_FN_ARG_PTR(context);
  TRACE_FN_ARG_UINT(num_devices);
  TRACE_FN_ARG_PTR(device_list);
  TRACE_FN_ARG_PTR(options);
  TRACE_FN_ARG_UINT(num_input_programs);
  TRACE_FN_ARG_PTR(input_programs);
  TRACE_FN_ARG_PTR(pfn_notify);
  TRACE_FN_ARG_PTR(user_data);
  TRACE_FN_ARG_END();
  return ret;
}

cl_program TRACE_FN(clCreateProgramWithBinary)(
    cl_context context,
    cl_uint num_devices,
    const cl_device_id *device_list,
    const size_t *lengths,
    const unsigned char **binaries,
    cl_int *binary_status,
    cl_int *errcode_ret) {
  auto ret = clCreateProgramWithBinary(context, num_devices,
                                       device_list, lengths,
                                       binaries, binary_status,
                                       errcode_ret);
  TRACE_FN_ARG_BEGIN();
  TRACE_FN_ARG_PTR(context);
  TRACE_FN_ARG_UINT(num_devices);
  TRACE_FN_ARG_PTR(device_list);
  TRACE_FN_ARG_PTR(lengths);
  TRACE_FN_ARG_PTR(binaries);
  TRACE_FN_ARG_PTR(binary_status);
  TRACE_FN_ARG_END();
  return ret;
}

cl_int TRACE_FN(clMemFreeINTEL)(
    clMemFreeINTEL_fn funcptr,
    cl_context context,
    void *ptr) {
  auto rc = (*funcptr)(context, ptr);
  TRACE_FN_ARG_BEGIN();
  TRACE_FN_ARG_PTR(context);
  TRACE_FN_ARG_PTR(ptr);
  TRACE_FN_ARG_END();
  return rc;
}

cl_int TRACE_FN(clReleaseCommandQueue)(
    cl_command_queue command_queue) {
  auto rc = clReleaseCommandQueue(command_queue);
  TRACE_FN_ARG_BEGIN();
  TRACE_FN_ARG_PTR(command_queue);
  TRACE_FN_ARG_END();
  return rc;
}

cl_int TRACE_FN(clReleaseContext)(
    cl_context context) {
  auto rc = clReleaseContext(context);
  TRACE_FN_ARG_BEGIN();
  TRACE_FN_ARG_PTR(context);
  TRACE_FN_ARG_END();
  return rc;
}

cl_int TRACE_FN(clReleaseKernel)(
    cl_kernel kernel) {
  auto rc = clReleaseKernel(kernel);
  TRACE_FN_ARG_BEGIN();
  TRACE_FN_ARG_PTR(kernel);
  TRACE_FN_ARG_END();
  return rc;
}

cl_int TRACE_FN(clReleaseMemObject)(
    cl_mem memobj) {
  auto rc = clReleaseMemObject(memobj);
  TRACE_FN_ARG_BEGIN();
  TRACE_FN_ARG_PTR(memobj);
  TRACE_FN_ARG_END();
  return rc;
}

cl_int TRACE_FN(clReleaseProgram)(
    cl_program program) {
  auto rc = clReleaseProgram(program);
  TRACE_FN_ARG_BEGIN();
  TRACE_FN_ARG_PTR(program);
  TRACE_FN_ARG_END();
  return rc;
}

cl_int TRACE_FN(clSetEventCallback)(
    cl_event event,
    cl_int command_exec_callback_type,
    void (CL_CALLBACK *pfn_notify)(cl_event, cl_int, void *),
    void *user_data) {
  auto rc = clSetEventCallback(event, command_exec_callback_type, pfn_notify,
                               user_data);
  TRACE_FN_ARG_BEGIN();
  TRACE_FN_ARG_PTR(event);
  TRACE_FN_ARG_INT(command_exec_callback_type);
  TRACE_FN_ARG_PTR(pfn_notify);
  TRACE_FN_ARG_PTR(user_data);
  TRACE_FN_ARG_END();
  return rc;
}

cl_int TRACE_FN(clSetKernelArg)(
    cl_kernel kernel,
    cl_uint arg_index,
    size_t arg_size,
    const void *arg_value) {
  auto rc = clSetKernelArg(kernel, arg_index, arg_size, arg_value);
  TRACE_FN_ARG_BEGIN();
  TRACE_FN_ARG_PTR(kernel);
  TRACE_FN_ARG_UINT(arg_index);
  TRACE_FN_ARG_SIZE(arg_size);
  TRACE_FN_ARG_PTR(arg_value);
  TRACE_FN_ARG_END();
  return rc;
}

cl_int TRACE_FN(clSetKernelArgSVMPointer)(
    cl_kernel kernel,
    cl_uint arg_index,
    const void *arg_value) {
  auto rc = clSetKernelArgSVMPointer(kernel, arg_index, arg_value);
  TRACE_FN_ARG_BEGIN();
  TRACE_FN_ARG_PTR(kernel);
  TRACE_FN_ARG_UINT(arg_index);
  TRACE_FN_ARG_PTR(arg_value);
  TRACE_FN_ARG_END();
  return rc;
}

cl_int TRACE_FN(clSetKernelArgMemPointerINTEL)(
    clSetKernelArgMemPointerINTEL_fn funcptr,
    cl_kernel kernel,
    cl_uint arg_index,
    const void *arg_value) {
  auto rc = (*funcptr)(kernel, arg_index, arg_value);
  TRACE_FN_ARG_BEGIN();
  TRACE_FN_ARG_PTR(kernel);
  TRACE_FN_ARG_UINT(arg_index);
  TRACE_FN_ARG_PTR(arg_value);
  TRACE_FN_ARG_END();
  return rc;
}

cl_int TRACE_FN(clSetKernelExecInfo)(
    cl_kernel kernel,
    cl_kernel_exec_info param_name,
    size_t param_value_size,
    const void *param_value) {
  auto rc = clSetKernelExecInfo(kernel, param_name, param_value_size,
                                param_value);
  TRACE_FN_ARG_BEGIN();
  TRACE_FN_ARG_PTR(kernel);
  TRACE_FN_ARG_UINT(param_name);
  TRACE_FN_ARG_SIZE(param_value_size);
  TRACE_FN_ARG_PTR(param_value);
  TRACE_FN_ARG_END();
  return rc;
}

void *TRACE_FN(clSharedMemAllocINTEL)(
    clSharedMemAllocINTEL_fn funcptr,
    cl_context context,
    cl_device_id device,
    const cl_mem_properties_intel *properties,
    size_t size,
    cl_uint alignment,
    cl_int *errcode_ret) {
  auto ret = (*funcptr)(context, device, properties, size, alignment,
                        errcode_ret);
  TRACE_FN_ARG_BEGIN();
  TRACE_FN_ARG_PTR(context);
  TRACE_FN_ARG_PTR(device);
  TRACE_FN_ARG_PTR(properties);
  TRACE_FN_ARG_SIZE(size);
  TRACE_FN_ARG_UINT(alignment);
  TRACE_FN_ARG_PTR(errcode_ret);
  TRACE_FN_ARG_END();
  return ret;
}

void *TRACE_FN(clSVMAlloc)(
    cl_context context,
    cl_svm_mem_flags flags,
    size_t size,
    cl_uint alignment) {
  auto ret = clSVMAlloc(context, flags, size, alignment);
  TRACE_FN_ARG_BEGIN();
  TRACE_FN_ARG_PTR(context);
  TRACE_FN_ARG_ULONG(flags);
  TRACE_FN_ARG_SIZE(size);
  TRACE_FN_ARG_UINT(alignment);
  TRACE_FN_ARG_END();
  return ret;
}

void TRACE_FN(clSVMFree)(
    cl_context context,
    void *svm_pointer) {
  clSVMFree(context, svm_pointer);
  TRACE_FN_ARG_BEGIN();
  TRACE_FN_ARG_PTR(context);
  TRACE_FN_ARG_PTR(svm_pointer);
  TRACE_FN_ARG_END();
}

cl_int TRACE_FN(clWaitForEvents)(
    cl_uint num_events,
    const cl_event *event_list) {
  auto rc = clWaitForEvents(num_events, event_list);
  TRACE_FN_ARG_BEGIN();
  TRACE_FN_ARG_UINT(num_events);
  TRACE_FN_ARG_PTR(event_list);
  TRACE_FN_ARG_END();
  return rc;
}

cl_int TRACE_FN(clSetProgramSpecializationConstant)(
    clSetProgramSpecializationConstant_fn funcptr,
    cl_program program,
    cl_uint spec_id,
    size_t spec_size,
    const void* spec_value) {
  auto rc = funcptr(program, spec_id, spec_size, spec_value);
  TRACE_FN_ARG_BEGIN();
  TRACE_FN_ARG_PTR(program);
  TRACE_FN_ARG_UINT(spec_id);
  TRACE_FN_ARG_SIZE(spec_size);
  TRACE_FN_ARG_PTR(spec_value);
  TRACE_FN_ARG_END();
  return rc;
}

/// Calls without error check
#define CALL_CL_SILENT(Rc, Fn, ...)                                            \
  do {                                                                         \
    if (DebugLevel > 1) {                                                      \
      IDP1("CL_CALLER: %s %s\n", TO_STRING(Fn), TO_STRING(( __VA_ARGS__ )));   \
      Rc = TRACE_FN(Fn)(__VA_ARGS__);                                          \
    } else {                                                                   \
      Rc = Fn(__VA_ARGS__);                                                    \
    }                                                                          \
  } while (0)

/// Calls that only have return code
#define CALL_CL(Rc, Fn, ...)                                                   \
  do {                                                                         \
    CALL_CL_SILENT(Rc, Fn, __VA_ARGS__);                                       \
    if (Rc != CL_SUCCESS) {                                                    \
      IDP("Error: %s:%s failed with error code %d, %s\n", __func__, #Fn, Rc,   \
         getCLErrorName(Rc));                                                  \
    }                                                                          \
  } while (0)

/// Emit warning for unsuccessful CL call
#define CALL_CLW(Rc, Fn, ...)                                                  \
  do {                                                                         \
    CALL_CL_SILENT(Rc, Fn, __VA_ARGS__);                                       \
    if (Rc != CL_SUCCESS) {                                                    \
      IDP("Warning: %s:%s returned %d, %s\n", __func__, #Fn, Rc,               \
         getCLErrorName(Rc));                                                  \
    }                                                                          \
  } while (0)

#define CALL_CL_RET(Ret, Fn, ...)                                              \
  do {                                                                         \
    cl_int rc;                                                                 \
    CALL_CL(rc, Fn, __VA_ARGS__);                                              \
    if (rc != CL_SUCCESS)                                                      \
      return Ret;                                                              \
  } while (0)

#define CALL_CLW_RET(Ret, Fn, ...)                                             \
  do {                                                                         \
    cl_int rc;                                                                 \
    CALL_CLW(rc, Fn, __VA_ARGS__);                                             \
    if (rc != CL_SUCCESS)                                                      \
      return Ret;                                                              \
  } while (0)

#define CALL_CL_EXIT_FAIL(Fn, ...)                                             \
  do {                                                                         \
    cl_int rc;                                                                 \
    CALL_CL(rc, Fn, __VA_ARGS__);                                              \
    if (rc != CL_SUCCESS)                                                      \
      exit(EXIT_FAILURE);                                                      \
  } while (0)

#define CALL_CL_RET_FAIL(Fn, ...) CALL_CL_RET(OFFLOAD_FAIL, Fn, __VA_ARGS__)
#define CALL_CL_RET_NULL(Fn, ...) CALL_CL_RET(nullptr, Fn, __VA_ARGS__)
#define CALL_CL_RET_ZERO(Fn, ...) CALL_CL_RET(0, Fn, __VA_ARGS__)
#define CALL_CL_RET_VOID(Fn, ...) CALL_CL_RET(, Fn, __VA_ARGS__)
#define CALL_CLW_RET_VOID(Fn, ...) CALL_CLW_RET(, Fn, __VA_ARGS__)

/// Calls that have return value and return code
#define CALL_CL_RVRC(Rv, Fn, Rc, ...)                                          \
  do {                                                                         \
    if (DebugLevel > 1) {                                                      \
      IDP1("CL_CALLER: %s %s\n", TO_STRING(Fn), TO_STRING(( __VA_ARGS__ )));   \
      Rv = TRACE_FN(Fn)(__VA_ARGS__, &Rc);                                     \
    } else {                                                                   \
      Rv = Fn(__VA_ARGS__, &Rc);                                               \
    }                                                                          \
    if (Rc != CL_SUCCESS) {                                                    \
      IDP("Error: %s:%s failed with error code %d, %s\n", __func__, #Fn, Rc,   \
         getCLErrorName(Rc));                                                  \
    }                                                                          \
  } while (0)

/// Calls that only have return value
#define CALL_CL_RV(Rv, Fn, ...)                                                \
  do {                                                                         \
    if (DebugLevel > 1) {                                                      \
      IDP1("CL_CALLER: %s %s\n", TO_STRING(Fn), TO_STRING(( __VA_ARGS__ )));   \
      Rv = TRACE_FN(Fn)(__VA_ARGS__);                                          \
    } else {                                                                   \
      Rv = Fn(__VA_ARGS__);                                                    \
    }                                                                          \
  } while (0)

/// Calls that don't return anything
#define CALL_CL_VOID(Fn, ...)                                                  \
  do {                                                                         \
    if (DebugLevel > 1) {                                                      \
      IDP1("CL_CALLER: %s %s\n", TO_STRING(Fn), TO_STRING(( __VA_ARGS__ )));   \
      TRACE_FN(Fn)(__VA_ARGS__);                                               \
    } else {                                                                   \
      Fn(__VA_ARGS__);                                                         \
    }                                                                          \
  } while (0)

/// Call extension function, return nothing
#define CALL_CL_EXT_VOID(DeviceId, Name, ...)                                  \
  do {                                                                         \
    Name##_fn Fn = reinterpret_cast<Name##_fn>(                                \
        DeviceInfo->getExtensionFunctionPtr(DeviceId, Name##Id));              \
    if (DebugLevel > 1) {                                                      \
      IDP1("CL_CALLER: %s %s\n",                                               \
           TO_STRING(Name), TO_STRING(( __VA_ARGS__ )));                       \
      TRACE_FN(Name)(Fn, __VA_ARGS__);                                         \
    } else {                                                                   \
      (*Fn)(__VA_ARGS__);                                                      \
    }                                                                          \
  } while (0)

/// Extension calls without error reporting
#define CALL_CL_EXT_SILENT(DeviceId, Rc, Name, ...)                            \
  do {                                                                         \
    Name##_fn Fn = reinterpret_cast<Name##_fn>(                                \
        DeviceInfo->getExtensionFunctionPtr(DeviceId, Name##Id));              \
    if (DebugLevel > 1) {                                                      \
      IDP1("CL_CALLER: %s %s\n",                                               \
           TO_STRING(Name), TO_STRING(( __VA_ARGS__ )));                       \
      Rc = TRACE_FN(Name)(Fn, __VA_ARGS__);                                    \
    } else {                                                                   \
      Rc = (*Fn)(__VA_ARGS__);                                                 \
    }                                                                          \
  } while (0)

/// Extension calls that only have return code
#define CALL_CL_EXT(DeviceId, Rc, Name, ...)                                   \
  do {                                                                         \
    CALL_CL_EXT_SILENT(DeviceId, Rc, Name, __VA_ARGS__);                       \
    if (Rc != CL_SUCCESS) {                                                    \
      IDP("Error: %s:%s failed with error code %d, %s\n",                      \
          __func__, TO_STRING(Name), Rc, getCLErrorName(Rc));                  \
    }                                                                          \
  } while (0)

/// Extension calls that have return value and return code
#define CALL_CL_EXT_RVRC(DeviceId, Rv, Name, Rc, ...)                          \
  do {                                                                         \
    Name##_fn Fn = reinterpret_cast<Name##_fn>(                                \
        DeviceInfo->getExtensionFunctionPtr(DeviceId, Name##Id));              \
    if (DebugLevel > 1) {                                                      \
      IDP1("CL_CALLER: %s %s\n",                                               \
           TO_STRING(Name), TO_STRING(( __VA_ARGS__ )));                       \
      Rv = TRACE_FN(Name)(Fn, __VA_ARGS__, &Rc);                               \
    } else {                                                                   \
      Rv = (*Fn)(__VA_ARGS__, &Rc);                                            \
    }                                                                          \
    if (Rc != CL_SUCCESS) {                                                    \
      IDP("Error: %s:%s failed with error code %d, %s\n",                      \
          __func__, TO_STRING(Name), Rc, getCLErrorName(Rc));                  \
    }                                                                          \
  } while (0)

#define CALL_CL_EXT_RET(DeviceId, Ret, Name, ...)                              \
  do {                                                                         \
    cl_int rc;                                                                 \
    CALL_CL_EXT(DeviceId, rc, Name, __VA_ARGS__);                              \
    if (rc != CL_SUCCESS)                                                      \
      return Ret;                                                              \
  } while (0)

#define CALL_CL_EXT_RET_FAIL(DeviceId, Name, ...)                              \
  CALL_CL_EXT_RET(DeviceId, OFFLOAD_FAIL, Name, __VA_ARGS__)
#define CALL_CL_EXT_RET_NULL(DeviceId, Name, ...)                              \
  CALL_CL_EXT_RET(DeviceId, nullptr, Name, __VA_ARGS__)

#endif // !defined(RTL_TRACE_H)
#endif // INTEL_COLLAB
