//==------------ opencl_shim.cpp - OpenCL extension for USM ----*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#include <CL/sycl/detail/clusm.hpp>
#include <string>


using namespace cl::sycl::detail::usm;
/***

  General philosophy: Try to use a CL extension for each function, 
   if it exists. Otherwise, fall back to CLUSM's USM-on-SVM.

 **/

static clHostMemAllocINTEL_fn pfn_clHostMemAllocINTEL = NULL;
CL_API_ENTRY void *CL_API_CALL
clHostMemAllocINTEL(cl_context context, cl_mem_properties_intel* properties,
                    size_t size, cl_uint alignment, cl_int *errcode_ret) {
  if (pfn_clHostMemAllocINTEL) {
    return pfn_clHostMemAllocINTEL(
      context,
      properties,
      size,
      alignment,
      errcode_ret);
  }
  else if (CLUSM *clusm = GetCLUSM()) {
      return 
        clusm->hostMemAlloc(context, properties, size, alignment, errcode_ret);
  }

  if (errcode_ret) 
    errcode_ret[0] = CL_INVALID_OPERATION;

  return nullptr;
}

static clDeviceMemAllocINTEL_fn pfn_clDeviceMemAllocINTEL = NULL;
CL_API_ENTRY void *CL_API_CALL
clDeviceMemAllocINTEL(cl_context context, cl_device_id device,
                      cl_mem_properties_intel* properties, // TBD: needed?
                      size_t size, cl_uint alignment, cl_int *errcode_ret) {
  if (pfn_clDeviceMemAllocINTEL) {
    return pfn_clDeviceMemAllocINTEL(
      context,
      device,
      properties,
      size,
      alignment,
      errcode_ret);
  }
  else if (CLUSM *clusm = GetCLUSM()) {
    return clusm->deviceMemAlloc(context,
                                 device,
                                 properties,
                                 size,
                                 alignment,
                                 errcode_ret);
  }

  if (errcode_ret)
    errcode_ret[0] = CL_INVALID_OPERATION;

  return nullptr;
}

static clSharedMemAllocINTEL_fn pfn_clSharedMemAllocINTEL = NULL;
CL_API_ENTRY void *CL_API_CALL
clSharedMemAllocINTEL(cl_context context, cl_device_id device,
                      cl_mem_properties_intel* properties, // TBD: needed?
                      size_t size, cl_uint alignment, cl_int *errcode_ret) {
  if (pfn_clSharedMemAllocINTEL) {
    return pfn_clSharedMemAllocINTEL(
      context,
      device,
      properties,
      size,
      alignment,
      errcode_ret);
  }
  else if (CLUSM *clusm = GetCLUSM()) {
    return clusm->sharedMemAlloc(context,
                                 device,
                                 properties,
                                 size,
                                 alignment,
                                 errcode_ret);
  }
  
  if (errcode_ret)
    errcode_ret[0] = CL_INVALID_OPERATION;
  
  return nullptr;
}

static clMemFreeINTEL_fn pfn_clMemFreeINTEL = NULL;
CL_API_ENTRY cl_int CL_API_CALL clMemFreeINTEL(cl_context context,
                                               const void *ptr) {
  if (pfn_clMemFreeINTEL) {
    return pfn_clMemFreeINTEL(context, ptr);
  }
  else if (CLUSM *clusm = GetCLUSM()) {
    return clusm->memFree(context, ptr);
  }

  return CL_INVALID_OPERATION;
}

static clGetMemAllocInfoINTEL_fn pfn_clGetMemAllocInfoINTEL = NULL;
CL_API_ENTRY cl_int CL_API_CALL clGetMemAllocInfoINTEL(
    cl_context context, const void *ptr, cl_mem_info_intel param_name,
    size_t param_value_size, void *param_value, size_t *param_value_size_ret) {
  if (pfn_clGetMemAllocInfoINTEL) {
    return pfn_clGetMemAllocInfoINTEL(
      context,
      ptr,
      param_name,
      param_value_size,
      param_value,
      param_value_size_ret);
  }
  else if (CLUSM *clusm = GetCLUSM()) {
    return clusm->getMemAllocInfoINTEL(
      context,
      ptr,
      param_name,
      param_value_size,
      param_value,
      param_value_size_ret);
  }

  return CL_INVALID_OPERATION;
}

static clSetKernelArgMemPointerINTEL_fn pfn_clSetKernelArgMemPointerINTEL = NULL;
CL_API_ENTRY cl_int CL_API_CALL clSetKernelArgMemPointerINTEL(
    cl_kernel kernel, cl_uint arg_index, const void *arg_value) {
  if (pfn_clSetKernelArgMemPointerINTEL) {
    return clSetKernelArgMemPointerINTEL(
      kernel,
      arg_index,
      arg_value);
  }
  else if (GetCLUSM()) {
    return clSetKernelArgSVMPointer(kernel, arg_index, arg_value);
  }

  return CL_INVALID_OPERATION;
}

static clEnqueueMemsetINTEL_fn pfn_clEnqueueMemsetINTEL = NULL;
CL_API_ENTRY cl_int CL_API_CALL
clEnqueueMemsetINTEL(cl_command_queue queue, void *dst_ptr, cl_int value,
                     size_t count, cl_uint num_events_in_wait_list,
                     const cl_event *event_wait_list, cl_event *event) {
  if (pfn_clEnqueueMemsetINTEL) {
    return pfn_clEnqueueMemsetINTEL(
      queue,
      dst_ptr,
      value,
      count,
      num_events_in_wait_list,
      event_wait_list,
      event);
  }
  else if (GetCLUSM()) {
    const cl_uchar pattern = (cl_uchar)value;

    return
        clEnqueueSVMMemFill(queue, dst_ptr, &pattern, sizeof(pattern), count,
                            num_events_in_wait_list, event_wait_list, event);
  }

  return CL_INVALID_OPERATION;
}

static clEnqueueMemcpyINTEL_fn pfn_clEnqueueMemcpyINTEL = NULL;
CL_API_ENTRY cl_int CL_API_CALL clEnqueueMemcpyINTEL(
    cl_command_queue queue, cl_bool blocking, void *dst_ptr,
    const void *src_ptr, size_t size, cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list, cl_event *event) {
  if (pfn_clEnqueueMemcpyINTEL) {
    return pfn_clEnqueueMemcpyINTEL(
      queue,
      blocking,
      dst_ptr,
      src_ptr,
      size,
      num_events_in_wait_list,
      event_wait_list,
      event);
  }
  else if (GetCLUSM()) {
    return 
        clEnqueueSVMMemcpy(queue, blocking, dst_ptr, src_ptr, size,
                           num_events_in_wait_list, event_wait_list, event);
  }

  return CL_INVALID_OPERATION;
}

static clEnqueueMigrateMemINTEL_fn pfn_clEnqueueMigrateMemINTEL = NULL;
CL_API_ENTRY cl_int CL_API_CALL clEnqueueMigrateMemINTEL(
    cl_command_queue queue, const void *ptr, size_t size,
    cl_mem_migration_flags flags, cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list, cl_event *event) {
  if (pfn_clEnqueueMigrateMemINTEL) {
    return pfn_clEnqueueMigrateMemINTEL(
      queue,
      ptr,
      size,
      flags,
      num_events_in_wait_list,
      event_wait_list,
      event);
  }
  else if (GetCLUSM()) {
    // We could check for OpenCL 2.1 and call the SVM migrate
    // functions, but for now we'll just enqueue a marker.
#if 0
    retVal = clEnqueueSVMMigrateMem(
      queue,
      1,
      &ptr,
      &size,
      flags,
      num_events_in_wait_list,
      event_wait_list,
      event );
#else
    return clEnqueueMarkerWithWaitList(queue, num_events_in_wait_list,
                                       event_wait_list, event);
#endif
  }

  return CL_INVALID_OPERATION;
}

static clEnqueueMemAdviseINTEL_fn pfn_clEnqueueMemAdviseINTEL = NULL;
CL_API_ENTRY cl_int CL_API_CALL clEnqueueMemAdviseINTEL(
    cl_command_queue queue, const void *ptr, size_t size,
    cl_mem_advice_intel advice, cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list, cl_event *event) {
  if (pfn_clEnqueueMemAdviseINTEL) {
    return pfn_clEnqueueMemAdviseINTEL(
      queue,
      ptr,
      size,
      advice,
      num_events_in_wait_list,
      event_wait_list,
      event);
  }
  else if (GetCLUSM()) {
    // TODO: What should we do here?
    // This isn't really supported yet.
    // Advice is typically safe to ignore,
    //  so a NOP will do.
    return clEnqueueMarkerWithWaitList(queue, num_events_in_wait_list,
                                       event_wait_list, event);
  }

  return CL_INVALID_OPERATION;
}

#ifdef __cplusplus
namespace cl {
namespace sycl {
namespace detail {
namespace cliext {
#endif

#define GET_EXTENSION( _funcname )                                      \
    pfn_ ## _funcname = ( _funcname ## _fn )                            \
        clGetExtensionFunctionAddressForPlatform(platform, #_funcname);

void initializeExtensions( cl_platform_id platform )
{
  GET_EXTENSION( clHostMemAllocINTEL );
  GET_EXTENSION( clDeviceMemAllocINTEL );
  GET_EXTENSION( clSharedMemAllocINTEL );
  GET_EXTENSION( clMemFreeINTEL );
  GET_EXTENSION( clGetMemAllocInfoINTEL );
  GET_EXTENSION( clSetKernelArgMemPointerINTEL );
  GET_EXTENSION( clEnqueueMemsetINTEL );
  GET_EXTENSION( clEnqueueMemcpyINTEL );
  GET_EXTENSION( clEnqueueMigrateMemINTEL );
  GET_EXTENSION( clEnqueueMemAdviseINTEL );
}

#ifdef __cplusplus
}
}
}
}
#endif
