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
clHostMemAllocINTEL(cl_context context, cl_mem_properties_intel *properties,
                    size_t size, cl_uint alignment, cl_int *errcode_ret) {
  void *retVal = nullptr;

  if (pfn_clHostMemAllocINTEL) {
    retVal = pfn_clHostMemAllocINTEL(context, properties, size, alignment,
                                     errcode_ret);
  } else if (CLUSM *clusm = GetCLUSM()) {
    retVal =
        clusm->hostMemAlloc(context, properties, size, alignment, errcode_ret);
  } else if (errcode_ret) {
    errcode_ret[0] = CL_INVALID_OPERATION;
  }

  return retVal;
}

static clDeviceMemAllocINTEL_fn pfn_clDeviceMemAllocINTEL = NULL;
CL_API_ENTRY void *CL_API_CALL
clDeviceMemAllocINTEL(cl_context context, cl_device_id device,
                      cl_mem_properties_intel *properties, // TBD: needed?
                      size_t size, cl_uint alignment, cl_int *errcode_ret) {
  void *retVal = nullptr;

  if (pfn_clDeviceMemAllocINTEL) {
    retVal = pfn_clDeviceMemAllocINTEL(context, device, properties, size,
                                       alignment, errcode_ret);
  } else if (CLUSM *clusm = GetCLUSM()) {
    retVal = clusm->deviceMemAlloc(context, device, properties, size, alignment,
                                   errcode_ret);
  } else if (errcode_ret) {
    errcode_ret[0] = CL_INVALID_OPERATION;
  }

  return retVal;
}

static clSharedMemAllocINTEL_fn pfn_clSharedMemAllocINTEL = NULL;
CL_API_ENTRY void *CL_API_CALL
clSharedMemAllocINTEL(cl_context context, cl_device_id device,
                      cl_mem_properties_intel *properties, // TBD: needed?
                      size_t size, cl_uint alignment, cl_int *errcode_ret) {
  void *retVal = nullptr;
  if (pfn_clSharedMemAllocINTEL) {
    retVal = pfn_clSharedMemAllocINTEL(context, device, properties, size,
                                       alignment, errcode_ret);
  } else if (CLUSM *clusm = GetCLUSM()) {
    retVal = clusm->sharedMemAlloc(context, device, properties, size, alignment,
                                   errcode_ret);
  } else if (errcode_ret) {
    errcode_ret[0] = CL_INVALID_OPERATION;
  }

  return retVal;
}

static clMemFreeINTEL_fn pfn_clMemFreeINTEL = NULL;
CL_API_ENTRY cl_int CL_API_CALL clMemFreeINTEL(cl_context context,
                                               const void *ptr) {
  cl_int retVal = CL_INVALID_OPERATION;

  if (pfn_clMemFreeINTEL) {
    retVal = pfn_clMemFreeINTEL(context, ptr);
  } else if (CLUSM *clusm = GetCLUSM()) {
    retVal = clusm->memFree(context, ptr);
  }

  return retVal;
}

static clGetMemAllocInfoINTEL_fn pfn_clGetMemAllocInfoINTEL = NULL;
CL_API_ENTRY cl_int CL_API_CALL clGetMemAllocInfoINTEL(
    cl_context context, const void *ptr, cl_mem_info_intel param_name,
    size_t param_value_size, void *param_value, size_t *param_value_size_ret) {
  cl_int retVal = CL_INVALID_OPERATION;

  if (pfn_clGetMemAllocInfoINTEL) {
    retVal =
        pfn_clGetMemAllocInfoINTEL(context, ptr, param_name, param_value_size,
                                   param_value, param_value_size_ret);
  } else if (CLUSM *clusm = GetCLUSM()) {
    retVal =
        clusm->getMemAllocInfoINTEL(context, ptr, param_name, param_value_size,
                                    param_value, param_value_size_ret);
  }

  return retVal;
}

static clSetKernelArgMemPointerINTEL_fn pfn_clSetKernelArgMemPointerINTEL =
    NULL;
CL_API_ENTRY cl_int CL_API_CALL clSetKernelArgMemPointerINTEL(
    cl_kernel kernel, cl_uint arg_index, const void *arg_value) {
  cl_int retVal = CL_INVALID_OPERATION;

  if (pfn_clSetKernelArgMemPointerINTEL) {
    retVal = pfn_clSetKernelArgMemPointerINTEL(kernel, arg_index, arg_value);
  } else if (GetCLUSM()) {
    retVal = clSetKernelArgSVMPointer(kernel, arg_index, arg_value);
  }

  return retVal;
}

static clEnqueueMemsetINTEL_fn pfn_clEnqueueMemsetINTEL = NULL;
CL_API_ENTRY cl_int CL_API_CALL
clEnqueueMemsetINTEL(cl_command_queue queue, void *dst_ptr, cl_int value,
                     size_t count, cl_uint num_events_in_wait_list,
                     const cl_event *event_wait_list, cl_event *event) {
  cl_int retVal = CL_INVALID_OPERATION;

  if (pfn_clEnqueueMemsetINTEL) {
    retVal = pfn_clEnqueueMemsetINTEL(queue, dst_ptr, value, count,
                                      num_events_in_wait_list, event_wait_list,
                                      event);
  } else if (GetCLUSM()) {
    const cl_uchar pattern = (cl_uchar)value;

    retVal =
        clEnqueueSVMMemFill(queue, dst_ptr, &pattern, sizeof(pattern), count,
                            num_events_in_wait_list, event_wait_list, event);
  }

  return retVal;
}

static clEnqueueMemcpyINTEL_fn pfn_clEnqueueMemcpyINTEL = NULL;
CL_API_ENTRY cl_int CL_API_CALL clEnqueueMemcpyINTEL(
    cl_command_queue queue, cl_bool blocking, void *dst_ptr,
    const void *src_ptr, size_t size, cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list, cl_event *event) {
  cl_int retVal = CL_INVALID_OPERATION;

  if (pfn_clEnqueueMemcpyINTEL) {
    retVal = pfn_clEnqueueMemcpyINTEL(queue, blocking, dst_ptr, src_ptr, size,
                                      num_events_in_wait_list, event_wait_list,
                                      event);
  } else if (GetCLUSM()) {
    retVal =
        clEnqueueSVMMemcpy(queue, blocking, dst_ptr, src_ptr, size,
                           num_events_in_wait_list, event_wait_list, event);
  }

  return retVal;
}

static clEnqueueMigrateMemINTEL_fn pfn_clEnqueueMigrateMemINTEL = NULL;
CL_API_ENTRY cl_int CL_API_CALL clEnqueueMigrateMemINTEL(
    cl_command_queue queue, const void *ptr, size_t size,
    cl_mem_migration_flags flags, cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list, cl_event *event) {
  cl_int retVal = CL_INVALID_OPERATION;

  if (pfn_clEnqueueMigrateMemINTEL) {
    retVal = pfn_clEnqueueMigrateMemINTEL(queue, ptr, size, flags,
                                          num_events_in_wait_list,
                                          event_wait_list, event);
  } else if (GetCLUSM()) {
    // We could check for OpenCL 2.1 and call the SVM migrate
    // functions, but for now we'll just enqueue a marker.
    retVal = clEnqueueMarkerWithWaitList(queue, num_events_in_wait_list,
                                         event_wait_list, event);
  }

  return retVal;
}

static clEnqueueMemAdviseINTEL_fn pfn_clEnqueueMemAdviseINTEL = NULL;
CL_API_ENTRY cl_int CL_API_CALL clEnqueueMemAdviseINTEL(
    cl_command_queue queue, const void *ptr, size_t size,
    cl_mem_advice_intel advice, cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list, cl_event *event) {
  cl_int retVal = CL_INVALID_OPERATION;

  if (pfn_clEnqueueMemAdviseINTEL) {
    retVal = pfn_clEnqueueMemAdviseINTEL(queue, ptr, size, advice,
                                         num_events_in_wait_list,
                                         event_wait_list, event);
  } else if (GetCLUSM()) {
    // TODO: What should we do here?
    // This isn't really supported yet.
    // Advice is typically safe to ignore,
    //  so a NOP will do.
    retVal = clEnqueueMarkerWithWaitList(queue, num_events_in_wait_list,
                                         event_wait_list, event);
  }

  return retVal;
}

#ifdef __cplusplus
namespace cl {
namespace sycl {
namespace detail {
namespace cliext {
#endif

#define GET_EXTENSION(_funcname)                                               \
  pfn_##_funcname = (_funcname##_fn)clGetExtensionFunctionAddressForPlatform(  \
      platform, #_funcname);

bool initializeExtensions(cl_platform_id platform) {
  GET_EXTENSION(clHostMemAllocINTEL);
  GET_EXTENSION(clDeviceMemAllocINTEL);
  GET_EXTENSION(clSharedMemAllocINTEL);
  GET_EXTENSION(clMemFreeINTEL);
  GET_EXTENSION(clGetMemAllocInfoINTEL);
  GET_EXTENSION(clSetKernelArgMemPointerINTEL);
  GET_EXTENSION(clEnqueueMemsetINTEL);
  GET_EXTENSION(clEnqueueMemcpyINTEL);
  GET_EXTENSION(clEnqueueMigrateMemINTEL);
  GET_EXTENSION(clEnqueueMemAdviseINTEL);

  return (pfn_clHostMemAllocINTEL && pfn_clDeviceMemAllocINTEL &&
          pfn_clSharedMemAllocINTEL && pfn_clMemFreeINTEL &&
          pfn_clSetKernelArgMemPointerINTEL && pfn_clEnqueueMemsetINTEL &&
          pfn_clEnqueueMemcpyINTEL);
}

#ifdef __cplusplus
} // namespace cliext
} // namespace detail
} // namespace sycl
} // namespace cl
#endif
