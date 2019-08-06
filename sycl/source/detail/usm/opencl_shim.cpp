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

// Changing to per-context tracking
// TODO: use piContext everywhere
static std::map<cl_context, clHostMemAllocINTEL_fn> pfn_clHostMemAllocINTEL;
static std::map<cl_context, clDeviceMemAllocINTEL_fn> pfn_clDeviceMemAllocINTEL;
static std::map<cl_context, clSharedMemAllocINTEL_fn> pfn_clSharedMemAllocINTEL;
static std::map<cl_context, clMemFreeINTEL_fn> pfn_clMemFreeINTEL;
static std::map<cl_context, clGetMemAllocInfoINTEL_fn>
    pfn_clGetMemAllocInfoINTEL;
static std::map<cl_context, clSetKernelArgMemPointerINTEL_fn>
    pfn_clSetKernelArgMemPointerINTEL;
static std::map<cl_context, clEnqueueMemsetINTEL_fn> pfn_clEnqueueMemsetINTEL;
static std::map<cl_context, clEnqueueMemcpyINTEL_fn> pfn_clEnqueueMemcpyINTEL;
static std::map<cl_context, clEnqueueMigrateMemINTEL_fn>
    pfn_clEnqueueMigrateMemINTEL;
static std::map<cl_context, clEnqueueMemAdviseINTEL_fn>
    pfn_clEnqueueMemAdviseINTEL;

CL_API_ENTRY void *CL_API_CALL
clHostMemAllocINTEL(cl_context context, cl_mem_properties_intel *properties,
                    size_t size, cl_uint alignment, cl_int *errcode_ret) {
  void *retVal = nullptr;

  if (pfn_clHostMemAllocINTEL[context]) {
    retVal = pfn_clHostMemAllocINTEL[context](context, properties, size,
                                              alignment, errcode_ret);
  } else if (CLUSM *clusm = GetCLUSM(context)) {
    retVal =
        clusm->hostMemAlloc(context, properties, size, alignment, errcode_ret);
  } else if (errcode_ret) {
    errcode_ret[0] = CL_INVALID_OPERATION;
  }

  return retVal;
}

CL_API_ENTRY void *CL_API_CALL
clDeviceMemAllocINTEL(cl_context context, cl_device_id device,
                      cl_mem_properties_intel *properties, // TBD: needed?
                      size_t size, cl_uint alignment, cl_int *errcode_ret) {
  void *retVal = nullptr;

  if (pfn_clDeviceMemAllocINTEL[context]) {
    retVal = pfn_clDeviceMemAllocINTEL[context](context, device, properties,
                                                size, alignment, errcode_ret);
  } else if (CLUSM *clusm = GetCLUSM(context)) {
    retVal = clusm->deviceMemAlloc(context, device, properties, size, alignment,
                                   errcode_ret);
  } else if (errcode_ret) {
    errcode_ret[0] = CL_INVALID_OPERATION;
  }

  return retVal;
}

CL_API_ENTRY void *CL_API_CALL
clSharedMemAllocINTEL(cl_context context, cl_device_id device,
                      cl_mem_properties_intel *properties, // TBD: needed?
                      size_t size, cl_uint alignment, cl_int *errcode_ret) {
  void *retVal = nullptr;
  if (pfn_clSharedMemAllocINTEL[context]) {
    retVal = pfn_clSharedMemAllocINTEL[context](context, device, properties,
                                                size, alignment, errcode_ret);
  } else if (CLUSM *clusm = GetCLUSM(context)) {
    retVal = clusm->sharedMemAlloc(context, device, properties, size, alignment,
                                   errcode_ret);
  } else if (errcode_ret) {
    errcode_ret[0] = CL_INVALID_OPERATION;
  }

  return retVal;
}

CL_API_ENTRY cl_int CL_API_CALL clMemFreeINTEL(cl_context context,
                                               const void *ptr) {
  cl_int retVal = CL_INVALID_OPERATION;

  if (pfn_clMemFreeINTEL[context]) {
    retVal = pfn_clMemFreeINTEL[context](context, ptr);
  } else if (CLUSM *clusm = GetCLUSM(context)) {
    retVal = clusm->memFree(context, ptr);
  }

  return retVal;
}

CL_API_ENTRY cl_int CL_API_CALL clGetMemAllocInfoINTEL(
    cl_context context, const void *ptr, cl_mem_info_intel param_name,
    size_t param_value_size, void *param_value, size_t *param_value_size_ret) {
  cl_int retVal = CL_INVALID_OPERATION;

  if (pfn_clGetMemAllocInfoINTEL[context]) {
    retVal = pfn_clGetMemAllocInfoINTEL[context](context, ptr, param_name,
                                                 param_value_size, param_value,
                                                 param_value_size_ret);
  } else if (CLUSM *clusm = GetCLUSM(context)) {
    retVal =
        clusm->getMemAllocInfoINTEL(context, ptr, param_name, param_value_size,
                                    param_value, param_value_size_ret);
  }

  return retVal;
}

CL_API_ENTRY cl_int CL_API_CALL clSetKernelArgMemPointerINTEL(
    cl_kernel kernel, cl_uint arg_index, const void *arg_value) {
  cl_int retVal = CL_INVALID_OPERATION;
  cl_context context;
  CHECK_OCL_CODE(clGetKernelInfo(kernel, CL_KERNEL_CONTEXT, sizeof(cl_context),
                                 &context, nullptr));

  if (pfn_clSetKernelArgMemPointerINTEL[context]) {
    retVal = pfn_clSetKernelArgMemPointerINTEL[context](kernel, arg_index,
                                                        arg_value);
  } else if (GetCLUSM(context)) {
    retVal = clSetKernelArgSVMPointer(kernel, arg_index, arg_value);
  }

  return retVal;
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueMemsetINTEL(cl_command_queue queue, void *dst_ptr, cl_int value,
                     size_t count, cl_uint num_events_in_wait_list,
                     const cl_event *event_wait_list, cl_event *event) {
  cl_int retVal = CL_INVALID_OPERATION;
  cl_context context;
  CHECK_OCL_CODE(clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT,
                                       sizeof(cl_context), &context, nullptr));

  if (pfn_clEnqueueMemsetINTEL[context]) {
    retVal = pfn_clEnqueueMemsetINTEL[context](queue, dst_ptr, value, count,
                                               num_events_in_wait_list,
                                               event_wait_list, event);
  } else if (GetCLUSM(context)) {
    const cl_uchar pattern = (cl_uchar)value;

    retVal =
        clEnqueueSVMMemFill(queue, dst_ptr, &pattern, sizeof(pattern), count,
                            num_events_in_wait_list, event_wait_list, event);
  }

  return retVal;
}

CL_API_ENTRY cl_int CL_API_CALL clEnqueueMemcpyINTEL(
    cl_command_queue queue, cl_bool blocking, void *dst_ptr,
    const void *src_ptr, size_t size, cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list, cl_event *event) {
  cl_int retVal = CL_INVALID_OPERATION;
  cl_context context;
  CHECK_OCL_CODE(clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT,
                                       sizeof(cl_context), &context, nullptr));

  if (pfn_clEnqueueMemcpyINTEL[context]) {
    retVal = pfn_clEnqueueMemcpyINTEL[context](
        queue, blocking, dst_ptr, src_ptr, size, num_events_in_wait_list,
        event_wait_list, event);
  } else if (GetCLUSM(context)) {
    retVal =
        clEnqueueSVMMemcpy(queue, blocking, dst_ptr, src_ptr, size,
                           num_events_in_wait_list, event_wait_list, event);
  }

  return retVal;
}

CL_API_ENTRY cl_int CL_API_CALL clEnqueueMigrateMemINTEL(
    cl_command_queue queue, const void *ptr, size_t size,
    cl_mem_migration_flags flags, cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list, cl_event *event) {
  cl_int retVal = CL_INVALID_OPERATION;
  cl_context context;
  CHECK_OCL_CODE(clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT,
                                       sizeof(cl_context), &context, nullptr));

  if (pfn_clEnqueueMigrateMemINTEL[context]) {
    retVal = pfn_clEnqueueMigrateMemINTEL[context](queue, ptr, size, flags,
                                                   num_events_in_wait_list,
                                                   event_wait_list, event);
  } else if (GetCLUSM(context)) {
    // We could check for OpenCL 2.1 and call the SVM migrate
    // functions, but for now we'll just enqueue a marker.
    retVal = clEnqueueMarkerWithWaitList(queue, num_events_in_wait_list,
                                         event_wait_list, event);
  }

  return retVal;
}

CL_API_ENTRY cl_int CL_API_CALL clEnqueueMemAdviseINTEL(
    cl_command_queue queue, const void *ptr, size_t size,
    cl_mem_advice_intel advice, cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list, cl_event *event) {
  cl_int retVal = CL_INVALID_OPERATION;
  cl_context context;
  CHECK_OCL_CODE(clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT,
                                       sizeof(cl_context), &context, nullptr));

  if (pfn_clEnqueueMemAdviseINTEL[context]) {
    retVal = pfn_clEnqueueMemAdviseINTEL[context](queue, ptr, size, advice,
                                                  num_events_in_wait_list,
                                                  event_wait_list, event);
  } else if (GetCLUSM(context)) {
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
  (_funcname##_fn)clGetExtensionFunctionAddressForPlatform(  \
      platform, #_funcname);

bool initializeExtensions(cl_context context, cl_platform_id platform) {
  pfn_clHostMemAllocINTEL[context] = GET_EXTENSION(clHostMemAllocINTEL);
  pfn_clDeviceMemAllocINTEL[context] = GET_EXTENSION(clDeviceMemAllocINTEL);
  pfn_clSharedMemAllocINTEL[context] = GET_EXTENSION(clSharedMemAllocINTEL);
  pfn_clMemFreeINTEL[context] = GET_EXTENSION(clMemFreeINTEL);
  pfn_clGetMemAllocInfoINTEL[context] = GET_EXTENSION(clGetMemAllocInfoINTEL);
  pfn_clSetKernelArgMemPointerINTEL[context] =
      GET_EXTENSION(clSetKernelArgMemPointerINTEL);
  pfn_clEnqueueMemsetINTEL[context] = GET_EXTENSION(clEnqueueMemsetINTEL);
  pfn_clEnqueueMemcpyINTEL[context] = GET_EXTENSION(clEnqueueMemcpyINTEL);
  pfn_clEnqueueMigrateMemINTEL[context] =
      GET_EXTENSION(clEnqueueMigrateMemINTEL);
  pfn_clEnqueueMemAdviseINTEL[context] = GET_EXTENSION(clEnqueueMemAdviseINTEL);

  return (
      pfn_clHostMemAllocINTEL[context] && pfn_clDeviceMemAllocINTEL[context] &&
      pfn_clSharedMemAllocINTEL[context] && pfn_clMemFreeINTEL[context] &&
      pfn_clSetKernelArgMemPointerINTEL[context] &&
      pfn_clEnqueueMemsetINTEL[context] && pfn_clEnqueueMemcpyINTEL[context]);
}

#ifdef __cplusplus
} // namespace cliext
} // namespace detail
} // namespace sycl
} // namespace cl
#endif
