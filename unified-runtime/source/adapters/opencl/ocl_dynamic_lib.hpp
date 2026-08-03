//===------- ocl_dynamic_lib.hpp - OpenCL Dynamic Loading ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//
#pragma once

#ifdef UR_STATIC_ADAPTER_OPENCL

// Include OpenCL headers BEFORE any redirection
#include <CL/cl.h>
#include <CL/cl_ext.h>

namespace ocl {

// Declare function pointers for all OpenCL functions using X-macros
#define OCL_FUNC(name) extern decltype(::name) *name##_ptr;
#define OCL_OPTIONAL_FUNC(name) OCL_FUNC(name)
#include "ocl_functions.def"
#undef OCL_OPTIONAL_FUNC
#undef OCL_FUNC

bool loadOCLLibrary();
void unloadOCLLibrary();

} // namespace ocl

// Only define the redirection macros if we're NOT in the implementation file
// The implementation file needs the original function names for decltype
#ifndef OCL_DYNAMIC_LIB_IMPL

// Redirect all OpenCL function calls to our dynamically loaded pointers
#define clBuildProgram ocl::clBuildProgram_ptr
#define clCompileProgram ocl::clCompileProgram_ptr
#define clCreateBuffer ocl::clCreateBuffer_ptr
#define clCreateCommandQueue ocl::clCreateCommandQueue_ptr
#define clCreateCommandQueueWithProperties                                     \
  ocl::clCreateCommandQueueWithProperties_ptr
#define clCreateContext ocl::clCreateContext_ptr
#define clCreateImage ocl::clCreateImage_ptr
#define clCreateKernel ocl::clCreateKernel_ptr
#define clCreateProgramWithBinary ocl::clCreateProgramWithBinary_ptr
#define clCreateProgramWithIL ocl::clCreateProgramWithIL_ptr
#define clCreateSampler ocl::clCreateSampler_ptr
#define clCreateSubBuffer ocl::clCreateSubBuffer_ptr
#define clCreateSubDevices ocl::clCreateSubDevices_ptr
#define clEnqueueBarrierWithWaitList ocl::clEnqueueBarrierWithWaitList_ptr
#define clEnqueueCopyBuffer ocl::clEnqueueCopyBuffer_ptr
#define clEnqueueCopyBufferRect ocl::clEnqueueCopyBufferRect_ptr
#define clEnqueueCopyImage ocl::clEnqueueCopyImage_ptr
#define clEnqueueFillBuffer ocl::clEnqueueFillBuffer_ptr
#define clEnqueueMapBuffer ocl::clEnqueueMapBuffer_ptr
#define clEnqueueMarkerWithWaitList ocl::clEnqueueMarkerWithWaitList_ptr
#define clEnqueueNDRangeKernel ocl::clEnqueueNDRangeKernel_ptr
#define clEnqueueReadBuffer ocl::clEnqueueReadBuffer_ptr
#define clEnqueueReadBufferRect ocl::clEnqueueReadBufferRect_ptr
#define clEnqueueReadImage ocl::clEnqueueReadImage_ptr
#define clEnqueueUnmapMemObject ocl::clEnqueueUnmapMemObject_ptr
#define clEnqueueWriteBuffer ocl::clEnqueueWriteBuffer_ptr
#define clEnqueueWriteBufferRect ocl::clEnqueueWriteBufferRect_ptr
#define clEnqueueWriteImage ocl::clEnqueueWriteImage_ptr
#define clFinish ocl::clFinish_ptr
#define clFlush ocl::clFlush_ptr
#define clGetCommandQueueInfo ocl::clGetCommandQueueInfo_ptr
#define clGetContextInfo ocl::clGetContextInfo_ptr
#define clGetDeviceAndHostTimer ocl::clGetDeviceAndHostTimer_ptr
#define clGetDeviceIDs ocl::clGetDeviceIDs_ptr
#define clGetDeviceInfo ocl::clGetDeviceInfo_ptr
#define clGetEventInfo ocl::clGetEventInfo_ptr
#define clGetEventProfilingInfo ocl::clGetEventProfilingInfo_ptr
#define clGetExtensionFunctionAddressForPlatform                               \
  ocl::clGetExtensionFunctionAddressForPlatform_ptr
#define clGetHostTimer ocl::clGetHostTimer_ptr
#define clGetImageInfo ocl::clGetImageInfo_ptr
#define clGetKernelInfo ocl::clGetKernelInfo_ptr
#define clGetKernelSubGroupInfo ocl::clGetKernelSubGroupInfo_ptr
#define clGetKernelWorkGroupInfo ocl::clGetKernelWorkGroupInfo_ptr
#define clGetMemObjectInfo ocl::clGetMemObjectInfo_ptr
#define clGetPlatformIDs ocl::clGetPlatformIDs_ptr
#define clGetPlatformInfo ocl::clGetPlatformInfo_ptr
#define clGetProgramBuildInfo ocl::clGetProgramBuildInfo_ptr
#define clGetProgramInfo ocl::clGetProgramInfo_ptr
#define clGetSamplerInfo ocl::clGetSamplerInfo_ptr
#define clLinkProgram ocl::clLinkProgram_ptr
#define clReleaseCommandQueue ocl::clReleaseCommandQueue_ptr
#define clReleaseContext ocl::clReleaseContext_ptr
#define clReleaseDevice ocl::clReleaseDevice_ptr
#define clReleaseEvent ocl::clReleaseEvent_ptr
#define clReleaseKernel ocl::clReleaseKernel_ptr
#define clReleaseMemObject ocl::clReleaseMemObject_ptr
#define clReleaseProgram ocl::clReleaseProgram_ptr
#define clReleaseSampler ocl::clReleaseSampler_ptr
#define clRetainContext ocl::clRetainContext_ptr
#define clRetainDevice ocl::clRetainDevice_ptr
#define clRetainEvent ocl::clRetainEvent_ptr
#define clSetEventCallback ocl::clSetEventCallback_ptr
#define clSetKernelArg ocl::clSetKernelArg_ptr
#define clSetKernelExecInfo ocl::clSetKernelExecInfo_ptr
#define clWaitForEvents ocl::clWaitForEvents_ptr
// clSetProgramSpecializationConstant and clSetContextDestructorCallback are
// not redirected here; they are accessed via per-adapter struct fields
// (clSetProgramSpecializationConstantFn / clSetContextDestructorCallbackFn).

#endif // OCL_DYNAMIC_LIB_IMPL

#endif // UR_STATIC_ADAPTER_OPENCL
