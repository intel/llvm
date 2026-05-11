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
#define OCL_FUNC(name, required) extern decltype(::name) *name##_ptr;
#include "ocl_functions.def"
#undef OCL_FUNC

bool loadOCLLibrary();
void unloadOCLLibrary();

} // namespace ocl

// Only define the redirection macros if we're NOT in the implementation file
// The implementation file needs the original function names for decltype
#ifndef OCL_DYNAMIC_LIB_IMPL

// Redirect all OpenCL function calls to our dynamically loaded pointers
// We use simple #define to replace the function name with our pointer
#define clGetPlatformIDs ocl::clGetPlatformIDs_ptr
#define clGetPlatformInfo ocl::clGetPlatformInfo_ptr
#define clGetDeviceIDs ocl::clGetDeviceIDs_ptr
#define clGetDeviceInfo ocl::clGetDeviceInfo_ptr
#define clCreateContext ocl::clCreateContext_ptr
#define clCreateContextFromType ocl::clCreateContextFromType_ptr
#define clRetainContext ocl::clRetainContext_ptr
#define clReleaseContext ocl::clReleaseContext_ptr
#define clGetContextInfo ocl::clGetContextInfo_ptr
#define clCreateCommandQueue ocl::clCreateCommandQueue_ptr
#define clRetainCommandQueue ocl::clRetainCommandQueue_ptr
#define clReleaseCommandQueue ocl::clReleaseCommandQueue_ptr
#define clGetCommandQueueInfo ocl::clGetCommandQueueInfo_ptr
#define clCreateBuffer ocl::clCreateBuffer_ptr
#define clRetainMemObject ocl::clRetainMemObject_ptr
#define clReleaseMemObject ocl::clReleaseMemObject_ptr
#define clGetMemObjectInfo ocl::clGetMemObjectInfo_ptr
#define clGetImageInfo ocl::clGetImageInfo_ptr
#define clCreateSampler ocl::clCreateSampler_ptr
#define clRetainSampler ocl::clRetainSampler_ptr
#define clReleaseSampler ocl::clReleaseSampler_ptr
#define clGetSamplerInfo ocl::clGetSamplerInfo_ptr
#define clCreateProgramWithSource ocl::clCreateProgramWithSource_ptr
#define clCreateProgramWithBinary ocl::clCreateProgramWithBinary_ptr
#define clRetainProgram ocl::clRetainProgram_ptr
#define clReleaseProgram ocl::clReleaseProgram_ptr
#define clBuildProgram ocl::clBuildProgram_ptr
#define clGetProgramInfo ocl::clGetProgramInfo_ptr
#define clGetProgramBuildInfo ocl::clGetProgramBuildInfo_ptr
#define clCreateKernel ocl::clCreateKernel_ptr
#define clCreateKernelsInProgram ocl::clCreateKernelsInProgram_ptr
#define clRetainKernel ocl::clRetainKernel_ptr
#define clReleaseKernel ocl::clReleaseKernel_ptr
#define clSetKernelArg ocl::clSetKernelArg_ptr
#define clGetKernelInfo ocl::clGetKernelInfo_ptr
#define clGetKernelWorkGroupInfo ocl::clGetKernelWorkGroupInfo_ptr
#define clWaitForEvents ocl::clWaitForEvents_ptr
#define clGetEventInfo ocl::clGetEventInfo_ptr
#define clRetainEvent ocl::clRetainEvent_ptr
#define clReleaseEvent ocl::clReleaseEvent_ptr
#define clGetEventProfilingInfo ocl::clGetEventProfilingInfo_ptr
#define clFlush ocl::clFlush_ptr
#define clFinish ocl::clFinish_ptr
#define clEnqueueReadBuffer ocl::clEnqueueReadBuffer_ptr
#define clEnqueueWriteBuffer ocl::clEnqueueWriteBuffer_ptr
#define clEnqueueCopyBuffer ocl::clEnqueueCopyBuffer_ptr
#define clEnqueueReadImage ocl::clEnqueueReadImage_ptr
#define clEnqueueWriteImage ocl::clEnqueueWriteImage_ptr
#define clEnqueueCopyImage ocl::clEnqueueCopyImage_ptr
#define clEnqueueCopyImageToBuffer ocl::clEnqueueCopyImageToBuffer_ptr
#define clEnqueueCopyBufferToImage ocl::clEnqueueCopyBufferToImage_ptr
#define clEnqueueMapBuffer ocl::clEnqueueMapBuffer_ptr
#define clEnqueueMapImage ocl::clEnqueueMapImage_ptr
#define clEnqueueUnmapMemObject ocl::clEnqueueUnmapMemObject_ptr
#define clEnqueueNDRangeKernel ocl::clEnqueueNDRangeKernel_ptr
#define clEnqueueNativeKernel ocl::clEnqueueNativeKernel_ptr
#define clEnqueueMarker ocl::clEnqueueMarker_ptr
#define clEnqueueWaitForEvents ocl::clEnqueueWaitForEvents_ptr
#define clEnqueueBarrier ocl::clEnqueueBarrier_ptr
#define clGetExtensionFunctionAddress ocl::clGetExtensionFunctionAddress_ptr
#define clCreateSubBuffer ocl::clCreateSubBuffer_ptr
#define clSetMemObjectDestructorCallback                                       \
  ocl::clSetMemObjectDestructorCallback_ptr
#define clCreateUserEvent ocl::clCreateUserEvent_ptr
#define clSetUserEventStatus ocl::clSetUserEventStatus_ptr
#define clSetEventCallback ocl::clSetEventCallback_ptr
#define clEnqueueReadBufferRect ocl::clEnqueueReadBufferRect_ptr
#define clEnqueueWriteBufferRect ocl::clEnqueueWriteBufferRect_ptr
#define clEnqueueCopyBufferRect ocl::clEnqueueCopyBufferRect_ptr
#define clCreateImage ocl::clCreateImage_ptr
#define clCompileProgram ocl::clCompileProgram_ptr
#define clLinkProgram ocl::clLinkProgram_ptr
#define clUnloadPlatformCompiler ocl::clUnloadPlatformCompiler_ptr
#define clGetKernelArgInfo ocl::clGetKernelArgInfo_ptr
#define clEnqueueFillBuffer ocl::clEnqueueFillBuffer_ptr
#define clEnqueueFillImage ocl::clEnqueueFillImage_ptr
#define clEnqueueMigrateMemObjects ocl::clEnqueueMigrateMemObjects_ptr
#define clEnqueueMarkerWithWaitList ocl::clEnqueueMarkerWithWaitList_ptr
#define clEnqueueBarrierWithWaitList ocl::clEnqueueBarrierWithWaitList_ptr
#define clGetExtensionFunctionAddressForPlatform                               \
  ocl::clGetExtensionFunctionAddressForPlatform_ptr
#define clCreateCommandQueueWithProperties                                     \
  ocl::clCreateCommandQueueWithProperties_ptr
#define clCreatePipe ocl::clCreatePipe_ptr
#define clGetPipeInfo ocl::clGetPipeInfo_ptr
#define clSVMAlloc ocl::clSVMAlloc_ptr
#define clSVMFree ocl::clSVMFree_ptr
#define clCreateSamplerWithProperties ocl::clCreateSamplerWithProperties_ptr
#define clSetKernelArgSVMPointer ocl::clSetKernelArgSVMPointer_ptr
#define clSetKernelExecInfo ocl::clSetKernelExecInfo_ptr
#define clEnqueueSVMFree ocl::clEnqueueSVMFree_ptr
#define clEnqueueSVMMemcpy ocl::clEnqueueSVMMemcpy_ptr
#define clEnqueueSVMMemFill ocl::clEnqueueSVMMemFill_ptr
#define clEnqueueSVMMap ocl::clEnqueueSVMMap_ptr
#define clEnqueueSVMUnmap ocl::clEnqueueSVMUnmap_ptr
#define clSetProgramSpecializationConstant                                     \
  ocl::clSetProgramSpecializationConstant_ptr
#define clSetProgramReleaseCallback ocl::clSetProgramReleaseCallback_ptr
#define clCreateBufferWithProperties ocl::clCreateBufferWithProperties_ptr
#define clCreateImageWithProperties ocl::clCreateImageWithProperties_ptr
#define clSetContextDestructorCallback ocl::clSetContextDestructorCallback_ptr
#define clCreateProgramWithIL ocl::clCreateProgramWithIL_ptr
#define clGetHostTimer ocl::clGetHostTimer_ptr
#define clGetDeviceAndHostTimer ocl::clGetDeviceAndHostTimer_ptr
#define clCreateSubDevices ocl::clCreateSubDevices_ptr
#define clRetainDevice ocl::clRetainDevice_ptr
#define clReleaseDevice ocl::clReleaseDevice_ptr

#endif // OCL_DYNAMIC_LIB_IMPL

#endif // UR_STATIC_ADAPTER_OPENCL
