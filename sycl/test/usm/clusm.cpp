// RUN: %clangxx -fsycl %s -o %t1.out -lOpenCL
// RUN: %CPU_RUN_PLACEHOLDER %t1.out -cpu
//==------------------- clusm.cpp - CLUSM API test --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/clusm.hpp>

#include "findplatforms.hpp"

#include <cstring>
#include <iostream>

using namespace cl::sycl::detail::usm;

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Please specify -cpu or -gpu!" << std::endl;
    return 1;
  }

  bool preferCPU = false;
  bool preferGPU = false;

  if (!strcmp(argv[1], "-cpu")) {
    preferCPU = true;
    preferGPU = false;
  }
  else if (!strcmp(argv[1], "-gpu")) {
    preferGPU = true;
    preferCPU = false;
  }
  
  CLUSM* clusm = GetCLUSM();

  cl_device_type  deviceType =
    (preferCPU)
    ? CL_DEVICE_TYPE_CPU
    : CL_DEVICE_TYPE_GPU;

  cl_int errorCode;
  cl_platform_id platform;
  cl_device_id device;

  if (!findPlatformAndDevice(deviceType, platform, device)) {
    return 2;
  }

  cl_context_properties ctxtProps[] =
  {
    CL_CONTEXT_PLATFORM,
    (cl_context_properties) platform,
    0
  };
  
  cl_context context = clCreateContextFromType(
    ctxtProps,
    deviceType,
    nullptr,
    nullptr,
    &errorCode);

  if (errorCode != CL_SUCCESS) return 4;

  cl_command_queue queue = clCreateCommandQueueWithProperties(context,
                                                              device,
                                                              nullptr,
                                                              &errorCode);
  
  if (errorCode != CL_SUCCESS) return 5;
  
  cl_int *f;
  f = (cl_int*) clusm->deviceMemAlloc(context,
                                      device,
                                      nullptr,
                                      sizeof(cl_int),
                                      0,
                                      &errorCode);
  
  if (errorCode != CL_SUCCESS) return 6;

  cl_int h_f = 42;

  errorCode = clEnqueueSVMMemcpy(queue,
                                 CL_TRUE,
                                 f,
                                 &h_f,
                                 sizeof(cl_int),
                                 0,
                                 nullptr,
                                 nullptr);

  if (errorCode != CL_SUCCESS) return 7;

  cl_int h_g;

  errorCode = clEnqueueSVMMemcpy(queue,
                                 CL_TRUE,
                                 &h_g,
                                 f,
                                 sizeof(cl_int),
                                 0,
                                 nullptr,
                                 nullptr);

  if (errorCode != CL_SUCCESS) return 8;

  if (h_g != h_f ) return 9;

  errorCode = clusm->memFree(context,  f);

  if (errorCode != CL_SUCCESS) return 10;

  cl_int *h;
  h = (cl_int*) clusm->hostMemAlloc(context,
                                    nullptr,
                                    sizeof(cl_int),
                                    0,
                                    &errorCode);
  
  if (errorCode != CL_SUCCESS) return 11;

  errorCode = clusm->memFree(context,  h);
  
  if (errorCode != CL_SUCCESS) return 12;

  cl_int *s;
  s = (cl_int*) clusm->sharedMemAlloc(context,
                                      device,
                                      nullptr,
                                      sizeof(cl_int),
                                      0,
                                      &errorCode);

  if (errorCode != CL_SUCCESS) return 13;

  errorCode = clusm->memFree(context,  s);
  
  if (errorCode != CL_SUCCESS) return 14;

  
  return 0;
}
