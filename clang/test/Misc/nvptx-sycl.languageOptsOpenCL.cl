// REQUIRES: nvptx-registered-target
// RUN: %clang_cc1 -fsycl-is-device -triple nvptx-unknown-unknown %s
// RUN: %clang_cc1 -fsycl-is-device -triple nvptx64-unknown-unknown %s

// OpenCL extensions enabled for NVPTX with SYCL
#ifndef cl_khr_fp16
#error "Missing cl_khr_fp16 define"
#endif
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef cl_khr_int64_base_atomics
#error "Missing cl_khr_int64_base_atomics define"
#endif
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

#ifndef cl_khr_int64_extended_atomics
#error "Missing cl_khr_int64_extended_atomics define"
#endif
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

#ifndef cl_khr_3d_image_writes
#error "Missing cl_khr_3d_image_writes define"
#endif
#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable
