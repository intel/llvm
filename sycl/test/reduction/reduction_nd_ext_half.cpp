// UNSUPPORTED: cuda
// OpenCL C 2.x alike work-group functions not yet supported by CUDA.
//
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// TODO: Enable the test for CPU/ACC when they support half type.
// RUNx: %CPU_RUN_PLACEHOLDER %t.out
// RUNx: %ACC_RUN_PLACEHOLDER %t.out

// TODO: Enable the test for HOST when it supports intel::reduce() and barrier()
// RUNx: env SYCL_DEVICE_TYPE=HOST %t.out

// This test performs basic checks of parallel_for(nd_range, reduction, func)
// used with 'half' type.

#include "reduction_nd_ext_type.hpp"

int main() {
  return runTests<half>("cl_khr_fp16");
}
