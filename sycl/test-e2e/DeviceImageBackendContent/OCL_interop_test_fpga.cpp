// REQUIRES: opencl, opencl_icd, opencl-aot, accelerator, aspect-usm_shared_allocations
// RUN: %clangxx -fsycl -fno-sycl-dead-args-optimization %opencl_lib -fsycl-targets=spir64_fpga %S/OCL_common.cpp -o %t.out
// RUN: %{run} %t.out
