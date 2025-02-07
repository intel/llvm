// REQUIRES: opencl, opencl_icd, aspect-usm_shared_allocations, !accelerator
// RUN: %clangxx -fsycl -fno-sycl-dead-args-optimization %opencl_lib %S/OCL_common.cpp -o %t.out
// RUN: %{run} %t.out
