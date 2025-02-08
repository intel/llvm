// REQUIRES: opencl, opencl_icd, aspect-usm_shared_allocations, opencl-aot, cpu
// RUN: %{build} -fno-sycl-dead-args-optimization -fsycl-targets=spir64_x86_64 %opencl_lib -o %t.out
// RUN: %{run} %t.out
#include "OCL_common.hpp"
