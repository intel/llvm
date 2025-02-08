// REQUIRES: opencl, opencl_icd, aspect-usm_shared_allocations, ocloc, gpu
// RUN: %{build} -fno-sycl-dead-args-optimization -fsycl-targets=spir64_gen %opencl_lib -o %t.out
// RUN: %{run} %t.out
#include "OCL_common.hpp"
