// REQUIRES: opencl, opencl_icd, aspect-usm_shared_allocations, !accelerator
// RUN: %{build} -fno-sycl-dead-args-optimization %opencl_lib -o %t.out
// RUN: %{run} %t.out
#include "OCL_common.hpp"
