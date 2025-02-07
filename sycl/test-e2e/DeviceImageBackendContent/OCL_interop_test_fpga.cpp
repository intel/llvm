// REQUIRES: opencl, opencl_icd, opencl-aot, accelerator, aspect-usm_shared_allocations
// RUN: %{build} -fno-sycl-dead-args-optimization %opencl_lib -fsycl-targets=spir64_fpga -o %t.out
// RUN: %{run} %t.out
#include "OCL_common.hpp"
