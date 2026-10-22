// REQUIRES: aspect-usm_device_allocations
// XFAIL: windows && run-mode && cuda
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/23547

// -- Test for compiling and loading a kernel bundle with a SYCLBIN containing
//    the use of optional kernel features.

// RUN: %clangxx --offload-new-driver -fsyclbin=executable %{sycl_target_opts} %{syclbin_exec_opts} %S/Inputs/optional_kernel_features.cpp -o %t.syclbin
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out %t.syclbin

#define SYCLBIN_EXECUTABLE_STATE

#include "Inputs/optional_kernel_features.hpp"
