// REQUIRES: aspect-usm_device_allocations

// -- Basic test for compiling and loading a SYCLBIN kernel_bundle in executable
// -- state.

// RUN: %clangxx --offload-new-driver -fsyclbin=executable %{sycl_target_opts} %{syclbin_exec_opts} %S/Inputs/basic_kernel.cpp -o %t.syclbin
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out %t.syclbin

#define SYCLBIN_EXECUTABLE_STATE

#include "Inputs/basic.hpp"
