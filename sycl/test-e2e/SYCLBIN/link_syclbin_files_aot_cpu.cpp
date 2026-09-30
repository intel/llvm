// REQUIRES: aspect-usm_shared_allocations, opencl-aot, cpu, opencl-cpu-rt

// -- Test for linking SYCLBIN files into an executable-state SYCLBIN file with
// -- -fsycl-link, compiling the linked device code ahead of time for the CPU
// -- with --offload-arch.

// RUN: %clangxx --offload-new-driver -fsyclbin=input %S/Inputs/exporting_function.cpp -o %t.export.syclbin
// RUN: %clangxx --offload-new-driver -fsyclbin=object %S/Inputs/importing_kernel.cpp -o %t.import.syclbin
// RUN: %clangxx -fsycl-link --offload-arch=corei7 %t.export.syclbin %t.import.syclbin -o %t.syclbin
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out %t.syclbin

#define SYCLBIN_EXECUTABLE_STATE

#include "Inputs/link_syclbin_files.hpp"
