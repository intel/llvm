// REQUIRES: aspect-usm_shared_allocations

// -- Test for linking object-state SYCLBIN files into an executable-state
// -- SYCLBIN file with -fsycl-link.

// ptxas currently fails to compile images with unresolved symbols.
// XFAIL: target-nvidia
// XFAIL-TRACKER: CMPLRLLVM-68810

// RUN: %clangxx --offload-new-driver -fsyclbin=object %S/Inputs/exporting_function.cpp -o %t.export.syclbin
// RUN: %clangxx --offload-new-driver -fsyclbin=object %S/Inputs/importing_kernel.cpp -o %t.import.syclbin
// RUN: %clangxx -fsycl-link %t.export.syclbin %t.import.syclbin -o %t.syclbin
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out %t.syclbin

#define SYCLBIN_EXECUTABLE_STATE

#include "Inputs/link_syclbin_files.hpp"
