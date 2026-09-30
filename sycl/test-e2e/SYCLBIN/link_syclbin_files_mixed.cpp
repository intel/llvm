// REQUIRES: aspect-usm_shared_allocations

// -- Test for linking an input-state and an object-state SYCLBIN file into an
// -- executable-state SYCLBIN file with -fsycl-link, using the default output
// -- file name.

// ptxas currently fails to compile images with unresolved symbols. Disable for
// other targets than SPIR-V until this has been resolved. (CMPLRLLVM-68810)
// REQUIRES: target-spir

// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: %clangxx --offload-new-driver -fsyclbin=input %S/Inputs/exporting_function.cpp -o %t.dir/export.syclbin
// RUN: %clangxx --offload-new-driver -fsyclbin=object %S/Inputs/importing_kernel.cpp -o %t.dir/import.syclbin
// RUN: cd %t.dir && %clangxx -fsycl-link export.syclbin import.syclbin
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out %t.dir/a.syclbin

#define SYCLBIN_EXECUTABLE_STATE

#include "Inputs/link_syclbin_files.hpp"
