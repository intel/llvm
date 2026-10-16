// REQUIRES: aspect-usm_shared_allocations

// -- Test for linking input-state SYCLBIN files into an executable-state
// -- SYCLBIN file with -fsycl-link.

// ptxas currently fails to compile images with unresolved symbols.
// XFAIL: target-nvidia
// XFAIL-TRACKER: CMPLRLLVM-68810

// RUN: %clangxx --offload-new-driver -fsyclbin=input %S/Inputs/exporting_function.cpp -o %t.export.syclbin
// RUN: %clangxx --offload-new-driver -fsyclbin=input %S/Inputs/importing_kernel.cpp -o %t.import.syclbin
// RUN: %clangxx -fsycl-link -Wno-unused-command-line-argument %t.export.syclbin %t.import.syclbin -o %t.syclbin
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out %t.syclbin

/// Linking must fail if a SYCL_EXTERNAL function is left undefined.
// RUN: not %clangxx -fsycl-link -Wno-unused-command-line-argument %t.import.syclbin -o %t.undef.syclbin 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK-UNDEF
// CHECK-UNDEF: error: undefined SYCL_EXTERNAL function 'TestFunc(int*, int)' in the SYCLBIN files being linked

/// Executable-state SYCLBIN files cannot be linked.
// RUN: not %clangxx -fsycl-link -Wno-unused-command-line-argument %t.syclbin %t.export.syclbin -o %t.exe.syclbin 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK-EXE
// CHECK-EXE: error: SYCLBIN file {{.*}} is in executable state; only SYCLBIN files in input or object state can be linked

#define SYCLBIN_EXECUTABLE_STATE

#include "Inputs/link_syclbin_files.hpp"
