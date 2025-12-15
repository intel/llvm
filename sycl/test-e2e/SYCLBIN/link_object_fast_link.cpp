// REQUIRES: aspect-usm_shared_allocations

// -- Test for linking two SYCLBIN kernel_bundle.

// REQUIRES: target-spir

// RUN: %clangxx --offload-new-driver -fsycl-rdc -fsyclbin=object %{syclbin_exec_opts} -fsycl-allow-device-image-dependencies %S/Inputs/exporting_function.cpp -o %t.export_w_aot.syclbin
// RUN: %clangxx --offload-new-driver -fsycl-rdc -fsyclbin=object %{syclbin_exec_opts} -fsycl-allow-device-image-dependencies %S/Inputs/importing_kernel.cpp -o %t.import_w_aot.syclbin
// RUN: %clangxx --offload-new-driver -fsyclbin=object -fsycl-allow-device-image-dependencies %S/Inputs/exporting_function.cpp -o %t.export.syclbin
// RUN: %clangxx --offload-new-driver -fsyclbin=object -fsycl-allow-device-image-dependencies %S/Inputs/importing_kernel.cpp -o %t.import.syclbin
// RUN: %{build} -o %t.out

// RUN: %{run} %t.out %t.export.syclbin %t.import.syclbin
// RUN: %{run} %t.out %t.export_w_aot.syclbin %t.import_w_aot.syclbin
// RUN: %{run} %t.out %t.export.syclbin %t.import_w_aot.syclbin
// RUN: %{run} %t.out %t.export_w_aot.syclbin %t.import.syclbin

#define SYCLBIN_OBJECT_STATE
#define SYCLBIN_USE_FAST_LINK

#include "Inputs/link.hpp"
