// REQUIRES: (opencl || level_zero)
// REQUIRES: aspect-usm_shared_allocations

// -- Test for linking where one kernel is runtime-compiled and one is compiled
// -- to SYCLBIN.

// RUN: %clangxx --offload-new-driver -fsyclbin=object -fsycl-allow-device-image-dependencies %S/Inputs/exporting_function.cpp -o %t.syclbin
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out %t.syclbin

#define SYCLBIN_OBJECT_STATE

#include "Inputs/link_rtc.hpp"
