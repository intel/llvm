// REQUIRES: (opencl || level_zero)
// REQUIRES: aspect-usm_shared_allocations

// -- Same cross-origin link scenario as link_rtc_bidir_object.cpp but the
// -- SYCLBIN is loaded in input state and compiled at runtime, exercising
// -- the same merged-image lookup path on the link result.

// RUN: %clangxx --offload-new-driver -fsyclbin=input %S/Inputs/importing_kernel_obj.cpp -o %t.syclbin
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out %t.syclbin

#define SYCLBIN_INPUT_STATE

#include "Inputs/link_rtc_bidir.hpp"
