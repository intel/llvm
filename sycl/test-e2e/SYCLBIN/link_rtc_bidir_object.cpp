// REQUIRES: (opencl || level_zero)
// REQUIRES: aspect-usm_shared_allocations

// -- Cross-origin link: SYCLBIN object provides a kernel that imports a
// -- SYCL_EXTERNAL function from a runtime-compiled SYCL source bundle.
// -- Both kernels (RTC-origin and SYCLBIN-origin) must be reachable after
// -- sycl::link({RTC_obj, SYCLBIN_obj}).
//
// -- Regression test for the case where tryGetExtensionKernel routed all
// -- lookups on the merged image through the RTC ProgramManager and made
// -- SYCLBIN-origin kernels unreachable.

// RUN: %clangxx --offload-new-driver -fsyclbin=object -fsycl-allow-device-image-dependencies %S/Inputs/importing_kernel_obj.cpp -o %t.syclbin
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out %t.syclbin

#define SYCLBIN_OBJECT_STATE

#include "Inputs/link_rtc_bidir.hpp"
