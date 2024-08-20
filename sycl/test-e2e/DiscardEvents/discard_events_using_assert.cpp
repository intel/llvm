// FIXME unsupported on CUDA and HIP until fallback libdevice becomes available
// UNSUPPORTED: cuda || hip
//
// UNSUPPORTED: ze_debug
// RUN: %{build} -o %t.out
//
// RUN: env SYCL_UR_TRACE=2 %{run} %t.out &> %t.txt ; FileCheck %s --input-file %t.txt
//
// The test checks that the last parameter is not `nullptr` for
// urEnqueueKernelLaunch.
//
// CHECK-NOT: ---> urEnqueueKernelLaunch({{.*}}.phEvent = nullptr
// CHECK: ---> urEnqueueKernelLaunch
// CHECK: -> UR_RESULT_SUCCESS
//
// CHECK: The test passed.

#include "discard_events_kernel_using_assert.hpp"
