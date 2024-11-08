// RUN: %{build} -DNDEBUG -o %t.out
//
// RUN: env SYCL_UR_TRACE=2 %{run} %t.out &> %t.txt ; FileCheck %s --input-file %t.txt
//
// The test checks that the last parameter is `nullptr` for
// urEnqueueKernelLaunch.
//
// CHECK: <--- urEnqueueKernelLaunch
// CHECK: .phEvent = nullptr
//
// CHECK: The test passed.

#include "discard_events_kernel_using_assert.hpp"
