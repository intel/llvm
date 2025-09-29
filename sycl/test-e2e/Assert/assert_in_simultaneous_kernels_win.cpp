// REQUIRES: windows
// RUN: %{build} -o %t.out %threads_lib
//
// Since this is a multi-threaded application enable memory tracking and
// deferred release feature in the Level Zero adapter to avoid releasing memory
// too early. This is necessary because currently SYCL RT sets indirect access
// flag for all kernels and the Level Zero runtime doesn't support deferred
// release yet.
//
// DEFINE: %{gpu_env} = env SYCL_PI_LEVEL_ZERO_TRACK_INDIRECT_ACCESS_MEMORY=1 SYCL_PI_SUPPRESS_ERROR_MESSAGE=1

// RUN: %if gpu %{ %{gpu_env} %} %{run} %t.out &> %t.txt ; FileCheck %s --input-file %t.txt
//
// FIXME Windows version prints '(null)' instead of '<unknown func>' once in a
// while for some insane reason.
// CHECK:      {{.*}}assert_in_simultaneous_kernels.hpp:16: {{<unknown func>|\(null\)}}: global id: [9,7,0], local id: [0,0,0]
// CHECK-SAME: Assertion `false && "from assert statement"` failed.
// CHECK-NOT:  The test ended.

#include "assert_in_simultaneous_kernels.hpp"
