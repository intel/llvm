// https://github.com/intel/llvm/issues/12797
// UNSUPPORTED: windows
// REQUIRES: windows
// RUN: %{build} -DSYCL_FALLBACK_ASSERT=1 -o %t.out %threads_lib
//
// Since this is a multi-threaded application enable memory tracking and
// deferred release feature in the Level Zero plugin to avoid releasing memory
// too early. This is necessary because currently SYCL RT sets indirect access
// flag for all kernels and the Level Zero runtime doesn't support deferred
// release yet.
//
// DEFINE: %{gpu_env} = env SYCL_PI_LEVEL_ZERO_TRACK_INDIRECT_ACCESS_MEMORY=1 SYCL_PI_SUPPRESS_ERROR_MESSAGE=1

// Shouldn't fail on ACC as fallback assert isn't enqueued there
// RUN: %if gpu %{ %{gpu_env} %} %{run} %t.out &> %t.txt ; FileCheck %s --input-file %t.txt %if acc %{ --check-prefix=CHECK-ACC %}
//
// FIXME Windows version prints '(null)' instead of '<unknown func>' once in a
// while for some insane reason.
// CHECK:      {{.*}}assert_in_simultaneous_kernels.hpp:16: {{<unknown func>|(null)}}: global id: [9,7,0], local id: [0,0,0]
// CHECK-SAME: Assertion `false && "from assert statement"` failed.
// CHECK-NOT:  The test ended.
//
// CHECK-ACC-NOT: {{.*}}assert_in_simultaneous_kernels.hpp:16: {{<unknown func>|(null)}}: global id: [9,7,0], local id: [0,0,0]
// CHECK-ACC:  The test ended.

#include "assert_in_simultaneous_kernels.hpp"
