// FIXME unsupported on CUDA and HIP until fallback libdevice becomes available
// UNSUPPORTED: cuda || hip
// RUN: %clangxx -DSYCL_FALLBACK_ASSERT=1 -fsycl -fsycl-targets=%sycl_triple -DDEFINE_NDEBUG_INFILE2 -I %S/Inputs %S/assert_in_simultaneously_multiple_tus.cpp %S/Inputs/kernels_in_file2.cpp -o %t.out %threads_lib
// RUN: %CPU_RUN_PLACEHOLDER %t.out &> %t.txt || true
// RUN: %CPU_RUN_PLACEHOLDER FileCheck %s --input-file %t.txt
//
// Since this is a multi-threaded application enable memory tracking and
// deferred release feature in the Level Zero plugin to avoid releasing memory
// too early. This is necessary because currently SYCL RT sets indirect access
// flag for all kernels and the Level Zero runtime doesn't support deferred
// release yet.
// RUN: env SYCL_PI_LEVEL_ZERO_TRACK_INDIRECT_ACCESS_MEMORY=1 %GPU_RUN_PLACEHOLDER %t.out &> %t.txt || true
// RUN: %GPU_RUN_PLACEHOLDER FileCheck %s --input-file %t.txt
// Shouldn't fail on ACC as fallback assert isn't enqueued there
// RUN: %ACC_RUN_PLACEHOLDER %t.out &> %t.txt
// RUN: %ACC_RUN_PLACEHOLDER FileCheck %s --check-prefix=CHECK-ACC --input-file %t.txt
//
// CHECK:      this message from file1
// CHECK-NOT:  this message from file2
// CHECK-NOT:  The test ended.
//
// CHECK-ACC: The test ended.
