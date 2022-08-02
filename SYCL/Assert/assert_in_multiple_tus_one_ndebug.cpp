// REQUIRES: linux
// RUN: %clangxx -DSYCL_FALLBACK_ASSERT=1 -fsycl -fsycl-targets=%sycl_triple -DDEFINE_NDEBUG_INFILE2 -I %S/Inputs %S/assert_in_multiple_tus.cpp %S/Inputs/kernels_in_file2.cpp -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out &> %t.txt || true
// RUN: %CPU_RUN_PLACEHOLDER FileCheck %s --input-file %t.txt
// RUN: %GPU_RUN_PLACEHOLDER %t.out &> %t.txt || true
// RUN: %GPU_RUN_PLACEHOLDER FileCheck %s --input-file %t.txt
// Shouldn't fail on ACC as fallback assert isn't enqueued there
// RUN: %ACC_RUN_PLACEHOLDER %t.out &> %t.txt
// RUN: %ACC_RUN_PLACEHOLDER FileCheck %s --check-prefix=CHECK-ACC --input-file %t.txt
//
// CHECK-NOT:  this message from calculus
// CUDA uses block/thread vs global/local id for SYCL, also it shows the
// position of a thread within the block, not the absolute ID.
// CHECK:      {{.*}}assert_in_multiple_tus.hpp:20: int checkFunction(): {{global id: \[5|block: \[1}},0,0],
// CHECK-SAME: {{.*}} [1,0,0] Assertion `X && "Nil in result"` failed.
// CHECK-NOT:  this message from file2
// CHECK-NOT:  The test ended.
//
// CHECK-ACC-NOT: {{.*}}assert_in_multiple_tus.hpp:20: int checkFunction(): {{.*}}
// CHECK-ACC: The test ended.
