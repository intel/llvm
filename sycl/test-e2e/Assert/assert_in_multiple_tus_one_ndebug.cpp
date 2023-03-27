// REQUIRES: linux

// https://github.com/intel/llvm/issues/7634
// UNSUPPORTED: hip

// RUN: %clangxx -DSYCL_FALLBACK_ASSERT=1 -fsycl -fsycl-targets=%sycl_triple -DDEFINE_NDEBUG_INFILE2 -I %S/Inputs %S/assert_in_multiple_tus.cpp %S/Inputs/kernels_in_file2.cpp -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out &> %t.cpu.txt || true
// RUN: %CPU_RUN_PLACEHOLDER FileCheck %s --input-file %t.cpu.txt
// RUN: %GPU_RUN_PLACEHOLDER %t.out &> %t.gpu.txt || true
// RUN: %GPU_RUN_PLACEHOLDER FileCheck %s --input-file %t.gpu.txt
// Shouldn't fail on ACC as fallback assert isn't enqueued there
// RUN: %ACC_RUN_PLACEHOLDER %t.out &> %t.acc.txt
// RUN: %ACC_RUN_PLACEHOLDER FileCheck %s --check-prefix=CHECK-ACC --input-file %t.acc.txt
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
