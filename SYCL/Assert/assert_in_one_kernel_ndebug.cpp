// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -DNDEBUG  %S/assert_in_one_kernel.cpp -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out %CPU_CHECK_PLACEHOLDER
// RUN: %GPU_RUN_PLACEHOLDER %t.out %GPU_CHECK_PLACEHOLDER
// RUN: %ACC_RUN_PLACEHOLDER %t.out %ACC_CHECK_PLACEHOLDER
//
// CHECK-NOT: from assert statement
// CHECK: The test ended.
