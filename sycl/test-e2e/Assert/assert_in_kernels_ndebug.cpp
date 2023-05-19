// RUN: %clangxx -fsycl -fsycl-targets=%{sycl_triple} -DNDEBUG %S/assert_in_kernels.cpp -o %t.out
// RUN: %{run} %t.out | FileCheck %s
//
// CHECK-NOT: One shouldn't see this message
// CHECK-NOT: from assert statement
// CHECK-NOT: test aborts earlier, one shouldn't see this message
// CHECK: The test ended.
