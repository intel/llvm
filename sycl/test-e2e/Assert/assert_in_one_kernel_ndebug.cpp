// RUN: %clangxx -fsycl %{sycl_target_opts} -DNDEBUG  %S/assert_in_one_kernel.cpp -o %t.out
// RUN: %{run} %t.out | FileCheck %s
//
// CHECK-NOT: from assert statement
// CHECK: The test ended.
