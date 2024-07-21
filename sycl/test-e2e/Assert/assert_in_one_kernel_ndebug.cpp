// https://github.com/intel/llvm/issues/7634
// UNSUPPORTED: hip
// RUN: %clangxx -fsycl -fsycl-targets=%{sycl_triple} -DNDEBUG  %S/assert_in_one_kernel.cpp -o %t.out
// RUN: %{run} %t.out | FileCheck %s
//
// CHECK-NOT: from assert statement
// CHECK: The test ended.
