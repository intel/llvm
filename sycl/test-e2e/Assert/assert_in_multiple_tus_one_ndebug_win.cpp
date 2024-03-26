// https://github.com/intel/llvm/issues/12797
// UNSUPPORTED: windows
// REQUIRES: windows
// RUN: %clangxx -DSYCL_FALLBACK_ASSERT=1 -fsycl -fsycl-targets=%{sycl_triple} -DDEFINE_NDEBUG_INFILE2 -I %S/Inputs %S/assert_in_multiple_tus.cpp %S/Inputs/kernels_in_file2.cpp -o %t.out
// Shouldn't fail on ACC as fallback assert isn't enqueued there
// RUN: %{run} %t.out &> %t.txt ; FileCheck %s --input-file %t.txt %if acc %{ --check-prefix=CHECK-ACC %}
//
// CHECK-NOT:  this message from calculus
// FIXME Windows version prints '(null)' instead of '<unknown func>' once in a
// while for some insane reason.
// CHECK:      {{.*}}assert_in_multiple_tus.hpp:22: {{<unknown func>|(null)}}: {{.*}} [5,0,0],
// CHECK-SAME: {{.*}} [1,0,0] Assertion `X && "Nil in result"` failed.
// CHECK-NOT:  this message from file2
// CHECK-NOT:  The test ended.
//
// CHECK-ACC-NOT: {{.*}}assert_in_multiple_tus.hpp:22: {{<unknown func>|(null)}}: {{.*}} [5,0,0],
// CHECK-ACC: The test ended.
