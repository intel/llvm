// This test is written with an aim to check that experimental::printf behaves
// in the same way as printf from C99/C11
//
// The test is written using conversion specifiers table from cppreference [1]
// [1]: https://en.cppreference.com/w/cpp/io/c/fprintf
//
// UNSUPPORTED: hip_amd
//
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out %CPU_CHECK_PLACEHOLDER
// RUN: %ACC_RUN_PLACEHOLDER %t.out %ACC_CHECK_PLACEHOLDER
// FIXME: Remove dedicated constant address space testing once generic AS
//        support is considered stable.
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.constant.out \
// RUN: -DTEST_CONSTANT_AS
// RUN: %CPU_RUN_PLACEHOLDER %t.constant.out %CPU_CHECK_PLACEHOLDER
// RUN: %ACC_RUN_PLACEHOLDER %t.constant.out %ACC_CHECK_PLACEHOLDER
//
// FIXME: Enable on GPU once %% conversion is supported there
// RUNx: %GPU_RUN_PLACEHOLDER %t.out %GPU_CHECK_PLACEHOLDER
// RUNx: %GPU_RUN_PLACEHOLDER %t.constant.out %GPU_CHECK_PLACEHOLDER
//
// CHECK: %c %s %d %i %o %x %X %u
// CHECK-NEXT: %f %F %e %E %a %A %g %G %n %p

#include <sycl/sycl.hpp>

#include <cstring>

#include "helper.hpp"

using namespace sycl;

void do_percent_specifier_test() {
  {
    FORMAT_STRING(fmt) = "%%c %%s %%d %%i %%o %%x %%X %%u\n"
                         "%%f %%F %%e %%E %%a %%A %%g %%G %%n %%p\n";
    ext::oneapi::experimental::printf(fmt);
  }
}

class PercentTest;

int main() {
  queue q;

  q.submit([](handler &cgh) {
    cgh.single_task<PercentTest>([]() { do_percent_specifier_test(); });
  });
  q.wait();

  return 0;
}
