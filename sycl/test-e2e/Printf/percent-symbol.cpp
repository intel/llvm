// This test is written with an aim to check that experimental::printf behaves
// in the same way as printf from C99/C11
//
// The test is written using conversion specifiers table from cppreference [1]
// [1]: https://en.cppreference.com/w/cpp/io/c/fprintf
//
// UNSUPPORTED: hip_amd
// XFAIL: cuda && windows
//
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out | FileCheck %s
// FIXME: Remove dedicated constant address space testing once generic AS
//        support is considered stable.
// RUN: %{build} -o %t.constant.out -DTEST_CONSTANT_AS
// RUN: %{run} %t.constant.out | FileCheck %s
//
// FIXME: Enable on GPU once %% conversion is supported there
// UNSUPPORTED: gpu
//
// CHECK: %c %s %d %i %o %x %X %u
// CHECK-NEXT: %f %F %e %E %a %A %g %G %n %p

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/builtins.hpp>

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
