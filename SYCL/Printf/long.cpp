// This test is written with an aim to check that experimental::printf behaves
// in the same way as printf from C99/C11
//
// The test is written using conversion specifiers table from cppreference [1]
// [1]: https://en.cppreference.com/w/cpp/io/c/fprintf
//
// FIXME: Once the Windows OpenCL CPU/ACC support is fixed, merge this test's
// contents into the common integer test.
// UNSUPPORTED: (windows && (cpu || accelerator)) || hip_amd
//
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out %CPU_CHECK_PLACEHOLDER
// RUN: %GPU_RUN_PLACEHOLDER %t.out %GPU_CHECK_PLACEHOLDER
// RUN: %ACC_RUN_PLACEHOLDER %t.out %ACC_CHECK_PLACEHOLDER
// FIXME: Remove dedicated constant address space testing once generic AS
//        support is considered stable.
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.constant.out \
// RUN: -DTEST_CONSTANT_AS
// RUN: %CPU_RUN_PLACEHOLDER %t.constant.out %CPU_CHECK_PLACEHOLDER
// RUN: %GPU_RUN_PLACEHOLDER %t.constant.out %GPU_CHECK_PLACEHOLDER
// RUN: %ACC_RUN_PLACEHOLDER %t.constant.out %ACC_CHECK_PLACEHOLDER

#include <sycl/sycl.hpp>

#include "helper.hpp"

using namespace sycl;

void do_d_i_test() { // %d, %i signed integer, decimal representation
  // Some reference values
  constexpr int INT_VALUE = 0x499602D3; // 1234567891
  constexpr long LONG_VALUE = INT_VALUE;
  constexpr long long LONG_LONG_VALUE =
      0x112210F4B2D230A2; // 1234567891011121314

  long ld = LONG_VALUE;
  long long lld = LONG_LONG_VALUE;

  FORMAT_STRING(fmt1) = "Decimal positive values:\n"
                        "\tlong int: %ld\n"
                        "\tlong long int: %lld\n";
  ext::oneapi::experimental::printf(fmt1, ld, lld);
  // CHECK: Decimal positive values:
  // CHECK-NEXT: long int: 1234567891
  // CHECK-NEXT: long long int: 1234567891011121314

  FORMAT_STRING(fmt2) = "Integer positive values:\n"
                        "\tlong int: %li\n"
                        "\tlong long int: %lli\n";
  ext::oneapi::experimental::printf(fmt2, ld, lld);
  // CHECK: Integer positive values:
  // CHECK-NEXT: long int: 1234567891
  // CHECK-NEXT: long long int: 1234567891011121314

  ld = -ld;
  lld = -lld;

  FORMAT_STRING(fmt3) = "Decimal negative values:\n"
                        "\tlong int: %ld\n"
                        "\tlong long int: %lld\n";
  ext::oneapi::experimental::printf(fmt3, ld, lld);
  // CHECK: Decimal negative values:
  // CHECK-NEXT: long int: -1234567891
  // CHECK-NEXT: long long int: -1234567891011121314

  FORMAT_STRING(fmt4) = "Integer negative values:\n"
                        "\tlong int: %li\n"
                        "\tlong long int: %lli\n";
  ext::oneapi::experimental::printf(fmt4, ld, lld);
  // CHECK: Integer negative values:
  // CHECK-NEXT: long int: -1234567891
  // CHECK-NEXT: long long int: -1234567891011121314
}

void do_o_test() { // %o unsigned integer, octal representation
  // Some reference values
  constexpr unsigned int INT_VALUE = 012345670123;
  constexpr unsigned long LONG_VALUE = INT_VALUE;
  constexpr unsigned long long LONG_LONG_VALUE = 01234567012345670123456;

  unsigned long lo = LONG_VALUE;
  unsigned long long llo = LONG_LONG_VALUE;

  FORMAT_STRING(fmt1) = "Octal:\n"
                        "\tunsigned long: %lo\n"
                        "\tunsigned long long: %llo\n";
  ext::oneapi::experimental::printf(fmt1, lo, llo);
  // CHECK: Octal:
  // CHECK-NEXT: unsigned long: 12345670123
  // CHECK-NEXT: unsigned long long: 1234567012345670123456
}

void do_x_test() { // %x, %X unsigned integer, hexadecimal representation
  // Some reference values
  constexpr unsigned int INT_VALUE = 0x12345678;
  constexpr unsigned long LONG_VALUE = INT_VALUE;
  constexpr unsigned long long LONG_LONG_VALUE = 0x123456789ABCDEF0;

  unsigned long lx = LONG_VALUE;
  unsigned long long llx = LONG_LONG_VALUE;

  FORMAT_STRING(fmt1) = "Hexadecimal:\n"
                        "\tunsigned long: %lx\n"
                        "\tunsigned long long: %llx\n";
  ext::oneapi::experimental::printf(fmt1, lx, llx);
  // CHECK: Hexadecimal:
  // CHECK-NEXT: unsigned long: 12345678
  // CHECK-NEXT: unsigned long long: 123456789abcdef0

  FORMAT_STRING(fmt2) = "Hexadecimal (capital letters):\n"
                        "\tunsigned long int: %lX\n"
                        "\tunsigned long long int: %llX\n";
  ext::oneapi::experimental::printf(fmt2, lx, llx);
  // CHECK: Hexadecimal (capital letters):
  // CHECK-NEXT: unsigned long int: 12345678
  // CHECK-NEXT: unsigned long long int: 123456789ABCDEF0
}

void do_u_test() { // %u unsigned integer, decimal representation
  // Some reference values
  constexpr int INT_VALUE = 0x499602D3; // 1234567891
  constexpr long LONG_VALUE = INT_VALUE;
  constexpr long long LONG_LONG_VALUE =
      0x112210F4B2D230A2; // 1234567891011121314

  unsigned long lu = LONG_VALUE;
  unsigned long long llu = LONG_LONG_VALUE;

  FORMAT_STRING(fmt1) = "Unsigned decimal:\n"
                        "\tunsigned long int: %lu\n"
                        "\tunsigned long long int: %llu\n";
  ext::oneapi::experimental::printf(fmt1, lu, llu);
  // CHECK: Unsigned decimal:
  // CHECK-NEXT: unsigned long int: 1234567891
  // CHECK-NEXT: unsigned long long int: 1234567891011121314
}

class IntTest;

int main() {
  queue q;

  q.submit([](handler &cgh) {
    cgh.single_task<IntTest>([]() {
      do_d_i_test();
      do_o_test();
      do_x_test();
      do_u_test();
    });
  });
  q.wait();

  return 0;
}
