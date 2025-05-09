// This test is written with an aim to check that experimental::printf behaves
// in the same way as printf from C99/C11
//
// The test is written using conversion specifiers table from cppreference [1]
// [1]: https://en.cppreference.com/w/cpp/io/c/fprintf
//
// UNSUPPORTED: target-amd
// FIXME: The 'short' type gets overflown with sporadic values on CUDA.
// XFAIL: cuda
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/14734

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out | FileCheck %s
// FIXME: Remove dedicated constant address space testing once generic AS
//        support is considered stable.
// RUN: %{build} -o %t.constant.out -DTEST_CONSTANT_AS
// RUN: %{run} %t.constant.out | FileCheck %s

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/builtins.hpp>

#include "helper.hpp"

using namespace sycl;

void do_d_i_test() { // %d, %i signed integer, decimal representation
  // Some reference values
  constexpr char CHAR_VALUE = 0x7B;     // 123
  constexpr short SHORT_VALUE = 0x3039; // 12345
  constexpr int INT_VALUE = 0x499602D3; // 1234567891
  constexpr long LONG_VALUE = INT_VALUE;
  constexpr long long LONG_LONG_VALUE =
      0x112210F4B2D230A2; // 1234567891011121314

  signed char hhd = CHAR_VALUE;
  short hd = SHORT_VALUE;
  int d = INT_VALUE;
  long ld = LONG_VALUE;
  long long lld = LONG_LONG_VALUE;
  intmax_t jd = LONG_LONG_VALUE;
  size_t zd = LONG_LONG_VALUE;
  ptrdiff_t td = LONG_LONG_VALUE;

  FORMAT_STRING(fmt1) = "Decimal positive values:\n"
                        "\tsigned char: %hhd\n"
                        "\tshort: %hd\n"
                        "\tint: %d\n"
                        "\tlong: %ld\n"
                        "\tlong long: %lld\n"
                        "\tintmax_t: %jd\n"
                        "\tsigned size_t: %zd\n"
                        "\tptrdiff_t: %td\n";
  ext::oneapi::experimental::printf(fmt1, hhd, hd, d, ld, lld, jd, zd, td);
  // CHECK: Decimal positive values:
  // CHECK-NEXT: signed char: 123
  // CHECK-NEXT: short: 12345
  // CHECK-NEXT: int: 1234567891
  // CHECK-NEXT: long: 1234567891
  // CHECK-NEXT: long long: 1234567891011121314
  // CHECK-NEXT: intmax_t: 1234567891011121314
  // CHECK-NEXT: signed size_t: 1234567891011121314
  // CHECK-NEXT: ptrdiff_t: 1234567891011121314

  FORMAT_STRING(fmt2) = "Integer positive values:\n"
                        "\tsigned char: %hhi\n"
                        "\tshort: %hi\n"
                        "\tint: %i\n"
                        "\tlong: %li\n"
                        "\tlong long: %lli\n"
                        "\tintmax_t: %ji\n"
                        "\tsigned size_t: %zi\n"
                        "\tptrdiff_t: %ti\n";
  ext::oneapi::experimental::printf(fmt2, hhd, hd, d, ld, lld, jd, zd, td);
  // CHECK: Integer positive values:
  // CHECK-NEXT: signed char: 123
  // CHECK-NEXT: short: 12345
  // CHECK-NEXT: int: 1234567891
  // CHECK-NEXT: long: 1234567891
  // CHECK-NEXT: long long: 1234567891011121314
  // CHECK-NEXT: intmax_t: 1234567891011121314
  // CHECK-NEXT: signed size_t: 1234567891011121314
  // CHECK-NEXT: ptrdiff_t: 1234567891011121314

  hhd = -hhd;
  hd = -hd;
  d = -d;
  ld = -ld;
  lld = -lld;
  jd = -jd;
  zd = -zd;
  td = -td;

  FORMAT_STRING(fmt3) = "Decimal negative values:\n"
                        "\tsigned char: %hhd\n"
                        "\tshort: %hd\n"
                        "\tint: %d\n"
                        "\tlong: %ld\n"
                        "\tlong long: %lld\n"
                        "\tintmax_t: %jd\n"
                        "\tsigned size_t: %zd\n"
                        "\tptrdiff_t: %td\n";
  ext::oneapi::experimental::printf(fmt3, hhd, hd, d, ld, lld, jd, zd, td);
  // CHECK: Decimal negative values:
  // CHECK-NEXT: signed char: -123
  // CHECK-NEXT: short: -12345
  // CHECK-NEXT: int: -1234567891
  // CHECK-NEXT: long: -1234567891
  // CHECK-NEXT: long long: -1234567891011121314
  // CHECK-NEXT: intmax_t: -1234567891011121314
  // CHECK-NEXT: signed size_t: -1234567891011121314
  // CHECK-NEXT: ptrdiff_t: -1234567891011121314

  FORMAT_STRING(fmt4) = "Integer negative values:\n"
                        "\tsigned char: %hhi\n"
                        "\tshort: %hi\n"
                        "\tint: %i\n"
                        "\tlong: %li\n"
                        "\tlong long: %lli\n"
                        "\tintmax_t: %ji\n"
                        "\tsigned size_t: %zi\n"
                        "\tptrdiff_t: %ti\n";
  ext::oneapi::experimental::printf(fmt4, hhd, hd, d, ld, lld, jd, zd, td);
  // CHECK: Integer negative values:
  // CHECK-NEXT: signed char: -123
  // CHECK-NEXT: short: -12345
  // CHECK-NEXT: int: -1234567891
  // CHECK-NEXT: long: -1234567891
  // CHECK-NEXT: long long: -1234567891011121314
  // CHECK-NEXT: intmax_t: -1234567891011121314
  // CHECK-NEXT: signed size_t: -1234567891011121314
  // CHECK-NEXT: ptrdiff_t: -1234567891011121314
}

void do_o_test() { // %o unsigned integer, octal representation
  // Some reference values
  constexpr unsigned char CHAR_VALUE = 0123;
  constexpr unsigned short SHORT_VALUE = 0123456;
  constexpr unsigned int INT_VALUE = 012345670123;
  constexpr unsigned long LONG_VALUE = INT_VALUE;
  constexpr unsigned long long LONG_LONG_VALUE = 01234567012345670123456;

  unsigned char hho = CHAR_VALUE;
  unsigned short ho = SHORT_VALUE;
  unsigned int o = INT_VALUE;
  unsigned long lo = LONG_VALUE;
  unsigned long long llo = LONG_LONG_VALUE;
  uintmax_t jo = LONG_LONG_VALUE;
  size_t zo = LONG_LONG_VALUE;
  ptrdiff_t to = LONG_LONG_VALUE;

  FORMAT_STRING(fmt1) = "Octal:\n"
                        "\tunsigned char: %hho\n"
                        "\tunsigned short: %ho\n"
                        "\tunsigned int: %o\n"
                        "\tunsigned long: %lo\n"
                        "\tunsigned long long: %llo\n"
                        "\tuintmax_t: %jo\n"
                        "\tsize_t: %zo\n"
                        "\tptrdiff_t (unsigned version): %to\n";
  ext::oneapi::experimental::printf(fmt1, hho, ho, o, lo, llo, jo, zo, to);
  // CHECK: Octal:
  // CHECK-NEXT: unsigned char: 123
  // CHECK-NEXT: unsigned short: 123456
  // CHECK-NEXT: unsigned int: 12345670123
  // CHECK-NEXT: unsigned long: 12345670123
  // CHECK-NEXT: unsigned long long: 1234567012345670123456
  // CHECK-NEXT: uintmax_t: 1234567012345670123456
  // CHECK-NEXT: size_t: 1234567012345670123456
  // CHECK-NEXT: ptrdiff_t (unsigned version): 1234567012345670123456
}

void do_x_test() { // %x, %X unsigned integer, hexadecimal representation
  // Some reference values
  constexpr unsigned char CHAR_VALUE = 0x12;
  constexpr unsigned short SHORT_VALUE = 0x1234;
  constexpr unsigned int INT_VALUE = 0x12345678;
  constexpr unsigned long LONG_VALUE = INT_VALUE;
  constexpr unsigned long long LONG_LONG_VALUE = 0x123456789ABCDEF0;

  unsigned char hhx = CHAR_VALUE;
  unsigned short hx = SHORT_VALUE;
  unsigned int x = INT_VALUE;
  unsigned long lx = LONG_VALUE;
  unsigned long long llx = LONG_LONG_VALUE;
  uintmax_t jx = LONG_LONG_VALUE;
  size_t zx = LONG_LONG_VALUE;
  ptrdiff_t tx = LONG_LONG_VALUE;

  FORMAT_STRING(fmt1) = "Hexadecimal:\n"
                        "\tunsigned char: %hhx\n"
                        "\tunsigned short: %hx\n"
                        "\tunsigned int: %x\n"
                        "\tunsigned long: %lx\n"
                        "\tunsigned long long: %llx\n"
                        "\tuintmax_t: %jx\n"
                        "\tsize_t: %zx\n"
                        "\tptrdiff_t: %tx\n";
  ext::oneapi::experimental::printf(fmt1, hhx, hx, x, lx, llx, jx, zx, tx);
  // CHECK: Hexadecimal:
  // CHECK-NEXT: unsigned char: 12
  // CHECK-NEXT: unsigned short: 1234
  // CHECK-NEXT: unsigned int: 12345678
  // CHECK-NEXT: unsigned long: 12345678
  // CHECK-NEXT: unsigned long long: 123456789abcdef0
  // CHECK-NEXT: uintmax_t: 123456789abcdef0
  // CHECK-NEXT: size_t: 123456789abcdef0
  // CHECK-NEXT: ptrdiff_t: 123456789abcdef0

  FORMAT_STRING(fmt2) = "Hexadecimal (capital letters):\n"
                        "\tunsigned char: %hhX\n"
                        "\tunsigned short: %hX\n"
                        "\tunsigned int: %X\n"
                        "\tunsigned long: %lX\n"
                        "\tunsigned long long: %llX\n"
                        "\tuintmax_t: %jX\n"
                        "\tsize_t: %zX\n"
                        "\tptrdiff_t: %tX\n";
  ext::oneapi::experimental::printf(fmt2, hhx, hx, x, lx, llx, jx, zx, tx);
  // CHECK: Hexadecimal (capital letters):
  // CHECK-NEXT: unsigned char: 12
  // CHECK-NEXT: unsigned short: 1234
  // CHECK-NEXT: unsigned int: 12345678
  // CHECK-NEXT: unsigned long: 12345678
  // CHECK-NEXT: unsigned long long: 123456789ABCDEF0
  // CHECK-NEXT: uintmax_t: 123456789ABCDEF0
  // CHECK-NEXT: size_t: 123456789ABCDEF0
  // CHECK-NEXT: ptrdiff_t: 123456789ABCDEF0
}

void do_u_test() { // %u unsigned integer, decimal representation
  // Some reference values
  constexpr char CHAR_VALUE = 0x7B;     // 123
  constexpr short SHORT_VALUE = 0x3039; // 12345
  constexpr int INT_VALUE = 0x499602D3; // 1234567891
  constexpr long LONG_VALUE = INT_VALUE;
  constexpr long long LONG_LONG_VALUE =
      0x112210F4B2D230A2; // 1234567891011121314

  unsigned char hhu = CHAR_VALUE;
  unsigned short hu = SHORT_VALUE;
  unsigned int u = INT_VALUE;
  unsigned long lu = LONG_VALUE;
  unsigned long long llu = LONG_LONG_VALUE;
  uintmax_t ju = LONG_LONG_VALUE;
  size_t zu = LONG_LONG_VALUE;
  ptrdiff_t tu = LONG_LONG_VALUE;

  FORMAT_STRING(fmt1) = "Unsigned decimal:\n"
                        "\tunsigned char: %hhu\n"
                        "\tunsigned short: %hu\n"
                        "\tunsigned int: %u\n"
                        "\tunsigned long: %lu\n"
                        "\tunsigned long long: %llu\n"
                        "\tuintmax_t: %ju\n"
                        "\tsize_t: %zu\n"
                        "\tptrdiff_t: %tu\n";
  ext::oneapi::experimental::printf(fmt1, hhu, hu, u, lu, llu, ju, zu, tu);
  // CHECK: Unsigned decimal:
  // CHECK-NEXT: unsigned char: 123
  // CHECK-NEXT: unsigned short: 12345
  // CHECK-NEXT: unsigned int: 1234567891
  // CHECK-NEXT: unsigned long: 1234567891
  // CHECK-NEXT: unsigned long long: 1234567891011121314
  // CHECK-NEXT: uintmax_t: 1234567891011121314
  // CHECK-NEXT: size_t: 1234567891011121314
  // CHECK-NEXT: ptrdiff_t: 1234567891011121314
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
