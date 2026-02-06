// This test is written with an aim to check that experimental::printf behaves
// in the same way as printf from C99/C11
//
// The test is written using conversion specifiers table from cppreference [1]
// [1]: https://en.cppreference.com/w/cpp/io/c/fprintf
//
// UNSUPPORTED: target-amd
// CUDA device-side printf does not support the hh length modifier and treats
// %hd as double. When running on the CUDA backend, this test avoids hh/%hd and
// uses %d/%i/%o/%x/%X/%u for int-promoted types (char/short) instead.
// See: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#formatted-output

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

void do_d_i_test(bool IsCuda) { // %d, %i signed integer, decimal representation
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

  FORMAT_STRING(fmt1_default) = "Decimal positive values:\n"
                                "\tsigned char: %hhd\n"
                                "\tshort: %hd\n"
                                "\tint: %d\n"
                                "\tlong: %ld\n"
                                "\tlong long: %lld\n"
                                "\tintmax_t: %jd\n"
                                "\tsigned size_t: %zd\n"
                                "\tptrdiff_t: %td\n";
  FORMAT_STRING(fmt1_cuda) = "Decimal positive values:\n"
                             "\tsigned char: %d\n"
                             "\tshort: %d\n"
                             "\tint: %d\n"
                             "\tlong: %ld\n"
                             "\tlong long: %lld\n"
                             "\tintmax_t: %lld\n"
                             "\tsigned size_t: %lld\n"
                             "\tptrdiff_t: %lld\n";
  if (IsCuda) {
    ext::oneapi::experimental::printf(fmt1_cuda, static_cast<int>(hhd),
                                      static_cast<int>(hd), d, ld, lld,
                                      static_cast<long long>(jd),
                                      static_cast<long long>(zd),
                                      static_cast<long long>(td));
  } else {
    ext::oneapi::experimental::printf(fmt1_default, hhd, hd, d, ld, lld, jd,
                                      zd, td);
  }
  // CHECK: Decimal positive values:
  // CHECK-NEXT: signed char: 123
  // CHECK-NEXT: short: 12345
  // CHECK-NEXT: int: 1234567891
  // CHECK-NEXT: long: 1234567891
  // CHECK-NEXT: long long: 1234567891011121314
  // CHECK-NEXT: intmax_t: 1234567891011121314
  // CHECK-NEXT: signed size_t: 1234567891011121314
  // CHECK-NEXT: ptrdiff_t: 1234567891011121314

  FORMAT_STRING(fmt2_default) = "Integer positive values:\n"
                                "\tsigned char: %hhi\n"
                                "\tshort: %hi\n"
                                "\tint: %i\n"
                                "\tlong: %li\n"
                                "\tlong long: %lli\n"
                                "\tintmax_t: %ji\n"
                                "\tsigned size_t: %zi\n"
                                "\tptrdiff_t: %ti\n";
  FORMAT_STRING(fmt2_cuda) = "Integer positive values:\n"
                             "\tsigned char: %i\n"
                             "\tshort: %i\n"
                             "\tint: %i\n"
                             "\tlong: %li\n"
                             "\tlong long: %lli\n"
                             "\tintmax_t: %lli\n"
                             "\tsigned size_t: %lli\n"
                             "\tptrdiff_t: %lli\n";
  if (IsCuda) {
    ext::oneapi::experimental::printf(fmt2_cuda, static_cast<int>(hhd),
                                      static_cast<int>(hd), d, ld, lld,
                                      static_cast<long long>(jd),
                                      static_cast<long long>(zd),
                                      static_cast<long long>(td));
  } else {
    ext::oneapi::experimental::printf(fmt2_default, hhd, hd, d, ld, lld, jd,
                                      zd, td);
  }
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

  FORMAT_STRING(fmt3_default) = "Decimal negative values:\n"
                                "\tsigned char: %hhd\n"
                                "\tshort: %hd\n"
                                "\tint: %d\n"
                                "\tlong: %ld\n"
                                "\tlong long: %lld\n"
                                "\tintmax_t: %jd\n"
                                "\tsigned size_t: %zd\n"
                                "\tptrdiff_t: %td\n";
  FORMAT_STRING(fmt3_cuda) = "Decimal negative values:\n"
                             "\tsigned char: %d\n"
                             "\tshort: %d\n"
                             "\tint: %d\n"
                             "\tlong: %ld\n"
                             "\tlong long: %lld\n"
                             "\tintmax_t: %lld\n"
                             "\tsigned size_t: %lld\n"
                             "\tptrdiff_t: %lld\n";
  if (IsCuda) {
    ext::oneapi::experimental::printf(fmt3_cuda, static_cast<int>(hhd),
                                      static_cast<int>(hd), d, ld, lld,
                                      static_cast<long long>(jd),
                                      static_cast<long long>(zd),
                                      static_cast<long long>(td));
  } else {
    ext::oneapi::experimental::printf(fmt3_default, hhd, hd, d, ld, lld, jd,
                                      zd, td);
  }
  // CHECK: Decimal negative values:
  // CHECK-NEXT: signed char: -123
  // CHECK-NEXT: short: -12345
  // CHECK-NEXT: int: -1234567891
  // CHECK-NEXT: long: -1234567891
  // CHECK-NEXT: long long: -1234567891011121314
  // CHECK-NEXT: intmax_t: -1234567891011121314
  // CHECK-NEXT: signed size_t: -1234567891011121314
  // CHECK-NEXT: ptrdiff_t: -1234567891011121314

  FORMAT_STRING(fmt4_default) = "Integer negative values:\n"
                                "\tsigned char: %hhi\n"
                                "\tshort: %hi\n"
                                "\tint: %i\n"
                                "\tlong: %li\n"
                                "\tlong long: %lli\n"
                                "\tintmax_t: %ji\n"
                                "\tsigned size_t: %zi\n"
                                "\tptrdiff_t: %ti\n";
  FORMAT_STRING(fmt4_cuda) = "Integer negative values:\n"
                             "\tsigned char: %i\n"
                             "\tshort: %i\n"
                             "\tint: %i\n"
                             "\tlong: %li\n"
                             "\tlong long: %lli\n"
                             "\tintmax_t: %lli\n"
                             "\tsigned size_t: %lli\n"
                             "\tptrdiff_t: %lli\n";
  if (IsCuda) {
    ext::oneapi::experimental::printf(fmt4_cuda, static_cast<int>(hhd),
                                      static_cast<int>(hd), d, ld, lld,
                                      static_cast<long long>(jd),
                                      static_cast<long long>(zd),
                                      static_cast<long long>(td));
  } else {
    ext::oneapi::experimental::printf(fmt4_default, hhd, hd, d, ld, lld, jd,
                                      zd, td);
  }
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

void do_o_test(bool IsCuda) { // %o unsigned integer, octal representation
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

  FORMAT_STRING(fmt1_default) = "Octal:\n"
                                "\tunsigned char: %hho\n"
                                "\tunsigned short: %ho\n"
                                "\tunsigned int: %o\n"
                                "\tunsigned long: %lo\n"
                                "\tunsigned long long: %llo\n"
                                "\tuintmax_t: %jo\n"
                                "\tsize_t: %zo\n"
                                "\tptrdiff_t (unsigned version): %to\n";
  FORMAT_STRING(fmt1_cuda) = "Octal:\n"
                             "\tunsigned char: %o\n"
                             "\tunsigned short: %o\n"
                             "\tunsigned int: %o\n"
                             "\tunsigned long: %lo\n"
                             "\tunsigned long long: %llo\n"
                             "\tuintmax_t: %llo\n"
                             "\tsize_t: %llo\n"
                             "\tptrdiff_t (unsigned version): %llo\n";
  if (IsCuda) {
    ext::oneapi::experimental::printf(
        fmt1_cuda, static_cast<unsigned int>(hho),
        static_cast<unsigned int>(ho), o, lo, llo,
        static_cast<unsigned long long>(jo),
        static_cast<unsigned long long>(zo),
        static_cast<unsigned long long>(to));
  } else {
    ext::oneapi::experimental::printf(fmt1_default, hho, ho, o, lo, llo, jo, zo,
                                      to);
  }
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

void do_x_test(bool IsCuda) { // %x, %X unsigned integer, hexadecimal representation
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

  FORMAT_STRING(fmt1_default) = "Hexadecimal:\n"
                                "\tunsigned char: %hhx\n"
                                "\tunsigned short: %hx\n"
                                "\tunsigned int: %x\n"
                                "\tunsigned long: %lx\n"
                                "\tunsigned long long: %llx\n"
                                "\tuintmax_t: %jx\n"
                                "\tsize_t: %zx\n"
                                "\tptrdiff_t: %tx\n";
  FORMAT_STRING(fmt1_cuda) = "Hexadecimal:\n"
                             "\tunsigned char: %x\n"
                             "\tunsigned short: %x\n"
                             "\tunsigned int: %x\n"
                             "\tunsigned long: %lx\n"
                             "\tunsigned long long: %llx\n"
                             "\tuintmax_t: %llx\n"
                             "\tsize_t: %llx\n"
                             "\tptrdiff_t: %llx\n";
  if (IsCuda) {
    ext::oneapi::experimental::printf(
        fmt1_cuda, static_cast<unsigned int>(hhx),
        static_cast<unsigned int>(hx), x, lx, llx,
        static_cast<unsigned long long>(jx),
        static_cast<unsigned long long>(zx),
        static_cast<unsigned long long>(tx));
  } else {
    ext::oneapi::experimental::printf(fmt1_default, hhx, hx, x, lx, llx, jx, zx,
                                      tx);
  }
  // CHECK: Hexadecimal:
  // CHECK-NEXT: unsigned char: 12
  // CHECK-NEXT: unsigned short: 1234
  // CHECK-NEXT: unsigned int: 12345678
  // CHECK-NEXT: unsigned long: 12345678
  // CHECK-NEXT: unsigned long long: 123456789abcdef0
  // CHECK-NEXT: uintmax_t: 123456789abcdef0
  // CHECK-NEXT: size_t: 123456789abcdef0
  // CHECK-NEXT: ptrdiff_t: 123456789abcdef0

  FORMAT_STRING(fmt2_default) = "Hexadecimal (capital letters):\n"
                                "\tunsigned char: %hhX\n"
                                "\tunsigned short: %hX\n"
                                "\tunsigned int: %X\n"
                                "\tunsigned long: %lX\n"
                                "\tunsigned long long: %llX\n"
                                "\tuintmax_t: %jX\n"
                                "\tsize_t: %zX\n"
                                "\tptrdiff_t: %tX\n";
  FORMAT_STRING(fmt2_cuda) = "Hexadecimal (capital letters):\n"
                             "\tunsigned char: %X\n"
                             "\tunsigned short: %X\n"
                             "\tunsigned int: %X\n"
                             "\tunsigned long: %lX\n"
                             "\tunsigned long long: %llX\n"
                             "\tuintmax_t: %llX\n"
                             "\tsize_t: %llX\n"
                             "\tptrdiff_t: %llX\n";
  if (IsCuda) {
    ext::oneapi::experimental::printf(
        fmt2_cuda, static_cast<unsigned int>(hhx),
        static_cast<unsigned int>(hx), x, lx, llx,
        static_cast<unsigned long long>(jx),
        static_cast<unsigned long long>(zx),
        static_cast<unsigned long long>(tx));
  } else {
    ext::oneapi::experimental::printf(fmt2_default, hhx, hx, x, lx, llx, jx, zx,
                                      tx);
  }
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

void do_u_test(bool IsCuda) { // %u unsigned integer, decimal representation
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

  FORMAT_STRING(fmt1_default) = "Unsigned decimal:\n"
                                "\tunsigned char: %hhu\n"
                                "\tunsigned short: %hu\n"
                                "\tunsigned int: %u\n"
                                "\tunsigned long: %lu\n"
                                "\tunsigned long long: %llu\n"
                                "\tuintmax_t: %ju\n"
                                "\tsize_t: %zu\n"
                                "\tptrdiff_t: %tu\n";
  FORMAT_STRING(fmt1_cuda) = "Unsigned decimal:\n"
                             "\tunsigned char: %u\n"
                             "\tunsigned short: %u\n"
                             "\tunsigned int: %u\n"
                             "\tunsigned long: %lu\n"
                             "\tunsigned long long: %llu\n"
                             "\tuintmax_t: %llu\n"
                             "\tsize_t: %llu\n"
                             "\tptrdiff_t: %llu\n";
  if (IsCuda) {
    ext::oneapi::experimental::printf(
        fmt1_cuda, static_cast<unsigned int>(hhu),
        static_cast<unsigned int>(hu), u, lu, llu,
        static_cast<unsigned long long>(ju),
        static_cast<unsigned long long>(zu),
        static_cast<unsigned long long>(tu));
  } else {
    ext::oneapi::experimental::printf(fmt1_default, hhu, hu, u, lu, llu, ju, zu,
                                      tu);
  }
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

  const bool IsCuda = (q.get_backend() == backend::ext_oneapi_cuda);

  q.submit([&](handler &cgh) {
    cgh.single_task<IntTest>([=]() {
      do_d_i_test(IsCuda);
      do_o_test(IsCuda);
      do_x_test(IsCuda);
      do_u_test(IsCuda);
    });
  });
  q.wait();

  return 0;
}
