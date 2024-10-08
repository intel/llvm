// This test is written with an aim to check that experimental::printf behaves
// in the same way as printf from C99/C11
//
// The test is written using conversion specifiers table from cppreference [1]
// [1]: https://en.cppreference.com/w/cpp/io/c/fprintf
//
// UNSUPPORTED: hip_amd
//
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out | FileCheck %s
// FIXME: Remove dedicated constant address space testing once generic AS
//        support is considered stable.
// RUN: %{build} -o %t.constant.out -DTEST_CONSTANT_AS
// RUN: %{run} %t.constant.out | FileCheck %s
//
// FIXME: wchar_t* is not supported on GPU
// FIXME: String literal prefixes (L, u8, u, U) are not functioning on Windows
//
// CHECK: c=a
// CHECK: literal strings: s=Hello World!
// CHECK_DISABLED: non-literal strings: s=Hello, World! ls=

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/builtins.hpp>

#include <cstring>

#include "helper.hpp"

using namespace sycl;

void do_char_string_test() {
  {
    // %c format specifier, single character
    FORMAT_STRING(fmt) = "c=%c\n"; // FIXME: lc is not tested

    char c = 'a';

    ext::oneapi::experimental::printf(fmt, c);
  }

  {
    // %s format specifier, character string
    FORMAT_STRING(fmt1) = "literal strings: s=%s %s\n";
    ext::oneapi::experimental::printf(fmt1, "Hello",
                                      // World!
                                      "\x57\x6F\x72\x6C\x64\x21");

    // FIXME: lack of support for non-literal strings in %s is an OpenCL
    // limitation
    /*
    FORMAT_STRING(fmt2) = "non-literal strings: s=%s ls=%ls\n";
    char str[20] = { '\0' };
    const char *s = "Hello, World!";
    for (int i = 0; i < 13; ++i) {
      str[i] = s[i];
    }

    // FIXME: ls is untested here
    ext::oneapi::experimental::printf(fmt2, str, "");
    */
  }
}

class CharTest;

int main() {
  queue q;

  q.submit([](handler &cgh) {
    cgh.single_task<CharTest>([]() { do_char_string_test(); });
  });
  q.wait();

  return 0;
}
