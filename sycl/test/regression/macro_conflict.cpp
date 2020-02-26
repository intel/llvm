// RUN: %clangxx -fsyntax-only -Xclang -verify %s -o %t.out
// expected-no-diagnostics
//
//===----------------------------------------------------------------------===//
// This test checks if the user-defined macros SUCCESS, FAIL, BLOCKED are
// conflicting with the symbols defined in SYCL header files.
//===----------------------------------------------------------------------===//

#define SUCCESS 0

#include <CL/sycl.hpp>

int main() {
  printf("hello world!\n");
  return 0;
}
