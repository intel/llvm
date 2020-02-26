// RUN: %clangxx -fsyntax-only -Xjclang -verify %s -o %t.out
// expected-no-diagnostics
//
//===----------------------------------------------------------------------===//
// This test checks if the user-defined macros SUCCESS, FAIL, BLOCKED are
// defined in global namespace by sycl.hpp
//===----------------------------------------------------------------------===//

#define SUCCESS 0
#define FAIL 1
#define BLOCKED 2

#include <CL/sycl.hpp>

int main() {
  printf("hello world!\n");
  return 0;
}
