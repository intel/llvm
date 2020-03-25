// REQUIRES: windows
// RUN: %clangxx -fsyntax-only -Xclang -verify %s -I %sycl_include -Xclang
// expected-no-diagnostics

#include "windows.h"
#include "CL/sycl.hpp"

int main() {
  int tmp = min(1, 4);
  return 0;
}
