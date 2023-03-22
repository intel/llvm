// REQUIRES: windows
// TODO: Re-enable the test when https://github.com/intel/llvm/issues/8717 fixed.
// XFAIL: windows
// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify %s -I %sycl_include
// expected-no-diagnostics

#include "windows.h"

#include "sycl.hpp"

int main() {
  int tmp = min(1, 4);
  return 0;
}
