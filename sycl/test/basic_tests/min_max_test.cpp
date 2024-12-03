// REQUIRES: windows
// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify %s
// RUN: %clangxx -fsycl -fpreview-breaking-changes -fsycl-device-only -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics

#include "windows.h"

#include <sycl/sycl.hpp>

int main() {
  int tmp = min(1, 4);
  return 0;
}
