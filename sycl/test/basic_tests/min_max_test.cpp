// REQUIRES: windows
// RUN: %clangxx -fsycl %s -o %t.out

#include "windows.h"
#include "CL/sycl.hpp"

int main() {

  sycl::device device{sycl::default_selector()};
  int tmp = min(1, 4);
  return 0;
}
