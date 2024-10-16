// REQUIRES: windows

// RUN: %clangxx --driver-mode=cl /std:c++17 /EHsc -I%sycl_include -I%opencl_include_dir %s -o %t.out /link /defaultlib:%sycl_static_libs_dir/sycl.lib
// RUN: %{run} %t.out

// This test checks that if program is linked with version-agnostic import library sycl.lib then sycl program works as expected.
// It is expected to crash if correct dll can't be loaded.

#include <iostream>
#include <sycl/detail/core.hpp>

using namespace sycl;

int main() {
  queue q;
}
