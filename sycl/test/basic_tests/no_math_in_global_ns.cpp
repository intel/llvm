// RUN: %clangxx -fsycl  -fsyntax-only -Xclang -verify %s -Xclang -verify-ignore-unexpected=warning,note
// RUN: %clangxx -fsycl  -fsyntax-only -Xclang -verify %s -Xclang -verify-ignore-unexpected=warning,note -fpreview-breaking-changes
// expected-no-diagnostics

// MSVC has the following includes:
// <ostream>
//   -> <ios>
//     -> <xlocnum>
//       -> <cmath>
//
// <functional>
//   -> <unordered_map>
//     -> <xhash>
//       -> <cmath>
//
// <vector> and <string> seem to include <cmath> implicitly as well.
// XFAIL: windows

#include <sycl/sycl.hpp>

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/pipes.hpp>
#include <sycl/ext/intel/fpga_device_selector.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/ext/intel/online_compiler.hpp>
#include <sycl/ext/oneapi/experimental/invoke_simd.hpp>

using namespace sycl;

int main() {
  queue q;
  q.single_task([=] { sqrt(1.0); }).wait();
  return 0;
}
