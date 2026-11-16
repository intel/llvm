// RUN: python3 %S/../perf_compile.py --name "sycl.hpp, device syntax-only" -- %{build} -fsycl-device-only -fsyntax-only
// RUN: python3 %S/../perf_compile.py --name "sycl.hpp, full build" -- %{build} -o %t.out

// PERF-DESCRIPTION: Compile time of an empty program including sycl/sycl.hpp.

#include <sycl/sycl.hpp>

int main() { return 0; }
