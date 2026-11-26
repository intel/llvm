// RUN: python3 %S/../perf_compile.py --benchmark-name "core.hpp, device syntax-only" -- %{build} -fsycl-device-only -fsyntax-only
// RUN: python3 %S/../perf_compile.py --benchmark-name "core.hpp, full build" -- %{build} -o %t.out

// BENCHMARK-DESCRIPTION: Compile time of an empty program with detail/core.hpp.

#include <sycl/detail/core.hpp>

int main() { return 0; }
