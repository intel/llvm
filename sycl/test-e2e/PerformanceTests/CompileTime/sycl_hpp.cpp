// RUN: time -f "Elapsed real time: %es" %{build} -fsycl-device-only -fsyntax-only
// RUN: time -f "Elapsed real time: %es" %{build} -o %t.out

#include <sycl/sycl.hpp>

int main() { return 0; }
