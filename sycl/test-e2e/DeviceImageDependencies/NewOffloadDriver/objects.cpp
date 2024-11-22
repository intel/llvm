// Test -fsycl-allow-device-image-dependencies with objects.

// UNSUPPORTED: cuda || hip
// UNSUPPORTED-INTENDED: Not implemented yet for Nvidia/AMD backends.

// RUN: %clangxx --offload-new-driver -fsycl %S/Inputs/a.cpp -I %S/Inputs -c -o %t_a.o
// RUN: %clangxx --offload-new-driver -fsycl %S/Inputs/b.cpp -I %S/Inputs -c -o %t_b.o
// RUN: %clangxx --offload-new-driver -fsycl %S/Inputs/c.cpp -I %S/Inputs -c -o %t_c.o
// RUN: %clangxx --offload-new-driver -fsycl %S/Inputs/d.cpp -I %S/Inputs -c -o %t_d.o
// RUN: %{build} --offload-new-driver -fsycl-allow-device-image-dependencies %t_a.o %t_b.o %t_c.o %t_d.o -I %S/Inputs -o %t.out
// RUN: %{run} %t.out

#include "a.hpp"
#include <iostream>
#include <sycl/detail/core.hpp>

using namespace sycl;

class ExeKernel;

int main() {
  int val = 0;
  {
    buffer<int, 1> buf(&val, range<1>(1));
    queue q;
    q.submit([&](handler &cgh) {
      auto acc = buf.get_access(cgh);
      cgh.single_task<ExeKernel>([=]() { acc[0] = levelA(acc[0]); });
    });
  }

  std::cout << "val=" << std::hex << val << "\n";
  if (val != 0xDCBA)
    return (1);
  return (0);
}
