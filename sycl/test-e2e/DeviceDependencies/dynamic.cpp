// Test -fsycl-allow-device-dependencies with dynamic libraries.

// REQUIRES: linux
// UNSUPPORTED: cuda || hip

// RUN: %clangxx -fsycl -fPIC -shared -fsycl-allow-device-dependencies %S/Inputs/a.cpp -I %S/Inputs -o %T/libdevice_a.so
// RUN: %clangxx -fsycl -fPIC -shared -fsycl-allow-device-dependencies %S/Inputs/b.cpp -I %S/Inputs -o %T/libdevice_b.so
// RUN: %clangxx -fsycl -fPIC -shared -fsycl-allow-device-dependencies %S/Inputs/c.cpp -I %S/Inputs -o %T/libdevice_c.so
// RUN: %clangxx -fsycl -fPIC -shared -fsycl-allow-device-dependencies %S/Inputs/d.cpp -I %S/Inputs -o %T/libdevice_d.so
// RUN: %{build} -fsycl-allow-device-dependencies -L%T -ldevice_a -ldevice_b -ldevice_c -ldevice_d -I %S/Inputs -o %t.out -Wl,-rpath=%T
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include "a.hpp"
#include <iostream>

using namespace sycl;

class ExeKernel;

int main() {
  int val = 0;
  {
    buffer<int, 1> buf(&val, range<1>(1));
    queue q;
    q.submit([&](handler &cgh) {
      auto acc = buf.get_access(cgh);
      cgh.single_task<ExeKernel>([=]() {acc[0] = levelA(acc[0]);});
    });
  }

  std::cout << "val=" << std::hex << val << "\n";
  if (val!=0xDCBA)
    return (1);  
  return(0);
}
