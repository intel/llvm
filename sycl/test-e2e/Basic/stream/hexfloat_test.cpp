// RUN: %{build} -o %t.out
// RUN: %{run} %t.out | FileCheck %s

// Test for issue #22057: hexfloat formatting should match host behavior

#include <iomanip>
#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/stream.hpp>

using namespace sycl;

int main() {
  queue q;

  float f = 3.14159f;
  double d = 2.71828;

  // Print expected host hex format
  std::cout << "Host hexfloat:" << std::endl;
  std::cout << "float: " << std::hexfloat << f << std::endl;
  std::cout << "double: " << std::hexfloat << d << std::endl;

  // CHECK: Host hexfloat:
  // CHECK: float: 0x1.{{[0-9a-f]+}}p+{{[0-9]+}}
  // CHECK: double: 0x1.{{[0-9a-f]+}}p+{{[0-9]+}}

  // Print device hex format
  q.submit([&](handler &h) {
     stream out(1024, 256, h);
     h.parallel_for(range<1>(1), [=](id<1>) {
       out << "Device hexfloat:" << endl;
       out << hexfloat << "float: " << f << endl;
       out << hexfloat << "double: " << d << endl;
     });
   }).wait();

  // CHECK: Device hexfloat:
  // CHECK: float: 0x1.{{[0-9a-f]+}}p+{{[0-9]+}}
  // CHECK: double: 0x1.{{[0-9a-f]+}}p+{{[0-9]+}}

  return 0;
}
