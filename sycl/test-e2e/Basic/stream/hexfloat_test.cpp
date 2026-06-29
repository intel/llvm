// REQUIRES: aspect-fp64
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out | FileCheck %s

// Test for issue #22057: hexfloat formatting should match host behavior

#include <iomanip>
#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/stream.hpp>

using namespace sycl;

int main() {
  queue q;

  float f = 3.14159f;
  double d = 2.71828;
  half h = 3.14f;
  ext::oneapi::bfloat16 bf = 2.71f;

  // Print expected host hex format
  std::cout << "Host hexfloat:" << std::endl;
  std::cout << "float: " << std::hexfloat << f << std::endl;
  std::cout << "double: " << std::hexfloat << d << std::endl;
  std::cout << "half: " << std::hexfloat << h << std::endl;
  std::cout << "bfloat16: " << std::hexfloat << bf << std::endl;

  // CHECK: Host hexfloat:
  // CHECK: float: 0x1.921fap+1
  // CHECK: double: 0x1.5bf0995aaf79p+1
  // CHECK: half: 0x1.92p+1
  // CHECK: bfloat16: 0x1.5ap+1

  // Print device hex format
  q.submit([&](handler &cgh) {
     stream out(1024, 256, cgh);
     cgh.parallel_for(range<1>(1), [=](id<1>) {
       out << "Device hexfloat:" << endl;
       out << hexfloat << "float: " << f << endl;
       out << hexfloat << "double: " << d << endl;
       out << hexfloat << "half: " << h << endl;
       out << hexfloat << "bfloat16: " << bf << endl;
     });
   }).wait();

  // CHECK: Device hexfloat:
  // CHECK: float: 0x1.921fa0p+1
  // CHECK: double: 0x1.5bf0995aaf790p+1
  // CHECK: half: 0x1.920000p+1
  // CHECK: bfloat16: 0x1.5a0000p+1

  return 0;
}
