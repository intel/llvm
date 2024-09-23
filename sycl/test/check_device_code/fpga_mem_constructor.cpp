// RUN: %clangxx -fsycl -c -fsycl-device-only -S -emit-llvm %s -o - | FileCheck %s

// Test that the complicated constructor is visible in device code

#include "sycl/sycl.hpp"

using namespace sycl;
namespace intel = sycl::ext::intel::experimental;   // for fpga_mem
namespace oneapi = sycl::ext::oneapi::experimental; // for properties

// CHECK: %class.foo = type { i32, i32 }

class foo {
private:
  int val;

public:
  int secret;

  // complicated constructor
  foo(int val) : val(val) {
    secret = 0;

    for (int i = 0; i < val; i++) {
      secret += static_cast<int>(sycl::sqrt(static_cast<float>(i)));
    }
  }
};

// CHECK: call {{.*}}sqrt

SYCL_EXTERNAL void fpga_mem_constructor() {
  intel::fpga_mem<foo> mem{42};
  volatile int ReadVal = mem.get().secret;
}
