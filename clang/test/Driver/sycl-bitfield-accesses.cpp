// RUN: %clangxx -O0 -fsycl -fsycl-targets=spir64 %s -S -emit-llvm -o- | FileCheck -check-prefix=CHECK-FINE %s
// RUN: %clangxx -O0 -fsycl %s -S -emit-llvm -o- | FileCheck -check-prefix=CHECK-COARSE %s

// CHECK-FINE: %struct.with_bitfield = type { i32, i32, i32, i32 }
// CHECK-COARSE: %struct.with_bitfield = type { i128 }
//
// Tests if fine grained access for SPIR targets is working

struct with_bitfield {
    unsigned int a : 32;
    unsigned int b : 32;
    unsigned int c : 32;
    unsigned int d : 32;
};

#include "sycl.hpp"

using namespace sycl;

int main() {
  sycl::queue queue;
  std::vector<unsigned int> vec(1);
  {
    sycl::buffer<unsigned int> buf(vec.data(), vec.size());
    queue.submit([&](sycl::handler &cgh) {
      sycl::accessor acc(buf, cgh, sycl::write_only, sycl::no_init);
      cgh.single_task<>([=]() {
        with_bitfield A;
        int accum[4];
        for (int i = 0; i < 4; i++) {
          accum[i] = i;
        }
        A.a = accum[0];
	A.b = accum[1];
	A.c = accum[2];
	A.d = accum[3];
        acc[0] = A.a + A.b + A.c + A.d;
      });
    });
    queue.wait();
  }

  return 0;
}
