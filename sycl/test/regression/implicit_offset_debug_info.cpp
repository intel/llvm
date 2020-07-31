// RUN: %clangxx -g -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// REQUIRES: cuda

// NOTE: Tests that the implicit global offset pass copies debug information

#include <CL/sycl.hpp>
using namespace cl::sycl;

int main() {
  queue q;
  buffer<uint64_t, 1> t1(10);
  q.submit([&](handler &cgh) {
    auto table = t1.get_access<access::mode::write>(cgh);
    cgh.parallel_for<class kernel>(10, [=](id<1> gtid) {
      table[gtid] = gtid[0];
    });
  });
  q.wait();
}