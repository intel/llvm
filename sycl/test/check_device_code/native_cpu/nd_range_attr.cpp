// REQUIRES: native_cpu
// RUN: %clangxx -fsycl-device-only -fsycl-targets=native_cpu %s -S -o - | FileCheck %s

#include <sycl/sycl.hpp>
using namespace sycl;

class Test;
class Test2;
class Test3;

int main() {
  sycl::queue deviceQueue;
  sycl::nd_range<1> r(1, 1);
  deviceQueue.submit([&](handler &h) {
    h.parallel_for<Test>(r, [=](nd_item<1> it) { it.barrier(); });
  });
  // CHECK-DAG: @_ZTS4Test({{.*}} !is_nd_range [[MDID:![0-9]*]]

  int res = 0;
  {
    buffer<int, 1> buf(&res, 1);
    deviceQueue.submit([&](handler &h) {
      auto acc = buf.template get_access<access::mode::write>(h);
      local_accessor<int, 1> local_acc(1, h);
      h.parallel_for<Test2>(r, [=](nd_item<1> it) {
        local_acc[0] = 1;
        acc[0] = local_acc[0];
      });
    });
    // CHECK-DAG: @_ZTS5Test2({{.*}} !is_nd_range [[MDID:![0-9]*]]
  }
  deviceQueue.submit([&](handler &h) {
    h.parallel_for<Test3>(1, [=](item<1> it) { it.get_id(); });
  });
  // CHECK-DAG: @_ZTS5Test3({{.*}} !is_nd_range [[MDNOT:![0-9]*]]

}

//CHECK:[[MDID]] = !{i1 true}
//CHECK:[[MDNOT]] = !{i1 false}
