// REQUIRES: native_cpu
// RUN: %clangxx -fsycl-device-only -fsycl-targets=native_cpu %s -O3 -S -o - | FileCheck %s
// RUN: %clangxx -fsycl-device-only -O0 -fsycl-targets=native_cpu %s -S -o - | FileCheck %s

#include <sycl/sycl.hpp>
using namespace sycl;

class Test;
class Test2;
class Test22;
class Test23;
class Test24;
class Test25;
class Test26;
class Test3;
class Test4;

template <typename AccT>
void use_local_acc(nd_item<1> it, AccT &acc,
                   const local_accessor<int, 1> &local_acc) {
  local_acc[it.get_local_id()[0]] = 1;
  acc[it.get_local_id()[0]] = local_acc[0];
}

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
      h.parallel_for<Test2>(
          r, [=](nd_item<1> it) { use_local_acc(it, acc, local_acc); });
    });
    // CHECK-DAG: @_ZTS5Test2({{.*}} !is_nd_range [[MDID:![0-9]*]]
  }
  {
    buffer<int, 1> buf(&res, 1);
    deviceQueue.submit([&](handler &h) {
      auto acc = buf.template get_access<access::mode::write>(h);
      h.parallel_for<Test22>(r, [=](nd_item<1> it) {
        acc[it.get_sub_group().get_group_id()] = 42;
      });
    });
    // CHECK-DAG: @_ZTS6Test22({{.*}} !is_nd_range [[MDID:![0-9]*]]
    deviceQueue.submit([&](handler &h) {
      auto acc = buf.template get_access<access::mode::write>(h);
      h.parallel_for<Test23>(r, [=](nd_item<1> it) {
        acc[it.get_sub_group().get_local_id()] = 42;
      });
    });
    // CHECK-DAG: @_ZTS6Test23({{.*}} !is_nd_range [[MDID:![0-9]*]]

    deviceQueue.submit([&](handler &h) {
      auto acc = buf.template get_access<access::mode::write>(h);
      h.parallel_for<Test24>(r, [=](nd_item<1> it) {
        acc[it.get_sub_group().get_local_range().get(0)] = 42;
      });
    });
    // CHECK-DAG: @_ZTS6Test24({{.*}} !is_nd_range [[MDID:![0-9]*]]

    deviceQueue.submit([&](handler &h) {
      auto acc = buf.template get_access<access::mode::write>(h);
      h.parallel_for<Test25>(r, [=](nd_item<1> it) {
        acc[it.get_sub_group().get_max_local_range().get(0)] = 42;
      });
    });
    // CHECK-DAG: @_ZTS6Test25({{.*}} !is_nd_range [[MDID:![0-9]*]]

    deviceQueue.submit([&](handler &h) {
      auto acc = buf.template get_access<access::mode::write>(h);
      h.parallel_for<Test26>(r, [=](nd_item<1> it) {
        acc[it.get_sub_group().get_group_range().get(0)] = 42;
      });
    });
    // CHECK-DAG: @_ZTS6Test26({{.*}} !is_nd_range [[MDID:![0-9]*]]
  }

  deviceQueue.submit([&](handler &h) {
    h.parallel_for<Test3>(1, [=](item<1> it) { it.get_id(); });
  });
  // CHECK-DAG: @_ZTS5Test3({{.*}} !is_nd_range [[MDNOT:![0-9]*]]

  buffer<int, 1> buf(&res, 1);
  deviceQueue.submit([&](sycl::handler &cgh) {
    auto acc = sycl::accessor(buf, cgh, sycl::write_only);
    cgh.parallel_for_work_group<Test4>(
        range<1>(1), range<1>(1),
        [=](auto group) { acc[group.get_group_id()] = 42; });
  });
  // CHECK-DAG: @_ZTS5Test4({{.*}} !is_nd_range [[MDID:![0-9]*]]
}

//CHECK:[[MDID]] = !{i1 true}
//CHECK:[[MDNOT]] = !{i1 false}
