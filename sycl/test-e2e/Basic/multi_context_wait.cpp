// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// UNSUPPORTED: gpu-intel-dg2
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/18734

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

#include <iostream>
#include <vector>

std::vector<sycl::event> submit_dependencies(sycl::queue q1, sycl::queue q2,
                                             int *mem1, int *mem2) {
  int delay_ops = 1024 * 1024;
  auto delay = [=] {
    volatile int value = delay_ops;
    while (--value)
      ;
  };

  auto ev1 =
      q1.parallel_for(sycl::range<1>(1024), [=]([[maybe_unused]] auto u) {
        delay();
        mem1[u.get_id()] = 1;
      });
  auto ev2 =
      q2.parallel_for(sycl::range<1>(1024), [=]([[maybe_unused]] auto u) {
        delay();
        mem2[u.get_id()] = 2;
      });

  return {ev1, ev2};
}

void test_host_task() {
  sycl::context c1{};
  sycl::context c2{};

  sycl::queue q1(c1, sycl::default_selector_v);
  sycl::queue q2(c2, sycl::default_selector_v);

  auto mem1 = sycl::malloc_host<int>(1024, q1);
  auto mem2 = sycl::malloc_host<int>(1024, q2);

  auto events = submit_dependencies(q1, q2, mem1, mem2);

  q2.submit([&](sycl::handler &cgh) {
    cgh.depends_on(events[0]);
    cgh.depends_on(events[1]);
    cgh.host_task([=]() {
      for (int i = 0; i < 1024; i++) {
        assert(mem1[i] == 1);
        assert(mem2[i] == 2);
      }
    });
  });

  q2.wait();

  sycl::free(mem1, c1);
  sycl::free(mem2, c2);
}

void test_kernel() {
  sycl::context c1{};
  sycl::context c2{};

  sycl::queue q1(c1, sycl::default_selector_v);
  sycl::queue q2(c2, sycl::default_selector_v);

  auto mem1 = sycl::malloc_device<int>(1024, q1);
  auto mem2 = sycl::malloc_device<int>(1024, q2);

  auto events = submit_dependencies(q1, q2, mem1, mem2);

  q2.submit([&](sycl::handler &cgh) {
    cgh.depends_on(events[0]);
    cgh.depends_on(events[1]);
    cgh.parallel_for(sycl::range<1>(1024), [=](auto item) {
      assert(mem1[item.get_id()] == 1);
      assert(mem2[item.get_id()] == 2);
    });
  });

  q2.wait();

  sycl::free(mem1, c1);
  sycl::free(mem2, c2);
}

int main() {
  test_host_task();
  test_kernel();

  return 0;
}
