// RUN: %{build} -o %t.out %cxx_std_optionc++20
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

#include <latch>
#include <thread>

using namespace sycl;

void test_host_task_dep() {
  queue q;

  std::latch start_execution{1};

  int x = 0;

  auto host_event = q.submit([&](handler &cgh) {
    cgh.host_task([&]() {
      start_execution.wait();
      x = 42;
    });
  });

  auto empty_cg_event =
      q.submit([&](handler &cgh) { cgh.depends_on(host_event); });

  assert(x == 0);
  start_execution.count_down();

  empty_cg_event.wait();
  assert(x == 42);
}

void test_device_event_dep() {
  queue q;

  std::latch start_execution{1};
  auto *p = sycl::malloc_shared<int>(1, q);
  *p = 0;

  auto host_event = q.submit(
      [&](handler &cgh) { cgh.host_task([&]() { start_execution.wait(); }); });
  auto device_event = q.single_task(host_event, [=]() { *p = 42; });
  auto empty_cg_event =
      q.submit([&](handler &cgh) { cgh.depends_on(device_event); });

  assert(*p == 0);
  start_execution.count_down();

  empty_cg_event.wait();
  assert(*p == 42);

  sycl::free(p, q);
}

void test_accessor_dep() {
  queue q;

  std::latch start_execution{1};
  auto *p = sycl::malloc_shared<int>(1, q);
  *p = 0;

  auto host_event = q.submit(
      [&](handler &cgh) { cgh.host_task([&]() { start_execution.wait(); }); });

  sycl::buffer<int, 1> b{1};
  auto device_event = q.submit([&](auto &cgh) {
    cgh.depends_on(host_event);
    sycl::accessor a{b, cgh};

    cgh.single_task([=]() {
      *p = 42;
      a[0] = 42;
    });
  });
  auto empty_cg_event =
      q.submit([&](handler &cgh) { sycl::accessor a{b, cgh}; });

  assert(*p == 0);
  start_execution.count_down();

  empty_cg_event.wait();
  assert(*p == 42);

  sycl::free(p, q);
}

int main() {
  test_host_task_dep();
  test_device_event_dep();
  test_accessor_dep();
  return 0;
}
