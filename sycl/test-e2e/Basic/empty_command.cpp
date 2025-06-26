// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

#include <thread>

using namespace sycl;

// std::latch is not available until c++20
class simple_latch {
  std::atomic<unsigned> counter;

public:
  simple_latch(unsigned init) : counter(init) {}

  void wait() {
    // block until the counter reaches zero;
    while (counter)
      std::this_thread::yield();
  }
  void count_down(unsigned by = 1) { counter.fetch_sub(by); }
};

void test_host_task_dep() {
  queue q;

  simple_latch start_execution{1};

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

  simple_latch start_execution{1};
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

  simple_latch start_execution{1};
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
