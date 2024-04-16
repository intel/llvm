// RUN: %{build} %threads_lib -o %t.out
// RUN: %{run} %t.out

// Check that ext_oneapi_submit_barrier works fine in the scenarios
// when provided waitlist consists of only empty events.

#include <iostream>
#include <mutex>
#include <thread>

#include <sycl/detail/core.hpp>

#include <sycl/properties/all_properties.hpp>

static constexpr int niter = 1024;
static constexpr int nthreads = 2;

std::array<std::mutex, nthreads> mutexes;
std::array<std::optional<sycl::event>, nthreads> events;

void threadFunction(int tid) {
  sycl::device dev;
  std::cout << dev.get_info<sycl::info::device::name>() << std::endl;
  sycl::context ctx{dev};
  sycl::queue q1{ctx, dev, {sycl::property::queue::in_order()}};
  sycl::queue q2{ctx, dev, {sycl::property::queue::in_order()}};
  for (int i = 0; i < niter; i++) {
    sycl::event ev1 = q1.ext_oneapi_submit_barrier();
    q2.ext_oneapi_submit_barrier({ev1});
    sycl::event ev2 = q2.ext_oneapi_submit_barrier();
    q1.ext_oneapi_submit_barrier({ev2});
  }
}

int main() {
  std::array<std::thread, nthreads> threads;

  for (int i = 0; i < nthreads; i++) {
    threads[i] = std::thread{threadFunction, i};
  }

  for (int i = 0; i < nthreads; i++) {
    threads[i].join();
  }
  std::cout << "All threads have finished." << std::endl;

  return 0;
}
