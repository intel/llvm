//==--- queue_submit.cpp - SYCL queue submit test --------------==//

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test submits same kernel via multiple threads to the same queue.
// It's a regression test for CMPLRLLVM-72408

#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>
#include <thread>

#define DIMS 1024

class kernel_set_value;

void submit(sycl::queue &queue, sycl::kernel &kernel) {
  int data[DIMS];
  try {
    sycl::buffer<int, 1> result_buf{data, sycl::range<1>{DIMS}};
    queue.submit([&](sycl::handler &cgh) {
      auto result_acc =
          result_buf.get_access<sycl::access::mode::discard_write>(cgh);
      cgh.set_arg(0, result_acc);
      cgh.parallel_for(sycl::range<1>{DIMS}, kernel);
    });
    queue.wait_and_throw();
  } catch (sycl::exception &e) {
    std::cerr << "Exception thrown: " << e.what() << "\n";
    return;
  }

  for (int i = 0; i < DIMS; i++)
    assert(data[i] == i);
}

void run_test(size_t numThreads) {
  sycl::queue queue(sycl::default_selector_v);
  sycl::kernel kernel =
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(
          queue.get_context())
          .get_kernel(sycl::get_kernel_id<kernel_set_value>());

  // Warm up.
  {
    sycl::buffer<int, 1> result_buf{sycl::range<1>{DIMS}};
    queue
        .submit([&](sycl::handler &cgh) {
          auto result_acc =
              result_buf.get_access<sycl::access::mode::discard_write>(cgh);
          cgh.parallel_for<kernel_set_value>(
              sycl::range<1>{DIMS},
              [=](sycl::id<1> idx) { result_acc[idx] = idx[0]; });
        })
        .wait_and_throw();
  }

  // Spawn multiple threads submitting the same kernel to the same queue.
  std::vector<std::thread> threads;
  for (size_t i = 0; i < numThreads; ++i)
    threads.push_back(std::thread(&submit, std::ref(queue), std::ref(kernel)));

  for (auto &t : threads)
    t.join();
}

int main() {
  run_test(10);
  return 0;
}
