//==--- queue_submit.cpp - SYCL queue submit test --------------==//

// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

// This test submits same kernel via multiple threads to the same queue.
// It's a regression test for CMPLRLLVM-72408

#include <iostream>
#include <sycl/sycl.hpp>
#include <thread>

#define DIMS 1024

class kernel_set_value;

void submit(sycl::queue *queue, sycl::kernel *kernel) {
  int data[DIMS];
  try {
    sycl::buffer<int, 1> result_buf{data, sycl::range<1>{DIMS}};
    queue->submit([&](sycl::handler &cgh) {
      auto result_acc =
          result_buf.get_access<sycl::access::mode::discard_write>(cgh);
      cgh.set_arg(0, result_acc);
      cgh.parallel_for(sycl::range<1>{DIMS}, *kernel);
    });
    queue->wait_and_throw();
  } catch (sycl::exception &e) {
    std::cerr << "Exception thrown: " << e.what() << "\n";
    return;
  }

  for (int i = 0; i < DIMS; i++) {
    if (data[i] != i) {
      std::cerr << "data[" << i << "] != " << i << " (got " << data[i] << ")\n";
    }
  }
}

class single_queue_with_kernel {
public:
  single_queue_with_kernel(size_t n)
      : numThreads(n), queue(sycl::default_selector_v),
        kernel(sycl::get_kernel_bundle<sycl::bundle_state::executable>(
                   queue.get_context())
                   .get_kernel(sycl::get_kernel_id<kernel_set_value>())) {}

  void run_threads() {
    if (0) {
      sycl::buffer<int, 1> result_buf{sycl::range<1>{DIMS}};
      queue.submit([&](sycl::handler &cgh) {
        auto result_acc =
            result_buf.get_access<sycl::access::mode::discard_write>(cgh);
        cgh.parallel_for<kernel_set_value>(
            sycl::range<1>{DIMS},
            [=](sycl::id<1> idx) { result_acc[idx] = idx[0]; });
      });
    }

    std::vector<std::thread *> threads;
    for (size_t i = 0; i < numThreads; ++i)
      threads.emplace_back(new std::thread(&submit, &queue, &kernel));

    for (auto &t : threads) {
      t->join();
      delete t;
    }
  }

private:
  sycl::queue queue;
  sycl::kernel kernel;
  size_t numThreads;
};

int main() {
  single_queue_with_kernel test_q(10);
  test_q.run_threads();
  return 0;
}
