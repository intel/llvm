// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <iostream>
#include <sycl/atomic_ref.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

int main() {

  sycl::device dev;
  sycl::queue q(dev);
  auto ctxt = q.get_context();

  // This test does not validate any output
  // Only that the work_item scope does not error
  try {

    // Allocate device memory
    int *data = sycl::malloc_device<int>(1, q);

    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(10, [=](sycl::id<> id) {
        data[0] = 0;

        // Check atomic_ref functionality
        sycl::atomic_ref<int, sycl::memory_order::relaxed,
                         sycl::memory_scope::work_item,
                         sycl::access::address_space::generic_space>
            at(data[0]);

        auto lock = at.is_lock_free();
        at.store(1);
        auto load = at.load();
        auto xch = at.exchange(2);
        auto weak =
            at.compare_exchange_weak(data[0], 3, sycl::memory_order::relaxed,
                                     sycl::memory_order::relaxed);
        auto strong =
            at.compare_exchange_strong(data[0], 4, sycl::memory_order::relaxed,
                                       sycl::memory_order::relaxed);
        auto fetch_add = at.fetch_add(5);
        auto fetch_sub = at.fetch_sub(6);
        auto fetch_and = at.fetch_and(7);
        auto fetch_or = at.fetch_or(8);
        auto fetch_xor = at.fetch_xor(9);
        auto fetch_min = at.fetch_min(10);
        auto fetch_max = at.fetch_max(11);
      });
    });
    q.wait_and_throw();

    sycl::free(data, q);

  } catch (sycl::exception e) {
    std::cerr << "SYCL exception caught! : " << e.what() << "\n";
    return 1;
  } catch (...) {
    std::cerr << "Unknown exception caught!\n";
    return 2;
  }

  std::cout << "Test passed!" << std::endl;
  return 0;
}
