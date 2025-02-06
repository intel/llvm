// REQUIRES: aspect-ext_oneapi_async_memory_alloc

// RUN: %{build} -o %t.out
// RUN: %{run-unfiltered-devices} %t.out

#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/properties/queue_properties.hpp>

#include <sycl/ext/oneapi/experimental/async_alloc/async_alloc.hpp>

// Uncomment to print additional test information
// #define VERBOSE_PRINT

// Uncomment to validate memory reuse
// This relies on unguaranteed behaviour and is off by default
// #define MEM_REUSE_CHECK

namespace syclexp = sycl::ext::oneapi::experimental;
class first_kernel;
class second_kernel;

int main() {

  sycl::device dev;
  sycl::property_list q_prop{sycl::property::queue::in_order()};
  sycl::queue q(q_prop);
  sycl::context ctx = q.get_context();
  size_t width = 8;
  std::vector<char> out(width);

#ifdef MEM_REUSE_CHECK
  // Expected is the sum of two sets of tids
  // Since freeing doesn't reset the memory
  std::vector<char> expected(width);
  for (int i = 0; i < width; i++) {
    expected[i] = 2 * i;
  }
#endif

  try {

    // Create pool
    syclexp::memory_pool memPool(ctx, dev, sycl::usm::alloc::device);

    // <--- First allocation, use, and free --->

    // Allocate in pool
    void *firstAlloc = syclexp::async_malloc_from_pool(q, width, memPool);

    // Use allocation in kernel
    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<first_kernel>(
          sycl::nd_range<1>{{width}, {width}}, [=](sycl::nd_item<1> it) {
            size_t dim0 = it.get_local_id(0);
            *((char *)firstAlloc + dim0) = static_cast<char>(dim0);
          });
    });

    // Free memory back to pool
    syclexp::async_free(q, firstAlloc);

    // <--- Second allocation, use, and free --->

    // Re-use allocation in pool
    void *secondAlloc = syclexp::async_malloc_from_pool(q, width, memPool);

    // Re-use allocation in kernel
    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<second_kernel>(
          sycl::nd_range<1>{{width}, {width}}, [=](sycl::nd_item<1> it) {
            size_t dim0 = it.get_local_id(0);
            *((char *)secondAlloc + dim0) =
                *((char *)secondAlloc + dim0) + static_cast<char>(dim0);
          });
    });

    q.memcpy(out.data(), secondAlloc, width);

    // Free memory back to pool
    syclexp::async_free(q, secondAlloc);

    // Wait and thus release memory back to OS
    q.wait_and_throw();

  } catch (sycl::exception e) {
    std::cerr << "SYCL exception caught! : " << e.what() << "\n";
    return 2;
  } catch (...) {
    std::cerr << "Unknown exception caught!\n";
    return 3;
  }

  bool validated = true;
#ifdef MEM_REUSE_CHECK
  for (int i = 0; i < width; i++) {
    bool mismatch = false;
    if (out[i] != expected[i]) {
      mismatch = true;
      validated = false;
    }

    if (mismatch) {
#ifdef VERBOSE_PRINT
      std::cerr << "Result mismatch! Expected: "
                << static_cast<int>(expected[i])
                << ", Actual: " << static_cast<int>(out[i]) << std::endl;
#else
      break;
#endif
    }
  }
#endif

  if (validated) {
    std::cout << "Test passed!" << std::endl;
    return 0;
  }

  std::cerr << "Test failed!" << std::endl;
  return 1;
}
