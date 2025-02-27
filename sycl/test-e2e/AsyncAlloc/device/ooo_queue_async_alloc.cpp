// REQUIRES: aspect-ext_oneapi_async_memory_alloc

// RUN: %{build} -o %t.out
// RUN: %{run-unfiltered-devices} %t.out

#include <iostream>
#include <sycl/detail/core.hpp>

#include <sycl/ext/oneapi/experimental/async_alloc/async_alloc.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/properties/queue_properties.hpp>

// Uncomment to print additional test information
// #define VERBOSE_PRINT

// Uncomment to validate memory reuse
// This relies on unguaranteed behaviour and should be off by default
// #define MEM_REUSE_CHECK

namespace syclexp = sycl::ext::oneapi::experimental;
class first_kernel;
class second_kernel;

int main() {

  sycl::queue q;
  size_t width = 8;
  std::vector<char> out(width);

#ifdef MEM_REUSE_CHECK
  std::vector<char> expected(width);
  // Expected is the sum of two sets of tids
  // Since freeing doesn't reset the memory
  for (int i = 0; i < width; i++) {
    expected[i] = 2 * i;
  }
#endif

  try {

    // <--- First allocation, use, and free --->

    // Allocate in pool
    void *firstAlloc = nullptr;

    sycl::event e1 = syclexp::submit_with_event(q, [&](sycl::handler &cgh) {
      firstAlloc = syclexp::async_malloc(cgh, sycl::usm::alloc::device, width);
    });

    // Use allocation in kernel
    sycl::event e2 = syclexp::submit_with_event(q, [&](sycl::handler &cgh) {
      cgh.parallel_for<first_kernel>(
          sycl::nd_range<1>{{width}, {width}}, [=](sycl::nd_item<1> it) {
            size_t dim0 = it.get_local_id(0);
            *((char *)firstAlloc + dim0) = static_cast<char>(dim0);
          });
      cgh.depends_on(e1);
    });

    // Free memory back to pool
    sycl::event e3 = syclexp::submit_with_event(q, [&](sycl::handler &cgh) {
      syclexp::async_free(cgh, firstAlloc);
      cgh.depends_on(e2);
    });

    // <--- Second allocation, use, and free --->

    // Re-use allocation in pool
    void *secondAlloc = nullptr;
    sycl::event e4 = syclexp::submit_with_event(q, [&](sycl::handler &cgh) {
      cgh.depends_on(e3);
      secondAlloc = syclexp::async_malloc(cgh, sycl::usm::alloc::device, width);
    });

    // Re-use allocation in kernel
    sycl::event e5 = syclexp::submit_with_event(q, [&](sycl::handler &cgh) {
      cgh.parallel_for<second_kernel>(
          sycl::nd_range<1>{{width}, {width}}, [=](sycl::nd_item<1> it) {
            size_t dim0 = it.get_local_id(0);
            *((char *)secondAlloc + dim0) =
                *((char *)secondAlloc + dim0) + static_cast<char>(dim0);
          });
      cgh.depends_on(e4);
    });

    sycl::event e6 = syclexp::submit_with_event(q, [&](sycl::handler &cgh) {
      syclexp::memcpy(cgh, out.data(), secondAlloc, width);
      cgh.depends_on(e5);
    });

    // Free memory back to pool
    sycl::event e7 = syclexp::submit_with_event(q, [&](sycl::handler &cgh) {
      syclexp::async_free(cgh, secondAlloc);
      cgh.depends_on(e6);
    });

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
