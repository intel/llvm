// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

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

  sycl::queue Q;
  size_t Width = 8;
  std::vector<char> Out(Width);

#ifdef MEM_REUSE_CHECK
  std::vector<char> Expected(Width);
  // Expected is the sum of two sets of tids
  // Since freeing doesn't reset the memory
  for (int i = 0; i < Width; i++) {
    Expected[i] = 2 * i;
  }
#endif

  try {

    // <--- First allocation, use, and free --->

    // Allocate in pool
    void *FirstAlloc = nullptr;

    sycl::event E1 = syclexp::submit_with_event(Q, [&](sycl::handler &CGH) {
      FirstAlloc = syclexp::async_malloc(CGH, sycl::usm::alloc::device, Width);
    });

    // Use allocation in kernel
    sycl::event E2 = syclexp::submit_with_event(Q, [&](sycl::handler &CGH) {
      CGH.parallel_for<first_kernel>(
          sycl::nd_range<1>{{Width}, {Width}}, [=](sycl::nd_item<1> It) {
            size_t Dim0 = It.get_local_id(0);
            *((char *)FirstAlloc + Dim0) = static_cast<char>(Dim0);
          });
      CGH.depends_on(E1);
    });

    // Free memory back to pool
    sycl::event E3 = syclexp::submit_with_event(Q, [&](sycl::handler &CGH) {
      syclexp::async_free(CGH, FirstAlloc);
      CGH.depends_on(E2);
    });

    // <--- Second allocation, use, and free --->

    // Re-use allocation in pool
    void *SecondAlloc = nullptr;
    sycl::event E4 = syclexp::submit_with_event(Q, [&](sycl::handler &CGH) {
      CGH.depends_on(E3);
      SecondAlloc = syclexp::async_malloc(CGH, sycl::usm::alloc::device, Width);
    });

    // Re-use allocation in kernel
    sycl::event E5 = syclexp::submit_with_event(Q, [&](sycl::handler &CGH) {
      CGH.parallel_for<second_kernel>(
          sycl::nd_range<1>{{Width}, {Width}}, [=](sycl::nd_item<1> It) {
            size_t Dim0 = It.get_local_id(0);
            *((char *)SecondAlloc + Dim0) =
                *((char *)SecondAlloc + Dim0) + static_cast<char>(Dim0);
          });
      CGH.depends_on(E4);
    });

    sycl::event E6 = syclexp::submit_with_event(Q, [&](sycl::handler &CGH) {
      syclexp::memcpy(CGH, Out.data(), SecondAlloc, Width);
      CGH.depends_on(E5);
    });

    // Free memory back to pool
    sycl::event E7 = syclexp::submit_with_event(Q, [&](sycl::handler &CGH) {
      syclexp::async_free(CGH, SecondAlloc);
      CGH.depends_on(E6);
    });

    // Wait and thus release memory back to OS
    Q.wait_and_throw();

  } catch (sycl::exception &E) {
    std::cerr << "SYCL exception caught! : " << E.what() << "\n";
    return 2;
  } catch (...) {
    std::cerr << "Unknown exception caught!\n";
    return 3;
  }

  bool Validated = true;
#ifdef MEM_REUSE_CHECK
  for (int i = 0; i < Width; i++) {
    bool Mismatch = false;
    if (Out[i] != Expected[i]) {
      Mismatch = true;
      Validated = false;
    }

    if (Mismatch) {
#ifdef VERBOSE_PRINT
      std::cerr << "Result mismatch! Expected: "
                << static_cast<int>(Expected[i])
                << ", Actual: " << static_cast<int>(Out[i]) << std::endl;
#else
      break;
#endif
    }
  }
#endif

  if (Validated) {
    std::cout << "Test passed!" << std::endl;
    return 0;
  }

  std::cerr << "Test failed!" << std::endl;
  return 1;
}
