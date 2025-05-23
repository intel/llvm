// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

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

// Check feature test macro
#if defined(SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC)
  assert(SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC == 1);
#if defined(VERBOSE_PRINT)
  std::cout << "SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC is defined!" << std::endl;
#endif
#else
  std::cerr
      << "SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC feature test macro is not defined!"
      << std::endl;
  return 4;
#endif // defined(SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC)

  sycl::device Dev;
  sycl::property_list QProp{sycl::property::queue::in_order()};
  sycl::queue Q(QProp);
  sycl::context Ctx = Q.get_context();
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
    // Check aspect
    bool AsyncSupport = Dev.has(sycl::aspect::ext_oneapi_async_memory_alloc);

    if (!AsyncSupport) {
      std::cerr << "ext_oneapi_async_memory_alloc aspect not supported when it "
                   "should be! \n";
      return 5;
    }

#ifdef VERBOSE_PRINT
    std::cout << "async memory alloc support: " << AsyncSupport << "\n";
#endif

    // <--- First allocation, use, and free --->

    // Allocate in pool
    void *FirstAlloc =
        syclexp::async_malloc(Q, sycl::usm::alloc::device, Width);

    // Use allocation in kernel
    Q.submit([&](sycl::handler &CGH) {
      CGH.parallel_for<first_kernel>(
          sycl::nd_range<1>{{Width}, {Width}}, [=](sycl::nd_item<1> It) {
            size_t Dim0 = It.get_local_id(0);
            *((char *)FirstAlloc + Dim0) = static_cast<char>(Dim0);
          });
    });

    // Free memory back to pool
    syclexp::async_free(Q, FirstAlloc);

    // <--- Second allocation, use, and free --->

    // Re-use allocation in pool
    void *SecondAlloc =
        syclexp::async_malloc(Q, sycl::usm::alloc::device, Width);

    // Re-use allocation in kernel
    Q.submit([&](sycl::handler &CGH) {
      CGH.parallel_for<second_kernel>(
          sycl::nd_range<1>{{Width}, {Width}}, [=](sycl::nd_item<1> It) {
            size_t Dim0 = It.get_local_id(0);
            *((char *)SecondAlloc + Dim0) =
                *((char *)SecondAlloc + Dim0) + static_cast<char>(Dim0);
          });
    });

    Q.memcpy(Out.data(), FirstAlloc, Width);

    // Free memory back to pool
    syclexp::async_free(Q, SecondAlloc);

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
