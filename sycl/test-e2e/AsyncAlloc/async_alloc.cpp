// REQUIRES: aspect-ext_oneapi_async_memory_alloc

// RUN: %{build} -o %t.out
// RUN: %{run-unfiltered-devices} %t.out

#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/properties/queue_properties.hpp>

#include <sycl/ext/oneapi/async_alloc/async_alloc.hpp>

// Uncomment to print additional test information
#define VERBOSE_PRINT
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
  return 1;
#endif // defined(SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC)

  sycl::device dev;
  sycl::property_list q_prop{sycl::property::queue::in_order()};
  sycl::queue q(q_prop);
  sycl::context ctx = q.get_context();
  size_t width = 8;

  try {
    // Check aspect
    bool asyncSupport = dev.has(sycl::aspect::ext_oneapi_async_memory_alloc);

    if (!asyncSupport) {
      std::cout << "ext_oneapi_async_memory_alloc aspect not supported when it "
                   "should be! \n";
      return 2;
    }

#ifdef VERBOSE_PRINT
    std::cout << "async memory alloc support: " << asyncSupport << "\n";
#endif

    // Create pool
    syclexp::memory_pool memPool(ctx, dev, sycl::usm::alloc::device);
    std::vector<char> out(width);

    // <--- First allocation, use, and free --->

    // Allocate in pool
    void *firstAlloc = syclexp::async_malloc_from_pool(q, width, memPool);

    // Use allocation in kernel
    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<first_kernel>(
          sycl::nd_range<1>{{width}, {width}}, [=](sycl::nd_item<1> it) {
            size_t dim0 = it.get_local_id(0);
            *((char *)firstAlloc + dim0) = (char)dim0;
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
                *((char *)secondAlloc + dim0) + (char)dim0;
          });
    });

    q.memcpy(out.data(), secondAlloc, width);

    // Free memory back to pool
    syclexp::async_free(q, secondAlloc);

    // Wait and thus release memory back to OS
    q.wait_and_throw();

    for (int i = 0; i < width; i++) {
      std::cout << (int)out[i] << ",";
    }
    std::cout << std::endl;

  } catch (sycl::exception e) {
    std::cerr << "SYCL exception caught! : " << e.what() << "\n";
    return 3;
  } catch (...) {
    std::cerr << "Unknown exception caught!\n";
    return 4;
  }

  std::cout << "Test passed!" << std::endl;
  return 0;
}
