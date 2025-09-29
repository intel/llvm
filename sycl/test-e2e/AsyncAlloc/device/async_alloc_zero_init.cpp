// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/properties/queue_properties.hpp>

#include <sycl/ext/oneapi/experimental/async_alloc/async_alloc.hpp>
#include <sycl/ext/oneapi/experimental/async_alloc/memory_pool.hpp>

// Uncomment to print additional test information
// #define VERBOSE_PRINT

namespace syclexp = sycl::ext::oneapi::experimental;
class first_kernel;

int main() {

  sycl::device Dev;
  sycl::property_list QProp{sycl::property::queue::in_order()};
  sycl::queue Q(QProp);
  sycl::context Ctx = Q.get_context();
  size_t Width = 8;
  std::vector<char> Out(Width);
  std::vector<char> Expected(Width);
  // Expect all zeroes
  for (int i = 0; i < Width; i++) {
    Expected[i] = 0;
  }

  try {

    // Create pool with zero_init property
    syclexp::properties PoolProps{syclexp::zero_init{}};
    syclexp::memory_pool MemPool(Ctx, Dev, sycl::usm::alloc::device, PoolProps);

    // <--- First allocation, use, and free --->

    // Allocate in pool
    void *FirstAlloc = syclexp::async_malloc_from_pool(Q, Width, MemPool);

    Q.memcpy(Out.data(), FirstAlloc, Width);

    // Free memory back to pool
    syclexp::async_free(Q, FirstAlloc);

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

  if (Validated) {
    std::cout << "Test passed!" << std::endl;
    return 0;
  }

  std::cerr << "Test failed!" << std::endl;
  return 1;
}
