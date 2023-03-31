// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out

// This test checks if users will successfully allocate 160, 0, and -16 bytes of
// shared memory, and also test user can call free() without worrying about
// nullptr or invalid memory descriptor returned from malloc.

#include <iostream>
#include <stdlib.h>
#include <sycl/sycl.hpp>
using namespace sycl;

int main(int argc, char *argv[]) {
  auto exception_handler = [](sycl::exception_list exceptions) {
    for (std::exception_ptr const &e : exceptions) {
      try {
        std::rethrow_exception(e);
      } catch (sycl::exception const &e) {
        std::cout << "Caught asynchronous SYCL "
                     "exception:\n"
                  << e.what() << std::endl;
      }
    }
  };

  queue myQueue(default_selector_v, exception_handler);
  std::cout << "Device: " << myQueue.get_device().get_info<info::device::name>()
            << std::endl;

  double *ia = (double *)malloc_shared(160, myQueue);
  double *ja = (double *)malloc_shared(0, myQueue);
  double *result = (double *)malloc_shared(-16, myQueue);

  assert(ia != nullptr);
  assert(ja == nullptr);
  assert(result == nullptr);

  std::cout << "ia : " << ia << " ja: " << ja << " result : " << result
            << std::endl;

  // followings should not throw CL_INVALID_VALUE
  sycl::free(ia, myQueue);
  sycl::free(nullptr, myQueue);
  sycl::free(ja, myQueue);
  sycl::free(result, myQueue);

  return 0;
}
