// REQUIRES: native_cpu
// RUN: %clangxx -fsycl -fsycl-targets=native_cpu %s -o %t
// RUN: env ONEAPI_DEVICE_SELECTOR=native_cpu:cpu %t
#include <sycl/sycl.hpp>

#include <array>
#include <iostream>

// Check that we correctly emit the kernel declaration in the native CPU
// helper header when not all kernel args are unused
struct myfun {
  int *ptr1;
  int *ptr2;
  int used1;
  int used2;
  char *notused;
  int used3;

  myfun(int *ptr1, int *ptr2, int used1, int used2, char *notused, int used3)
      : ptr1(ptr1), ptr2(ptr2), used1(used1), used2(used2), notused(notused),
        used3(used3) {}
  void operator()(sycl::id<1> id) const {
    ptr1[id] = ptr2[id] + used1 + used2 + used3;
  }
};

int main() {
  const size_t N = 4;
  std::array<int, N> A = {{0, 0, 0, 0}}, B = {{1, 2, 3, 4}};
  sycl::queue deviceQueue;
  sycl::range<1> numOfItems{N};
  auto a_ptr = sycl::malloc_device<int>(N, deviceQueue);
  auto b_ptr = sycl::malloc_device<int>(N, deviceQueue);
  deviceQueue.memcpy(a_ptr, A.data(), N * sizeof(int)).wait();
  deviceQueue.memcpy(b_ptr, B.data(), N * sizeof(int)).wait();
  int used1 = 1;
  int used2 = 2;
  int used3 = 3;
  char *unused = nullptr;
  myfun TheFun(a_ptr, b_ptr, used1, used2, unused, used3);

  deviceQueue
      .submit([&](sycl::handler &cgh) { cgh.parallel_for(numOfItems, TheFun); })
      .wait();
  deviceQueue.memcpy(A.data(), a_ptr, N * sizeof(int)).wait();

  for (unsigned int i = 0; i < N; i++) {
    if (A[i] != B[i] + used1 + used2 + used3) {
      std::cout << "The results are incorrect (element " << i << " is " << A[i]
                << "!\n";
      return 1;
    }
  }
  sycl::free(a_ptr, deviceQueue);
  sycl::free(b_ptr, deviceQueue);
  std::cout << "The results are correct!\n";
  return 0;
}
