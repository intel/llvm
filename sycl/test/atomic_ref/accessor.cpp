// RUN: %clangxx -fsycl -fsycl-unnamed-lambda -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>
#include <vector>
using namespace sycl;
using namespace sycl::intel;

// Equivalent to add_test from add.cpp
// Uses atomic_accessor instead of atomic_ref
template <typename T>
void accessor_test(queue q, size_t N) {
  T sum = 0;
  std::vector<T> output(N);
  std::fill(output.begin(), output.end(), 0);
  {
    buffer<T> sum_buf(&sum, 1);
    buffer<T> output_buf(output.data(), output.size());

    q.submit([&](handler &cgh) {
#if __cplusplus > 201402L
      auto sum = atomic_accessor(sum_buf, cgh, relaxed_order, device_scope);
      static_assert(std::is_same<decltype(sum), atomic_accessor<T, 1, intel::memory_order::relaxed, intel::memory_scope::device>>::value, "atomic_accessor type incorrectly deduced");
#else
      auto sum = atomic_accessor<T, 1, intel::memory_order::relaxed, intel::memory_scope::device>(sum_buf, cgh);
#endif
      auto out = output_buf.template get_access<access::mode::discard_write>(cgh);
      cgh.parallel_for(range<1>(N), [=](item<1> it) {
        int gid = it.get_id(0);
        static_assert(std::is_same<decltype(sum[0]), atomic_ref<T, intel::memory_order::relaxed, intel::memory_scope::device, access::address_space::global_space>>::value, "atomic_accessor returns incorrect atomic_ref");
        out[gid] = sum[0].fetch_add(T(1));
      });
    });
  }

  // All work-items increment by 1, so final value should be equal to N
  assert(sum == N);

  // Fetch returns original value: will be in [0, N-1]
  auto min_e = std::min_element(output.begin(), output.end());
  auto max_e = std::max_element(output.begin(), output.end());
  assert(*min_e == 0 && *max_e == N - 1);

  // Intermediate values should be unique
  std::sort(output.begin(), output.end());
  assert(std::unique(output.begin(), output.end()) == output.end());
}

int main() {
  queue q;
  constexpr int N = 32;
  accessor_test<int>(q, N);
  std::cout << "Test passed." << std::endl;
}
