// RUN: %clangxx -fsycl -fsycl-unnamed-lambda -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>
#include <sycl/sycl.hpp>
#include <vector>
using namespace sycl;
using namespace sycl::ext::oneapi;

// Equivalent to add_test from add.cpp
// Uses atomic_accessor instead of atomic_ref
template <typename T> void accessor_test(queue q, size_t N) {
  T sum = 0;
  std::vector<T> output(N, 0);
  {
    buffer<T> sum_buf(&sum, 1);
    buffer<T> output_buf(output.data(), output.size());

    q.submit([&](handler &cgh) {
#if __cplusplus > 201402L
      static_assert(
          std::is_same<decltype(atomic_accessor(sum_buf, cgh, relaxed_order,
                                                device_scope)),
                       atomic_accessor<T, 1, memory_order::relaxed,
                                       memory_scope::device>>::value,
          "atomic_accessor type incorrectly deduced");
#endif
      auto sum =
          atomic_accessor<T, 1, memory_order::relaxed, memory_scope::device>(
              sum_buf, cgh);
      auto out =
          output_buf.template get_access<access::mode::discard_write>(cgh);
      cgh.parallel_for(range<1>(N), [=](item<1> it) {
        int gid = it.get_id(0);
        static_assert(
            std::is_same<decltype(sum[0]),
                         ::sycl::ext::oneapi::atomic_ref<
                             T, memory_order::relaxed, memory_scope::device,
                             access::address_space::global_space>>::value,
            "atomic_accessor returns incorrect atomic_ref");
        out[gid] = sum[0].fetch_add(T(1));
      });
    });
  }

  // All work-items increment by 1, so final value should be equal to N
  assert(sum == N);

  // Intermediate values should be unique
  std::sort(output.begin(), output.end());
  assert(std::unique(output.begin(), output.end()) == output.end());

  // Fetch returns original value: will be in [0, N-1]
  auto min_e = output[0];
  auto max_e = output[output.size() - 1];
  assert(min_e == 0 && max_e == N - 1);
}

// Simplified form of accessor_test for local memory
template <typename T>
void local_accessor_test(queue q, size_t N, size_t L = 8) {
  assert(N % L == 0);
  std::vector<T> output(N / L, 0);
  {
    buffer<T> output_buf(output.data(), output.size());
    q.submit([&](handler &cgh) {
      auto sum =
          atomic_accessor<T, 1, memory_order::relaxed, memory_scope::device,
                          access::target::local>(1, cgh);
      auto out = output_buf.template get_access<access::mode::read_write>(cgh);
      cgh.parallel_for(nd_range<1>(N, L), [=](nd_item<1> it) {
        int grp = it.get_group(0);
        sum[0].store(0);
        it.barrier();
        static_assert(
            std::is_same<decltype(sum[0]),
                         ::sycl::ext::oneapi::atomic_ref<
                             T, memory_order::relaxed, memory_scope::device,
                             access::address_space::local_space>>::value,
            "local atomic_accessor returns incorrect atomic_ref");
        T result = sum[0].fetch_add(T(1));
        if (result == it.get_local_range(0) - 1) {
          out[grp] = result;
        }
      });
    });
  }

  // All work-items increment by 1, and last in the group writes out old value
  // All values should be L-1
  assert(std::all_of(output.begin(), output.end(),
                     [=](T x) { return x == L - 1; }));
}

int main() {
  queue q;
  constexpr int N = 32;
  accessor_test<int>(q, N);
  local_accessor_test<int>(q, N);
  std::cout << "Test passed." << std::endl;
}
