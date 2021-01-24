// RUN: %clangxx -fsycl -fsycl-unnamed-lambda -DSYCL_USE_NATIVE_FP_ATOMICS \
// RUN: -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// TODO: Remove items from UNSUPPORTED once corresponding backends support
// "native" implementation
// UNSUPPORTED: gpu, cpu, cuda

#include <CL/sycl.hpp>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>
#include <vector>
using namespace sycl;
using namespace sycl::ONEAPI;

template <typename T> void max_test(queue q, size_t N) {
  T initial = std::numeric_limits<T>::lowest();
  T val = initial;
  std::vector<T> output(N);
  std::fill(output.begin(), output.end(), std::numeric_limits<T>::max());
  {
    buffer<T> val_buf(&val, 1);
    buffer<T> output_buf(output.data(), output.size());

    q.submit([&](handler &cgh) {
      auto val = val_buf.template get_access<access::mode::read_write>(cgh);
      auto out =
          output_buf.template get_access<access::mode::discard_write>(cgh);
      cgh.parallel_for(range<1>(N), [=](item<1> it) {
        int gid = it.get_id(0);
        auto atm = atomic_ref<T, ONEAPI::memory_order::relaxed,
                              ONEAPI::memory_scope::device,
                              access::address_space::global_space>(val[0]);

        // +1 accounts for lowest() returning 0 for unsigned types
        out[gid] = atm.fetch_max(T(gid) + 1);
      });
    });
  }

  // Final value should be equal to N
  assert(val == N);

  // Only one work-item should have received the initial value
  assert(std::count(output.begin(), output.end(), initial) == 1);

  // fetch_max returns original value
  // Intermediate values should all be >= initial value
  for (int i = 0; i < N; ++i) {
    assert(output[i] >= initial);
  }
}

int main() {
  queue q;
  std::string version = q.get_device().get_info<info::device::version>();

  constexpr int N = 32;
  max_test<int>(q, N);
  max_test<unsigned int>(q, N);
  max_test<long>(q, N);
  max_test<unsigned long>(q, N);
  max_test<long long>(q, N);
  max_test<unsigned long long>(q, N);
  max_test<float>(q, N);
  max_test<double>(q, N);

  std::cout << "Test passed." << std::endl;
}
