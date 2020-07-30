// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>
#include <algorithm>
#include <cassert>
#include <numeric>
#include <vector>
using namespace sycl;
using namespace sycl::intel;

template <typename T>
class compare_exchange_kernel;

template <typename T>
void compare_exchange_test(queue q, size_t N) {
  const T initial = T(N);
  T compare_exchange = initial;
  std::vector<T> output(N);
  std::fill(output.begin(), output.end(), T(0));
  {
    buffer<T> compare_exchange_buf(&compare_exchange, 1);
    buffer<T> output_buf(output.data(), output.size());

    q.submit([&](handler &cgh) {
      auto exc = compare_exchange_buf.template get_access<access::mode::read_write>(cgh);
      auto out = output_buf.template get_access<access::mode::discard_write>(cgh);
      cgh.parallel_for<compare_exchange_kernel<T>>(range<1>(N), [=](item<1>
                                                                        it) {
        size_t gid = it.get_id(0);
        auto atm = atomic_ref<T, intel::memory_order::relaxed, intel::memory_scope::device, access::address_space::global_space>(exc[0]);
        T result = T(N); // Avoid copying pointer
        bool success = atm.compare_exchange_strong(result, (T)gid);
        if (success) {
          out[gid] = result;
        } else {
          out[gid] = T(gid);
        }
      });
    });
  }

  // Only one work-item should have received the initial sentinel value
  assert(std::count(output.begin(), output.end(), initial) == 1);

  // All other values should be the index itself or the sentinel value
  for (size_t i = 0; i < N; ++i) {
    assert(output[i] == T(i) || output[i] == initial);
  }
}

int main() {
  queue q;
  std::string version = q.get_device().get_info<info::device::version>();
  if (version < std::string("2.0")) {
    std::cout << "Skipping test\n";
    return 0;
  }

  constexpr int N = 32;
  compare_exchange_test<int>(q, N);
  compare_exchange_test<unsigned int>(q, N);
  compare_exchange_test<long>(q, N);
  compare_exchange_test<unsigned long>(q, N);
  compare_exchange_test<long long>(q, N);
  compare_exchange_test<unsigned long long>(q, N);
  compare_exchange_test<float>(q, N);
  compare_exchange_test<double>(q, N);
  compare_exchange_test<char *>(q, N);

  std::cout << "Test passed." << std::endl;
}
