// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>
#include <algorithm>
#include <cassert>
#include <numeric>
#include <vector>
using namespace sycl;
using namespace sycl::intel;

template <typename T>
class exchange_kernel;

template <typename T>
void exchange_test(queue q, size_t N) {
  const T initial = std::numeric_limits<T>::max();
  T exchange = initial;
  std::vector<T> output(N);
  std::fill(output.begin(), output.end(), 0);
  {
    buffer<T> exchange_buf(&exchange, 1);
    buffer<T> output_buf(output.data(), output.size());

    q.submit([&](handler &cgh) {
      auto exc = exchange_buf.template get_access<access::mode::read_write>(cgh);
      auto out = output_buf.template get_access<access::mode::discard_write>(cgh);
      cgh.parallel_for<exchange_kernel<T>>(range<1>(N), [=](item<1> it) {
        int gid = it.get_id(0);
        auto atm = atomic_ref<T, intel::memory_order::relaxed, intel::memory_scope::device, access::address_space::global_space>(exc[0]);
        out[gid] = atm.exchange(gid);
      });
    });
  }

  // Only one work-item should have received the initial sentinel value
  assert(std::count(output.begin(), output.end(), initial) == 1);

  // All other values should be unique; each work-item replaces the value it reads with its own ID
  std::sort(output.begin(), output.end());
  assert(std::unique(output.begin(), output.end()) == output.end());
}

int main() {
  queue q;
  std::string version = q.get_device().get_info<info::device::version>();
  if (version < std::string("2.0")) {
    std::cout << "Skipping test\n";
    return 0;
  }

  constexpr int N = 32;

  // TODO: Enable missing tests when supported
  exchange_test<int>(q, N);
  exchange_test<unsigned int>(q, N);
  exchange_test<long>(q, N);
  exchange_test<unsigned long>(q, N);
  exchange_test<long long>(q, N);
  exchange_test<unsigned long long>(q, N);
  exchange_test<float>(q, N);
  exchange_test<double>(q, N);
  //exchange_test<char*>(q, N);

  std::cout << "Test passed." << std::endl;
}
