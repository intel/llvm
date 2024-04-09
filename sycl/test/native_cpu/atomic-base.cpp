// Simple test that checks that we can run a simple applications that uses
// builtins
// REQUIRES: native_cpu
// RUN: %clangxx -fsycl -fsycl-targets=native_cpu %s -o %t
// RUN: env ONEAPI_DEVICE_SELECTOR="native_cpu:cpu" %t
#include <sycl/sycl.hpp>
#include <vector>

using namespace sycl;

int add_pre_inc_test(queue q, size_t N) {
  constexpr auto scope = memory_scope::device;
  constexpr auto space = access::address_space::global_space;
  int sum = 0;
  {
    buffer<int> sum_buf(&sum, 1);

    q.submit([&](handler &cgh) {
      auto sum = sum_buf.template get_access<access::mode::read_write>(cgh);
      cgh.parallel_for(range<1>(N), [=](item<1> it) {
        int gid = it.get_id(0);
        auto atm =
            sycl::atomic_ref<int, memory_order::relaxed, scope, space>(sum[0]);
        ++atm;
      });
    });
  }

  // All work-items increment by 1, so final value should be equal to N
  return sum;
}

int main() {
  const int N = 10;
  sycl::queue q;
  int res = add_pre_inc_test(q, N);
  if (res != N) {
    std::cout << "Error, result is " << res << " but should be " << N << "\n";
    return 1;
  }
  return 0;
}
