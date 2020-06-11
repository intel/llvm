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
class store_kernel;

template <typename T>
void store_test(queue q, size_t N) {
  T initial = std::numeric_limits<T>::max();
  T store = initial;
  {
    buffer<T> store_buf(&store, 1);
    q.submit([&](handler &cgh) {
      auto st = store_buf.template get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<store_kernel<T>>(range<1>(N), [=](item<1> it) {
        int gid = it.get_id(0);
        auto atm = atomic_ref<T, intel::memory_order::relaxed, intel::memory_scope::device, access::address_space::global_space>(st[0]);
        atm.store(T(gid));
      });
    });
  }

  // The initial value should have been overwritten by a work-item ID
  // Atomicity isn't tested here, but support for store() is
  assert(store != initial);
  assert(store >= T(0) && store <= T(N - 1));
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
  store_test<int>(q, N);
  store_test<unsigned int>(q, N);
  store_test<long>(q, N);
  store_test<unsigned long>(q, N);
  store_test<long long>(q, N);
  store_test<unsigned long long>(q, N);
  store_test<float>(q, N);
  store_test<double>(q, N);
  //store_test<char*>(q, N);

  std::cout << "Test passed." << std::endl;
}
