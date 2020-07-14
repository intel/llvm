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
class load_kernel;

template <typename T>
void load_test(queue q, size_t N) {
  T initial = 42;
  T load = initial;
  std::vector<T> output(N);
  std::fill(output.begin(), output.end(), 0);
  {
    buffer<T> load_buf(&load, 1);
    buffer<T> output_buf(output.data(), output.size());

    q.submit([&](handler &cgh) {
      auto ld = load_buf.template get_access<access::mode::read_write>(cgh);
      auto out = output_buf.template get_access<access::mode::discard_write>(cgh);
      cgh.parallel_for<load_kernel<T>>(range<1>(N), [=](item<1> it) {
        int gid = it.get_id(0);
        auto atm = atomic_ref<T, intel::memory_order::relaxed, intel::memory_scope::device, access::address_space::global_space>(ld[0]);
        out[gid] = atm.load();
      });
    });
  }

  // All work-items should read the same value
  // Atomicity isn't tested here, but support for load() is
  assert(std::all_of(output.begin(), output.end(), [&](T x) { return (x == initial); }));
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
  load_test<int>(q, N);
  load_test<unsigned int>(q, N);
  load_test<long>(q, N);
  load_test<unsigned long>(q, N);
  load_test<long long>(q, N);
  load_test<unsigned long long>(q, N);
  load_test<float>(q, N);
  load_test<double>(q, N);
  //load_test<char*>(q, N);

  std::cout << "Test passed." << std::endl;
}
