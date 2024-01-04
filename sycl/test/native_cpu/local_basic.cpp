// REQUIRES: native_cpu_be
// RUN: %clangxx -fsycl -fsycl-targets=native_cpu %s -o %t
// RUN: env ONEAPI_DEVICE_SELECTOR="native_cpu:cpu" %t
#include <sycl/sycl.hpp>

using namespace sycl;

class Test;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;
int test(queue q, const unsigned localSize) {
  const unsigned N = localSize;
  constexpr unsigned NumG = 2;
  range<1> localR{N};
  range<1> globalR{NumG * N};
  buffer<int, 1> Buffer(globalR);
  q.submit([&](handler &h) {
    auto acc = Buffer.get_access<sycl_write>(h);
    local_accessor<int, 1> local_acc1(localR, h);
    local_accessor<int, 1> local_acc2(localR, h);
    h.parallel_for<Test>(nd_range<1>{globalR, localR}, [=](nd_item<1> it) {
      auto lID = it.get_local_id(0);
      auto gID = it.get_global_id(0);
      local_acc1[lID] = gID;
      local_acc2[lID] = gID;
      acc[gID] = local_acc1[lID] + local_acc2[lID];
    });
  });
  sycl::host_accessor HostAccessor{Buffer, sycl::read_only};
  for (unsigned i = 0; i < N * NumG; i++) {
    if (HostAccessor[i] != 2 * i) {
      std::cout << "Error\n";
      return 1;
    }
  }
  return 0;
}

int main() {
  queue q;
  auto res1 = test(q, 10);
  auto res2 = test(q, 20);
  return res1 || res2;
}
