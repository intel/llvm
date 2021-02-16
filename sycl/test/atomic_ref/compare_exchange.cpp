// RUN: %clangxx -fsycl -fsycl-unnamed-lambda -fsycl-device-only -S %s -o - \
// RUN: | FileCheck %s --check-prefix=CHECK-LLVM
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %RUN_ON_HOST %t.out

#include <CL/sycl.hpp>
#include <algorithm>
#include <cassert>
#include <numeric>
#include <vector>
using namespace sycl;
using namespace sycl::ONEAPI;

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
        auto atm = atomic_ref<T, ONEAPI::memory_order::relaxed,
                              ONEAPI::memory_scope::device,
                              access::address_space::global_space>(exc[0]);
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
  // CHECK-LLVM: declare dso_local spir_func i32
  // CHECK-LLVM-SAME: @_Z{{[0-9]+}}__spirv_AtomicCompareExchange
  // CHECK-LLVM-SAME: (i32 addrspace(1)*, i32, i32, i32, i32, i32)
  compare_exchange_test<int>(q, N);
  // CHECK-LLVM: declare dso_local spir_func i32
  // CHECK-LLVM-SAME: @_Z{{[0-9]+}}__spirv_AtomicCompareExchange
  // CHECK-LLVM-SAME: (i32 addrspace(1)*, i32, i32, i32, i32, i32)
  compare_exchange_test<unsigned int>(q, N);
  // CHECK-LLVM: declare dso_local spir_func i[[long:(32)|(64)]]
  // CHECK-LLVM-SAME: @_Z{{[0-9]+}}__spirv_AtomicCompareExchange
  // CHECK-LLVM-SAME: (i[[long]] addrspace(1)*, i32, i32, i32,
  // CHECK-LLVM-SAME: i[[long]], i[[long]])
  compare_exchange_test<long>(q, N);
  // CHECK-LLVM: declare dso_local spir_func i[[long]]
  // CHECK-LLVM-SAME: @_Z{{[0-9]+}}__spirv_AtomicCompareExchange
  // CHECK-LLVM-SAME: (i[[long]] addrspace(1)*, i32, i32, i32,
  // CHECK-LLVM-SAME: i[[long]], i[[long]])
  compare_exchange_test<unsigned long>(q, N);
  // CHECK-LLVM: declare dso_local spir_func i64
  // CHECK-LLVM-SAME: @_Z{{[0-9]+}}__spirv_AtomicCompareExchange
  // CHECK-LLVM-SAME: (i64 addrspace(1)*, i32, i32, i32, i64, i64)
  compare_exchange_test<long long>(q, N);
  // CHECK-LLVM: declare dso_local spir_func i64
  // CHECK-LLVM-SAME: @_Z{{[0-9]+}}__spirv_AtomicCompareExchange
  // CHECK-LLVM-SAME: (i64 addrspace(1)*, i32, i32, i32, i64, i64)
  compare_exchange_test<unsigned long long>(q, N);
  // The remaining functions use the already-declared ones on the IR level
  compare_exchange_test<float>(q, N);
  compare_exchange_test<double>(q, N);
  compare_exchange_test<char *>(q, N);

  std::cout << "Test passed." << std::endl;
}
