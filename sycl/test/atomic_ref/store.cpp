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
class store_kernel;

template <typename T>
void store_test(queue q, size_t N) {
  T initial = T(N);
  T store = initial;
  {
    buffer<T> store_buf(&store, 1);
    q.submit([&](handler &cgh) {
      auto st = store_buf.template get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<store_kernel<T>>(range<1>(N), [=](item<1> it) {
        size_t gid = it.get_id(0);
        auto atm = atomic_ref<T, ONEAPI::memory_order::relaxed,
                              ONEAPI::memory_scope::device,
                              access::address_space::global_space>(st[0]);
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
  // CHECK-LLVM: declare dso_local spir_func void
  // CHECK-LLVM-SAME: @_Z{{[0-9]+}}__spirv_AtomicStore
  // CHECK-LLVM-SAME: (i32 addrspace(1)*, i32, i32, i32)
  store_test<int>(q, N);
  // CHECK-LLVM: declare dso_local spir_func void
  // CHECK-LLVM-SAME: @_Z{{[0-9]+}}__spirv_AtomicStore
  // CHECK-LLVM-SAME: (i32 addrspace(1)*, i32, i32, i32)
  store_test<unsigned int>(q, N);
  // CHECK-LLVM: declare dso_local spir_func void
  // CHECK-LLVM-SAME: @_Z{{[0-9]+}}__spirv_AtomicStore{{.*}}(i[[long:(32)|(64)]]
  // CHECK-LLVM-SAME:  addrspace(1)*, i32, i32, i[[long]])
  store_test<long>(q, N);
  // CHECK-LLVM: declare dso_local spir_func void
  // CHECK-LLVM-SAME: @_Z{{[0-9]+}}__spirv_AtomicStore
  // CHECK-LLVM-SAME: (i[[long]] addrspace(1)*, i32, i32, i[[long]])
  store_test<unsigned long>(q, N);
  // CHECK-LLVM: declare dso_local spir_func void
  // CHECK-LLVM-SAME: @_Z{{[0-9]+}}__spirv_AtomicStore
  // CHECK-LLVM-SAME: (i64 addrspace(1)*, i32, i32, i64)
  store_test<long long>(q, N);
  // CHECK-LLVM: declare dso_local spir_func void
  // CHECK-LLVM-SAME: @_Z{{[0-9]+}}__spirv_AtomicStore
  // CHECK-LLVM-SAME: (i64 addrspace(1)*, i32, i32, i64)
  store_test<unsigned long long>(q, N);
  // The remaining functions use the already-declared ones on the IR level
  store_test<float>(q, N);
  store_test<double>(q, N);
  store_test<char *>(q, N);

  std::cout << "Test passed." << std::endl;
}
