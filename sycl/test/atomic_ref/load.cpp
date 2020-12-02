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
class load_kernel;

template <typename T>
void load_test(queue q, size_t N) {
  T initial = T(42);
  T load = initial;
  std::vector<T> output(N);
  std::fill(output.begin(), output.end(), T(0));
  {
    buffer<T> load_buf(&load, 1);
    buffer<T> output_buf(output.data(), output.size());

    q.submit([&](handler &cgh) {
      auto ld = load_buf.template get_access<access::mode::read_write>(cgh);
      auto out = output_buf.template get_access<access::mode::discard_write>(cgh);
      cgh.parallel_for<load_kernel<T>>(range<1>(N), [=](item<1> it) {
        size_t gid = it.get_id(0);
        auto atm = atomic_ref<T, ONEAPI::memory_order::relaxed,
                              ONEAPI::memory_scope::device,
                              access::address_space::global_space>(ld[0]);
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
  // CHECK-LLVM: declare dso_local spir_func i32
  // CHECK-LLVM-SAME: @_Z{{[0-9]+}}__spirv_AtomicLoad
  // CHECK-LLVM-SAME: (i32 addrspace(1)*, i32, i32)
  load_test<int>(q, N);
  // CHECK-LLVM: declare dso_local spir_func i32
  // CHECK-LLVM-SAME: @_Z{{[0-9]+}}__spirv_AtomicLoad
  // CHECK-LLVM-SAME: (i32 addrspace(1)*, i32, i32)
  load_test<unsigned int>(q, N);
  // CHECK-LLVM: declare dso_local spir_func i[[long:(32)|(64)]]
  // CHECK-LLVM-SAME: @_Z{{[0-9]+}}__spirv_AtomicLoad
  // CHECK-LLVM-SAME: (i[[long]] addrspace(1)*, i32, i32)
  load_test<long>(q, N);
  // CHECK-LLVM: declare dso_local spir_func i[[long]]
  // CHECK-LLVM-SAME: @_Z{{[0-9]+}}__spirv_AtomicLoad
  // CHECK-LLVM-SAME: (i[[long]] addrspace(1)*, i32, i32)
  load_test<unsigned long>(q, N);
  // CHECK-LLVM: declare dso_local spir_func i64
  // CHECK-LLVM-SAME: @_Z{{[0-9]+}}__spirv_AtomicLoad
  // CHECK-LLVM-SAME: (i64 addrspace(1)*, i32, i32)
  load_test<long long>(q, N);
  // CHECK-LLVM: declare dso_local spir_func i64
  // CHECK-LLVM-SAME: @_Z{{[0-9]+}}__spirv_AtomicLoad
  // CHECK-LLVM-SAME: (i64 addrspace(1)*, i32, i32)
  load_test<unsigned long long>(q, N);
  // The remaining functions use the already-declared ones on the IR level
  load_test<float>(q, N);
  load_test<double>(q, N);
  load_test<char *>(q, N);

  std::cout << "Test passed." << std::endl;
}
