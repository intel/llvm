// RUN: %clangxx -fsycl -fsycl-unnamed-lambda -DSYCL_USE_NATIVE_FP_ATOMICS \
// RUN:  -fsycl-device-only -S %s -o - | FileCheck %s --check-prefix=CHECK-LLVM
// RUN: %clangxx -fsycl -fsycl-unnamed-lambda -fsycl-device-only -S %s -o - \
// RUN: | FileCheck %s --check-prefix=CHECK-LLVM-EMU
// RUN: %clangxx -fsycl -fsycl-unnamed-lambda -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %RUN_ON_HOST %t.out

#include <CL/sycl.hpp>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>
#include <vector>
using namespace sycl;
using namespace sycl::ONEAPI;

template <typename T>
void max_test(queue q, size_t N) {
  T initial = std::numeric_limits<T>::lowest();
  T val = initial;
  std::vector<T> output(N);
  std::fill(output.begin(), output.end(), std::numeric_limits<T>::max());
  {
    buffer<T> val_buf(&val, 1);
    buffer<T> output_buf(output.data(), output.size());

    q.submit([&](handler &cgh) {
      auto val = val_buf.template get_access<access::mode::read_write>(cgh);
      auto out = output_buf.template get_access<access::mode::discard_write>(cgh);
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
  if (version < std::string("2.0")) {
    std::cout << "Skipping test\n";
    return 0;
  }

  constexpr int N = 32;
  // CHECK-LLVM: declare dso_local spir_func i32
  // CHECK-LLVM-SAME: @_Z{{[0-9]+}}__spirv_AtomicSMax
  // CHECK-LLVM-SAME: (i32 addrspace(1)*, i32, i32, i32)
  max_test<int>(q, N);
  // CHECK-LLVM: declare dso_local spir_func i32
  // CHECK-LLVM-SAME: @_Z{{[0-9]+}}__spirv_AtomicUMax
  // CHECK-LLVM-SAME: (i32 addrspace(1)*, i32, i32, i32)
  max_test<unsigned int>(q, N);
  // CHECK-LLVM: declare dso_local spir_func i[[long:(32)|(64)]]
  // CHECK-LLVM-SAME: @_Z{{[0-9]+}}__spirv_AtomicSMax
  // CHECK-LLVM-SAME: (i[[long]] addrspace(1)*, i32, i32, i[[long]])
  max_test<long>(q, N);
  // CHECK-LLVM: declare dso_local spir_func i[[long]]
  // CHECK-LLVM-SAME: @_Z{{[0-9]+}}__spirv_AtomicUMax
  // CHECK-LLVM-SAME: (i[[long]] addrspace(1)*, i32, i32, i[[long]])
  max_test<unsigned long>(q, N);
  // CHECK-LLVM: declare dso_local spir_func i64
  // CHECK-LLVM-SAME: @_Z{{[0-9]+}}__spirv_AtomicSMax
  // CHECK-LLVM-SAME: (i64 addrspace(1)*, i32, i32, i64)
  max_test<long long>(q, N);
  // CHECK-LLVM: declare dso_local spir_func i64
  // CHECK-LLVM-SAME: @_Z{{[0-9]+}}__spirv_AtomicUMax
  // CHECK-LLVM-SAME: (i64 addrspace(1)*, i32, i32, i64)
  max_test<unsigned long long>(q, N);
  // CHECK-LLVM: declare dso_local spir_func float
  // CHECK-LLVM-SAME: @_Z{{[0-9]+}}__spirv_AtomicFMaxEXT
  // CHECK-LLVM-SAME: (float addrspace(1)*, i32, i32, float)
  // CHECK-LLVM-EMU: declare {{.*}} i32 @{{.*}}__spirv_AtomicLoad
  // CHECK-LLVM-EMU-SAME: (i32 addrspace(1)*, i32, i32)
  // CHECK-LLVM-EMU: declare {{.*}} i32 @{{.*}}__spirv_AtomicCompareExchange
  // CHECK-LLVM-EMU-SAME: (i32 addrspace(1)*, i32, i32, i32, i32, i32)
  max_test<float>(q, N);
  // CHECK-LLVM: declare dso_local spir_func double
  // CHECK-LLVM-SAME: @_Z{{[0-9]+}}__spirv_AtomicFMaxEXT
  // CHECK-LLVM-SAME: (double addrspace(1)*, i32, i32, double)
  // CHECK-LLVM-EMU: declare {{.*}} i64 @{{.*}}__spirv_AtomicLoad
  // CHECK-LLVM-EMU-SAME: (i64 addrspace(1)*, i32, i32)
  // CHECK-LLVM-EMU: declare {{.*}} i64 @{{.*}}__spirv_AtomicCompareExchange
  // CHECK-LLVM-EMU-SAME: (i64 addrspace(1)*, i32, i32, i32, i64, i64)
  max_test<double>(q, N);

  std::cout << "Test passed." << std::endl;
}
