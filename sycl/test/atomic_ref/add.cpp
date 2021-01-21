// TODO: Once NVPTX accepts the __spirv_AtomicF*() IR, remove the XFAIL mark
// XFAIL: cuda

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

template <typename T, typename Difference = T>
void add_fetch_test(queue q, size_t N) {
  T sum = 0;
  std::vector<T> output(N);
  std::fill(output.begin(), output.end(), T(0));
  {
    buffer<T> sum_buf(&sum, 1);
    buffer<T> output_buf(output.data(), output.size());

    q.submit([&](handler &cgh) {
      auto sum = sum_buf.template get_access<access::mode::read_write>(cgh);
      auto out =
          output_buf.template get_access<access::mode::discard_write>(cgh);
      cgh.parallel_for(range<1>(N), [=](item<1> it) {
        int gid = it.get_id(0);
        auto atm = atomic_ref<T, ONEAPI::memory_order::relaxed,
                              ONEAPI::memory_scope::device,
                              access::address_space::global_space>(sum[0]);
        out[gid] = atm.fetch_add(Difference(1));
      });
    });
  }

  // All work-items increment by 1, so final value should be equal to N
  assert(sum == T(N));

  // Fetch returns original value: will be in [0, N-1]
  auto min_e = std::min_element(output.begin(), output.end());
  auto max_e = std::max_element(output.begin(), output.end());
  assert(*min_e == T(0) && *max_e == T(N - 1));

  // Intermediate values should be unique
  std::sort(output.begin(), output.end());
  assert(std::unique(output.begin(), output.end()) == output.end());
}

template <typename T, typename Difference = T>
void add_plus_equal_test(queue q, size_t N) {
  T sum = 0;
  std::vector<T> output(N);
  std::fill(output.begin(), output.end(), T(0));
  {
    buffer<T> sum_buf(&sum, 1);
    buffer<T> output_buf(output.data(), output.size());

    q.submit([&](handler &cgh) {
      auto sum = sum_buf.template get_access<access::mode::read_write>(cgh);
      auto out =
          output_buf.template get_access<access::mode::discard_write>(cgh);
      cgh.parallel_for(range<1>(N), [=](item<1> it) {
        int gid = it.get_id(0);
        auto atm = atomic_ref<T, ONEAPI::memory_order::relaxed,
                              ONEAPI::memory_scope::device,
                              access::address_space::global_space>(sum[0]);
        out[gid] = atm += Difference(1);
      });
    });
  }

  // All work-items increment by 1, so final value should be equal to N
  assert(sum == T(N));

  // += returns updated value: will be in [1, N]
  auto min_e = std::min_element(output.begin(), output.end());
  auto max_e = std::max_element(output.begin(), output.end());
  assert(*min_e == T(1) && *max_e == T(N));

  // Intermediate values should be unique
  std::sort(output.begin(), output.end());
  assert(std::unique(output.begin(), output.end()) == output.end());
}

template <typename T, typename Difference = T>
void add_pre_inc_test(queue q, size_t N) {
  T sum = 0;
  std::vector<T> output(N);
  std::fill(output.begin(), output.end(), T(0));
  {
    buffer<T> sum_buf(&sum, 1);
    buffer<T> output_buf(output.data(), output.size());

    q.submit([&](handler &cgh) {
      auto sum = sum_buf.template get_access<access::mode::read_write>(cgh);
      auto out =
          output_buf.template get_access<access::mode::discard_write>(cgh);
      cgh.parallel_for(range<1>(N), [=](item<1> it) {
        int gid = it.get_id(0);
        auto atm = atomic_ref<T, ONEAPI::memory_order::relaxed,
                              ONEAPI::memory_scope::device,
                              access::address_space::global_space>(sum[0]);
        out[gid] = ++atm;
      });
    });
  }

  // All work-items increment by 1, so final value should be equal to N
  assert(sum == T(N));

  // Pre-increment returns updated value: will be in [1, N]
  auto min_e = std::min_element(output.begin(), output.end());
  auto max_e = std::max_element(output.begin(), output.end());
  assert(*min_e == T(1) && *max_e == T(N));

  // Intermediate values should be unique
  std::sort(output.begin(), output.end());
  assert(std::unique(output.begin(), output.end()) == output.end());
}

template <typename T, typename Difference = T>
void add_post_inc_test(queue q, size_t N) {
  T sum = 0;
  std::vector<T> output(N);
  std::fill(output.begin(), output.end(), T(0));
  {
    buffer<T> sum_buf(&sum, 1);
    buffer<T> output_buf(output.data(), output.size());

    q.submit([&](handler &cgh) {
      auto sum = sum_buf.template get_access<access::mode::read_write>(cgh);
      auto out =
          output_buf.template get_access<access::mode::discard_write>(cgh);
      cgh.parallel_for(range<1>(N), [=](item<1> it) {
        int gid = it.get_id(0);
        auto atm = atomic_ref<T, ONEAPI::memory_order::relaxed,
                              ONEAPI::memory_scope::device,
                              access::address_space::global_space>(sum[0]);
        out[gid] = atm++;
      });
    });
  }

  // All work-items increment by 1, so final value should be equal to N
  assert(sum == T(N));

  // Post-increment returns original value: will be in [0, N-1]
  auto min_e = std::min_element(output.begin(), output.end());
  auto max_e = std::max_element(output.begin(), output.end());
  assert(*min_e == T(0) && *max_e == T(N - 1));

  // Intermediate values should be unique
  std::sort(output.begin(), output.end());
  assert(std::unique(output.begin(), output.end()) == output.end());
}

template <typename T, typename Difference = T>
void add_test(queue q, size_t N) {
  add_fetch_test<T, Difference>(q, N);
  add_plus_equal_test<T, Difference>(q, N);
  add_pre_inc_test<T, Difference>(q, N);
  add_post_inc_test<T, Difference>(q, N);
}

// Floating-point types do not support pre- or post-increment
template <> void add_test<float>(queue q, size_t N) {
  add_fetch_test<float>(q, N);
  // CHECK-LLVM: declare dso_local spir_func float
  // CHECK-LLVM-SAME: @_Z{{[0-9]+}}__spirv_AtomicFAddEXT
  // CHECK-LLVM-SAME: (float addrspace(1)*, i32, i32, float)
  // CHECK-LLVM-EMU: declare {{.*}} i32 @{{.*}}__spirv_AtomicLoad
  // CHECK-LLVM-EMU-SAME: (i32 addrspace(1)*, i32, i32)
  // CHECK-LLVM-EMU: declare {{.*}} i32 @{{.*}}__spirv_AtomicCompareExchange
  // CHECK-LLVM-EMU-SAME: (i32 addrspace(1)*, i32, i32, i32, i32, i32)
  add_plus_equal_test<float>(q, N);
}
template <> void add_test<double>(queue q, size_t N) {
  add_fetch_test<double>(q, N);
  // CHECK-LLVM: declare dso_local spir_func double
  // CHECK-LLVM-SAME: @_Z{{[0-9]+}}__spirv_AtomicFAddEXT
  // CHECK-LLVM-SAME: double addrspace(1)*, i32, i32, double)
  // CHECK-LLVM-EMU: declare {{.*}} i64 @{{.*}}__spirv_AtomicLoad
  // CHECK-LLVM-EMU-SAME: (i64 addrspace(1)*, i32, i32)
  // CHECK-LLVM-EMU: declare {{.*}} i64 @{{.*}}__spirv_AtomicCompareExchange
  // CHECK-LLVM-EMU-SAME: (i64 addrspace(1)*, i32, i32, i32, i64, i64)
  add_plus_equal_test<double>(q, N);
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
  // CHECK-LLVM-SAME: @_Z{{[0-9]+}}__spirv_AtomicIAdd
  // CHECK-LLVM-SAME: (i32 addrspace(1)*, i32, i32, i32)
  add_test<int>(q, N);
  // CHECK-LLVM: declare dso_local spir_func i32
  // CHECK-LLVM-SAME: @_Z{{[0-9]+}}__spirv_AtomicIAdd
  // CHECK-LLVM-SAME: (i32 addrspace(1)*, i32, i32, i32)
  add_test<unsigned int>(q, N);
  // CHECK-LLVM: declare dso_local spir_func i[[long:(32)|(64)]]
  // CHECK-LLVM-SAME: @_Z{{[0-9]+}}__spirv_AtomicIAdd
  // CHECK-LLVM-SAME: (i[[long]] addrspace(1)*, i32, i32, i[[long]])
  add_test<long>(q, N);
  // CHECK-LLVM: declare dso_local spir_func i[[long]]
  // CHECK-LLVM-SAME: @_Z{{[0-9]+}}__spirv_AtomicIAdd
  // CHECK-LLVM-SAME: (i[[long]] addrspace(1)*, i32, i32, i[[long]])
  add_test<unsigned long>(q, N);
  // CHECK-LLVM: declare dso_local spir_func i64
  // CHECK-LLVM-SAME: @_Z{{[0-9]+}}__spirv_AtomicIAdd
  // CHECK-LLVM-SAME: (i64 addrspace(1)*, i32, i32, i64)
  add_test<long long>(q, N);
  // CHECK-LLVM: declare dso_local spir_func i64
  // CHECK-LLVM-SAME: @_Z{{[0-9]+}}__spirv_AtomicIAdd
  // CHECK-LLVM-SAME: (i64 addrspace(1)*, i32, i32, i64)
  add_test<unsigned long long>(q, N);
  // Floating point-typed functions have been instantiated earlier
  add_test<float>(q, N);
  add_test<double>(q, N);
  // CHECK-LLVM: declare dso_local spir_func i64
  // CHECK-LLVM-SAME: @_Z{{[0-9]+}}__spirv_AtomicLoad
  // CHECK-LLVM-SAME: i64 addrspace(1)*, i32, i32)
  // CHECK-LLVM: declare dso_local spir_func i64
  // CHECK-LLVM-SAME: @_Z{{[0-9]+}}__spirv_AtomicCompareExchange
  // CHECK-LLVM-SAME: i64 addrspace(1)*, i32, i32, i32, i64, i64)
  add_test<char *, ptrdiff_t>(q, N);

  std::cout << "Test passed." << std::endl;
}
