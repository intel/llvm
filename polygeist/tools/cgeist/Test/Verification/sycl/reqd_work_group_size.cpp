// RUN: clang++  -fsycl -fsycl-device-only -O0 -w -emit-mlir %s -o - | FileCheck %s --check-prefix=CHECK-MLIR
// RUN: clang++  -fsycl -fsycl-device-only -O0 -w -S -emit-llvm -fsycl-targets=spir64-unknown-unknown-syclmlir %s -o - | FileCheck %s --check-prefix=CHECK-LLVM

#include <sycl/sycl.hpp>
using namespace sycl;
static constexpr unsigned N = 8;

// CHECK-MLIR-LABEL: kernel_parallel_for_id
// CHECK-SAME:         reqd_work_group_size = [4, 2]

// CHECK-LLVM-LABEL: kernel_parallel_for_id
// CHECK-SAME:         !reqd_work_group_size [[MD:!.*]] {
// CHECK: [[MD]] = !{i32 4, i32 2}

void parallel_for_id(std::array<int, N * N> &A, queue q) {
  auto range = sycl::range<2>{N, N};

  {
    buffer a(A.data(), range);
    q.submit([&](handler &cgh) {
      auto A = a.get_access<access::mode::write>(cgh);
      cgh.parallel_for<class kernel_parallel_for_id>(range, [=](id<2> Id) [[sycl::reqd_work_group_size(2, 4)]] {
        A[Id] = Id.get(0) + Id.get(1);
      });
    });
  }
}
