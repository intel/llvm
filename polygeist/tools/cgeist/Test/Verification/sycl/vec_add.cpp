// RUN: clang++ -fsycl -fsycl-device-only -O0 -w -emit-mlir %s -o - | FileCheck %s --check-prefix=CHECK-MLIR
// RUN: clang++ -fsycl -fsycl-device-only -O0 -w -S -emit-llvm -fsycl-targets=spir64-unknown-unknown-syclmlir %s -o - | FileCheck %s --check-prefix=CHECK-LLVM

#include <sycl/sycl.hpp>
#define N 32

// CHECK-MLIR:      gpu.func {{.*}}vec_add_simple({{.*}})
// CHECK-MLIR-SAME: kernel attributes {llvm.cconv = #llvm.cconv<spir_kernelcc>, llvm.linkage = #llvm.linkage<weak_odr>,
// CHECK-MLIR:      func.call [[FUNC:@.*vec_add_device_simple.*]]({{.*}}) :
// CHECK-MLIR-NEXT: gpu.return

// CHECK-MLIR:      func.func private [[FUNC]]({{.*}})
// CHECK-SAME:      attributes {llvm.cconv = #llvm.cconv<spir_funccc>, llvm.linkage = #llvm.linkage<internal>
// CHECK-MLIR-DAG:  [[V1:%.*]] = affine.load {{.*}}[0] : memref<?xf32, 4>
// CHECK-MLIR-DAG:  [[V2:%.*]] = affine.load {{.*}}[0] : memref<?xf32, 4>
// CHECK-MLIR-NEXT: [[RESULT:%.*]] = arith.addf [[V1]], [[V2]] : f32
// CHECK-MLIR:      affine.store [[RESULT]], {{.*}}[0] : memref<?xf32, 4>

// CHECK-LLVM:       define internal spir_func void [[FUNC:@_ZZZ21vec_add_device_simpleRSt5arrayIfLm32EES1_S1_ENKUlRN4sycl3_V17handlerEE_clES5_ENKUlNS3_2idILi1EEEE_clES8_]]({{.*}}) #[[FUNCATTRS:[0-9]+]]
// CHECK-LLVM-DAG:   [[V1:%.*]] = load float, float addrspace(4)* {{.*}}, align 4
// CHECK-LLVM-DAG:   [[V2:%.*]] = load float, float addrspace(4)* {{.*}}, align 4
// CHECK-LLVM:       [[RESULT:%.*]] = fadd float [[V1]], [[V2]]
// CHECK-LLVM:       store float [[RESULT]], float addrspace(4)* {{.*}}, align 4

// CHECK-LLVM:       define weak_odr spir_kernel void @{{.*}}vec_add_simple({{.*}}) #[[FUNCATTRS]]
// CHECK-LLVM:       call spir_func void [[FUNC]]

void vec_add_device_simple(std::array<float, N> &VA, std::array<float, N> &VB,
                           std::array<float, N> &VC) {
  auto q = sycl::queue{};
  auto range = sycl::range<1>{N};

  {
    auto bufA = sycl::buffer<float, 1>{VA.data(), range};
    auto bufB = sycl::buffer<float, 1>{VB.data(), range};
    auto bufC = sycl::buffer<float, 1>{VC.data(), range};

    q.submit([&](sycl::handler &cgh) {
      auto A = bufA.get_access<sycl::access::mode::read>(cgh);
      auto B = bufB.get_access<sycl::access::mode::read>(cgh);
      auto C = bufC.get_access<sycl::access::mode::write>(cgh);

      // kernel
      cgh.parallel_for<class vec_add_simple>(
          range, [=](sycl::id<1> id) { C[id] = A[id] + B[id]; });
    });
  }
}

void init(std::array<float, N> &h_a, std::array<float, N> &h_b,
          std::array<float, N> &h_c, std::array<float, N> &h_r) {
  for (int i = 0; i < N; i++) {
    h_a[i] = 1.0f;
    h_b[i] = 2.0f;
    h_c[i] = 0.0f;
    h_r[i] = 0.0f;
  }
}

void vec_add_host(std::array<float, N> &h_a, std::array<float, N> &h_b,
                  std::array<float, N> &h_r) {
  for (int i = 0; i < N; i++) {
    h_r[i] = h_a[i] + h_b[i];
  }
}

bool check_result(std::array<float, N> &h_c, std::array<float, N> &h_r) {
  for (int i = 0; i < N; i++) {
    if (h_r[i] != h_c[i]) {
      std::cerr << "Mismatch at element " << i << "\n";
      return false;
    }
  }
  return true;
}

int main() {
  std::array<float, N> h_a;
  std::array<float, N> h_b;
  std::array<float, N> h_c;
  std::array<float, N> h_r; // (result)

  // initialize vectors
  init(h_a, h_b, h_c, h_r);

  vec_add_host(h_a, h_b, h_r);

  vec_add_device_simple(h_a, h_b, h_c);

  if (!check_result(h_c, h_r))
    exit(1);

  std::cout << "Results are correct\n";
  return 0;
}

// Keep at the end of the file.
// CHECK-LLVM: attributes #[[FUNCATTRS]] = { convergent mustprogress noinline norecurse nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="{{.*}}/polygeist/tools/cgeist/Test/Verification/sycl/vec_add.cpp" }
