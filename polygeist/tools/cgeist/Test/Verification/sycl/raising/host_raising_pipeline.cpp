// RUN: clang++ -O2 %s -S -emit-mlir -o - -fsycl -fsycl-raise-host -Xcgeist -no-early-drop-host-code -Xcgeist -print-pipeline 2>&1 | FileCheck %s

// CHECK-LABEL: Canonicalization pipeline:
// CHECK:       Pass Manager with 2 passes:
// CHECK:       any(
// CHECK-SAME:    gpu.module(any({{.*}})),
// CHECK-SAME:    gpu.module(any({{.*}})))

// CHECK-LABEL: Optimization pipeline:
// CHECK:       Pass Manager with 8 passes:
// CHECK:       any(
// CHECK-SAME:    sycl-raise-host{ }
// CHECK-SAME:    sycl-constant-propagation{relaxed-aliasing=false}
// CHECK-SAME:    arg-promotion
// CHECK-SAME:    kernel-disjoint-specialization{relaxed-aliasing=false}
// CHECK-SAME:    gpu.module(any({{.*}})),loop-internalization{relaxed-aliasing=false shared-memory-size=32000 unroll-factor=4}
// CHECK-SAME:    gpu.module(any({{.*}}))
// CHECK-SAME:    inliner{max-num-iters=3 mode=simple remove-dead-callees=true})

#include <sycl/sycl.hpp>

std::vector<float> init(std::size_t);

class KernelName;

int main() {
  constexpr std::size_t N = 1024;
  const std::vector<float> a = init(N);
  std::vector<float> b(N);
  short foo = 123;
  sycl::float4 vec1{1, 2, 3, 4};
  sycl::half8 vec2{1, 2, 3, 4, 5, 6, 7, 8};
  sycl::queue q;
  {
    sycl::buffer<float> buff_a(a);
    sycl::buffer<float> buff_b(b);
    q.submit([&](sycl::handler &cgh) {
      long bar = 456;
      sycl::accessor acc_a(buff_a, cgh, sycl::read_only);
      auto acc_b = buff_b.get_access<sycl::access::mode::write>(cgh);
      cgh.parallel_for<KernelName>(N, [=](sycl::id<1> i) {
        acc_b[i] = acc_a[i] + foo * bar + vec1[1] - vec2[2];
      });
    });
  }
}
