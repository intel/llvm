// RUN: clang++ -O1 %s -S -emit-mlir -o - -fsycl -fsycl-raise-host -Xclang -opaque-pointers | FileCheck %s
// RUN: clang++ -O2 %s -S -emit-mlir -o - -fsycl -fsycl-raise-host -Xclang -opaque-pointers | FileCheck %s
// RUN: clang++ -O3 %s -S -emit-mlir -o - -fsycl -fsycl-raise-host -Xclang -opaque-pointers | FileCheck %s

#include <sycl/sycl.hpp>

std::vector<float> init(std::size_t);

class KernelName;

// COM: Check we can detect kernel assignment to a sycl::handler:

// CHECK-DAG: gpu.func @_ZTS10KernelName
// CHECK-DAG: gpu.func @_ZTSN4sycl3_V16detail19__pf_kernel_wrapperI10KernelNameEE

// CHECK-DAG: sycl.host.handler.set_kernel %{{.*}} -> @device_functions::@_ZTSN4sycl3_V16detail19__pf_kernel_wrapperI10KernelNameEE : !llvm.ptr
// CHECK-DAG: sycl.host.handler.set_kernel %{{.*}} -> @device_functions::@_ZTS10KernelName : !llvm.ptr

int main() {
  constexpr std::size_t N = 1024;
  const std::vector<float> a = init(N);
  const std::vector<float> b = init(N);
  std::vector<float> c(N);
  sycl::queue q;
  {
    sycl::buffer<float> buff_a(a);
    sycl::buffer<float> buff_b(b);
    sycl::buffer<float> buff_c(c);
    q.submit([&](sycl::handler &cgh) {
	       sycl::accessor acc_a(buff_a, cgh, sycl::read_only);
	       sycl::accessor acc_b(buff_b, cgh, sycl::read_only);
	       sycl::accessor acc_c(buff_c, cgh, sycl::write_only, sycl::no_init);
	       cgh.parallel_for<KernelName>(N, [=](sycl::id<1> i) { acc_c[i] = acc_a[i] + acc_b[i]; });
	     });
  }
}

