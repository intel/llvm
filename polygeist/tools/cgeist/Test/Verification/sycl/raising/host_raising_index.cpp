// RUN: clang++ -O1 %s -S -emit-mlir -o - -fsycl -fsycl-raise-host -Xclang -opaque-pointers | FileCheck %s
// RUN: clang++ -O2 %s -S -emit-mlir -o - -fsycl -fsycl-raise-host -Xclang -opaque-pointers | FileCheck %s
// RUN: clang++ -O3 %s -S -emit-mlir -o - -fsycl -fsycl-raise-host -Xclang -opaque-pointers | FileCheck %s

#include <sycl/sycl.hpp>

std::vector<float> init(std::size_t);

class KernelName;

// CHECK-DAG: llvm.mlir.global private unnamed_addr constant @[[RANGE_STR:.*]]("range\00")
// CHECK-DAG: llvm.mlir.global private unnamed_addr constant @[[KERNEL_STR:.*]]("kernel\00")

// CHECK-DAG: gpu.func @_ZTS10KernelName
// CHECK-DAG: gpu.func @_ZTSN4sycl3_V16detail19__pf_kernel_wrapperI10KernelNameEE

// COM: Check we can detect buffers contruction

// CHECK-LABEL: llvm.func @main
// CHECK:         sycl.host.constructor({{.*}}) {type = !sycl.buffer<[1, !llvm.void]>}  : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr) -> ()
// CHECK:         sycl.host.constructor({{.*}}) {type = !sycl.buffer<[1, !llvm.void]>}  : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr) -> ()
// CHECK:         sycl.host.constructor({{.*}}) {type = !sycl.buffer<[1, !llvm.void]>}  : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr) -> ()

// COM: Check we can detect accessors construction

// CHECK-LABEL: llvm.func internal @_ZNSt17_Function_handlerIFvRN4sycl3_V17handlerEEZ4mainEUlS3_E_E9_M_invokeERKSt9_Any_dataS3_
// CHECK-DAG:     %[[N:.*]] = arith.constant 1024 : index
// CHECK:          sycl.host.constructor({{.*}}) {type = !sycl_accessor_1_21llvm2Evoid_r_gb} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CHECK:          sycl.host.constructor({{.*}}) {type = !sycl_accessor_1_21llvm2Evoid_r_gb} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CHECK:          sycl.host.constructor({{.*}}) {type = !sycl_accessor_1_21llvm2Evoid_w_gb} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()

// COM: Check we can detect nd-range assignment with a scalar argument

// CHECK:         %[[G_SIZE:.*]] = sycl.range.constructor(%[[N]]) : (index) -> memref<1x!sycl_range_1_>
// CHECK:         sycl.host.handler.set_nd_range %[[HANDLER:.*]] -> %[[G_SIZE]] : !llvm.ptr, memref<1x!sycl_range_1_>

// COM: Check we can detect kernel assignment to a sycl::handler:

// CHECK-DAG: sycl.host.handler.set_kernel %[[HANDLER]] -> @device_functions::@_ZTSN4sycl3_V16detail19__pf_kernel_wrapperI10KernelNameEE : !llvm.ptr
// CHECK-DAG: sycl.host.handler.set_kernel %[[HANDLER]] -> @device_functions::@_ZTS10KernelName : !llvm.ptr

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
