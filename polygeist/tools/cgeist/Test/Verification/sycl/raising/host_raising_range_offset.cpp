// RUN: clang++ -O1 %s -S -emit-mlir -o - -fsycl -fsycl-raise-host | FileCheck %s
// RUN: clang++ -O2 %s -S -emit-mlir -o - -fsycl -fsycl-raise-host | FileCheck %s
// RUN: clang++ -O3 %s -S -emit-mlir -o - -fsycl -fsycl-raise-host | FileCheck %s

#include <sycl/sycl.hpp>

std::vector<float> init(std::size_t);

class KernelName;

// CHECK-DAG: llvm.mlir.global private unnamed_addr constant @[[KERNEL_STR:.*]]("kernel\00")

// CHECK-DAG: gpu.func @_ZTS10KernelName

// COM: Check we can detect buffers contruction

// CHECK-LABEL: llvm.func local_unnamed_addr @main
// CHECK:         sycl.host.constructor({{.*}}) {type = !sycl.buffer<[1, !llvm.void]>}  : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr) -> ()
// CHECK:         sycl.host.constructor({{.*}}) {type = !sycl.buffer<[1, !llvm.void]>}  : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr) -> ()
// CHECK:         sycl.host.constructor({{.*}}) {type = !sycl.buffer<[1, !llvm.void]>}  : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr) -> ()

// COM: Check we can detect accessors construction

// CHECK-LABEL: llvm.func internal @_ZNSt17_Function_handlerIFvRN4sycl3_V17handlerEEZ4mainEUlS3_E_E9_M_invokeERKSt9_Any_dataS3_
// CHECK-DAG:      %[[N:.*]] = llvm.mlir.constant(1024 : i64) : i64
// CHECK-DAG:      %[[OFFSET:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK:          sycl.host.constructor({{.*}}) {type = !sycl_accessor_1_21llvm2Evoid_r_gb} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CHECK:          sycl.host.constructor({{.*}}) {type = !sycl_accessor_1_21llvm2Evoid_r_gb} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CHECK:          sycl.host.constructor({{.*}}) {type = !sycl_accessor_1_21llvm2Evoid_w_gb} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()

// COM: Check we can detect nd-range assignment with range and offset arguments

// CHECK:         sycl.host.constructor(%[[G_SIZE:.*]], %[[N]]) {type = !sycl_range_1_} : (!llvm.ptr, i64) -> ()
// CHECK:         sycl.host.constructor(%[[OFF:.*]], %[[OFFSET]]) {type = !sycl_id_1_} : (!llvm.ptr, i64) -> ()
// CHECK:         sycl.host.handler.set_nd_range %[[HANDLER:.*]] -> range %[[G_SIZE]], offset %[[OFF]] : !llvm.ptr, !llvm.ptr, !llvm.ptr

// COM: Check we can detect kernel assignment to a sycl::handler and subsequently raise it to a `schedule_kernel` op

// CHECK-DAG: sycl.host.schedule_kernel %[[HANDLER]] -> @device_functions::@_ZTS10KernelName[range %[[G_SIZE]], offset %[[OFF]]]({{.*}}) : (!llvm.ptr, !llvm.ptr, {{.*}}) -> ()

int main() {
  constexpr std::size_t N = 1024;
  constexpr std::size_t Offset = 2;
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
	       cgh.parallel_for<KernelName>(sycl::range<1>(N), sycl::id<1>(Offset),
					    [=](sycl::id<1> i) { acc_c[i] = acc_a[i] + acc_b[i]; });
	     });
  }
}
