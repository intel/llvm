// RUN: clang++ -fsycl -fsycl-device-only -O0 -w -emit-mlir %s -o - | FileCheck %s --check-prefix=CHECK-MLIR
// RUN: clang++ -fsycl -fsycl-device-only -O0 -w -S -emit-llvm -fsycl-targets=spir64-unknown-unknown-syclmlir %s -o - | FileCheck %s --check-prefix=CHECK-LLVM

#include <sycl/sycl.hpp>

// CHECK-MLIR-DAG: !sycl_array_1_ = !sycl.array<[1], (memref<1xi64, 4>)>
// CHECK-MLIR-DAG: !sycl_id_1_ = !sycl.id<[1], (!sycl_array_1_)>
// CHECK-MLIR-DAG: !sycl_range_1_ = !sycl.range<[1], (!sycl_array_1_)>

// CHECK-LLVM-DAG: %"class.sycl::_V1::accessor.1" = type { %"class.sycl::_V1::detail::AccessorImplDevice.1", { i32 addrspace(1)* } }
// CHECK-LLVM-DAG: %"class.sycl::_V1::detail::AccessorImplDevice.1" = type { %"class.sycl::_V1::id.1", %"class.sycl::_V1::range.1", %"class.sycl::_V1::range.1" }
// CHECK-LLVM-DAG: %"class.sycl::_V1::id.1" = type { %"class.sycl::_V1::detail::array.1" }
// CHECK-LLVM-DAG: %"class.sycl::_V1::detail::array.1" = type { [1 x i64] }
// CHECK-LLVM-DAG: %"class.sycl::_V1::range.1" = type { %"class.sycl::_V1::detail::array.1" }
// CHECK-LLVM-DAG: %"class.sycl::_V1::item.1.true" = type { %"struct.sycl::_V1::detail::ItemBase.1.true" }
// CHECK-LLVM-DAG: %"struct.sycl::_V1::detail::ItemBase.1.true" = type { %"class.sycl::_V1::range.1", %"class.sycl::_V1::id.1", %"class.sycl::_V1::id.1" }

// CHECK-MLIR: gpu.module @device_functions
// CHECK-MLIR-LABEL: gpu.func @_ZTS8kernel_1(
// CHECK-MLIR:         %arg0: memref<?xi32, 1> {llvm.noundef},
// CHECK-MLIR-SAME:    %arg1: memref<?x!sycl_range_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_range_1_, llvm.noundef},
// CHECK-MLIR-SAME:    %arg2: memref<?x!sycl_range_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_range_1_, llvm.noundef},
// CHECK-MLIR-SAME:    %arg3: memref<?x!sycl_id_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_id_1_, llvm.noundef})
// CHECK-MLIR-SAME:  kernel attributes {[[CCONV:llvm.cconv = #llvm.cconv<spir_kernelcc>]], [[LINKAGE:llvm.linkage = #llvm.linkage<weak_odr>]],
// CHECK-MLIR-NOT: gpu.func kernel

// CHECK-LLVM: define weak_odr spir_kernel void @_ZTS8kernel_1(i32 addrspace(1)* {{.*}}, [[RANGE_TY:%"class.sycl::_V1::range.1"]]* noundef byval([[RANGE_TY]]) align 8 {{.*}}, [[RANGE_TY]]* noundef byval([[RANGE_TY]]) align 8 {{.*}}, 
// CHECK-LLVM-SAME:  [[ID_TY:%"class.sycl::_V1::id.1"]]* noundef byval([[ID_TY]]) align 8 {{.*}}) #[[FUNCATTRS:[0-9]+]]
class kernel_1 {
 sycl::accessor<sycl::cl_int, 1, sycl::access::mode::read_write> A;

public:
	kernel_1(sycl::accessor<sycl::cl_int, 1, sycl::access::mode::read_write> A) : A(A) {}

  void operator()(sycl::id<1> id) const {
    A[id] = 42;
  }
};

void host_1() {
  auto q = sycl::queue{};
  auto range = sycl::range<1>{1};

  {
    auto buf = sycl::buffer<int, 1>{nullptr, range};
    q.submit([&](sycl::handler &cgh) {
      auto A = buf.get_access<sycl::access::mode::read_write>(cgh);
      auto ker =  kernel_1{A};
      cgh.parallel_for<kernel_1>(range, ker);
    });
  }
}

// CHECK-MLIR: gpu.func @_ZTSZZ6host_2vENKUlRN4sycl3_V17handlerEE_clES2_E8kernel_2(
// CHECK-MLIR:          %arg0: memref<?xi32, 1> {llvm.noundef},
// CHECK-MLIR-SAME:     %arg1: memref<?x!sycl_range_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_range_1_, llvm.noundef},
// CHECK-MLIR-SAME:     %arg2: memref<?x!sycl_range_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_range_1_, llvm.noundef},
// CHECK-MLIR-SAME:     %arg3: memref<?x!sycl_id_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_id_1_, llvm.noundef}
// CHECK-MLIR-SAME:     kernel attributes {[[CCONV]], [[LINKAGE]], {{.*}}}
// CHECK-MLIR-NOT: gpu.func kernel

// CHECK-LLVM: define weak_odr spir_kernel void @_ZTSZZ6host_2vENKUlRN4sycl3_V17handlerEE_clES2_E8kernel_2({{.*}}) #[[FUNCATTRS]]

void host_2() {
  auto q = sycl::queue{};
  auto range = sycl::range<1>{1};

  {
    auto buf = sycl::buffer<int, 1>{nullptr, range};
    q.submit([&](sycl::handler &cgh) {
      auto A = buf.get_access<sycl::access::mode::read_write>(cgh);
      cgh.parallel_for<class kernel_2>(range, [=](sycl::id<1> id) {
        A[id] = 42;
      });
    });
  }
}

// CHECK-MLIR-NOT: SYCLKernel =
SYCL_EXTERNAL void function_1(sycl::item<2, true> item) {
  auto id = item.get_id(0);
}

// Keep at the end of the file.
// CHECK-LLVM: attributes #[[FUNCATTRS]] = { convergent mustprogress noinline norecurse nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="{{.*}}/polygeist/tools/cgeist/Test/Verification/sycl/kernels.cpp" }
