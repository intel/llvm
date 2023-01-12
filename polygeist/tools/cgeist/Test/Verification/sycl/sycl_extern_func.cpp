// RUN: clang++ -fsycl -fsycl-device-only -O0 -w -emit-mlir %s -o - | FileCheck %s --check-prefix=CHECK-MLIR
// RUN: clang++ -fsycl -fsycl-device-only -O0 -w -S -emit-llvm -fsycl-targets=spir64-unknown-unknown-syclmlir %s -o - | FileCheck %s --check-prefix=CHECK-LLVM

#include <sycl/sycl.hpp>

// CHECK-MLIR: gpu.module @device_functions

// CHECK-MLIR-LABEL: gpu.func @_ZTSZZ16host_single_taskvENKUlRN4sycl3_V17handlerEE_clES2_E18kernel_single_task
// CHECK-MLIR-SAME: kernel attributes {llvm.cconv = #llvm.cconv<spir_kernelcc>, llvm.linkage = #llvm.linkage<weak_odr>,

// CHECK-MLIR-LABEL: func.func private @_ZZZ16host_single_taskvENKUlRN4sycl3_V17handlerEE_clES2_ENKUlvE_clEv
// CHECK-MLIR: call @cons_5() : () -> ()

// CHECK-MLIR-LABEL: func.func @cons_5() attributes {llvm.cconv = #llvm.cconv<spir_funccc>, llvm.linkage = #llvm.linkage<external>
// CHECK-MLIR:           sycl.constructor(%{{.*}})
// CHECK-MLIR-NEXT:      return

// CHECK-LLVM-LABEL: define spir_func void @cons_5()
// CHECK-LLVM-SAME:  #[[FUNCATTRS1:[0-9]+]] {
// CHECK-LLVM-NEXT:  [[ACCESSOR:%.*]] = alloca %"class.sycl::_V1::accessor.1", align 8
// CHECK-LLVM-NEXT:  [[ACAST:%.*]] = addrspacecast %"class.sycl::_V1::accessor.1"* [[ACCESSOR]] to %"class.sycl::_V1::accessor.1" addrspace(4)*
// CHECK-LLVM-NEXT:  call spir_func void @_ZN4sycl3_V18accessorIiLi1ELNS0_6access4modeE1025ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEC1Ev(%"class.sycl::_V1::accessor.1" addrspace(4)* [[ACAST]])

// CHECK-LLVM-LABEL: define weak_odr spir_kernel void @_ZTSZZ16host_single_taskvENKUlRN4sycl3_V17handlerEE_clES2_E18kernel_single_task
// CHECK-LLVM-SAME: #[[FUNCATTRS1]]
// CHECK: call void @cons_5()

extern "C" SYCL_EXTERNAL void cons_5() {
  sycl::accessor<sycl::cl_int, 1, sycl::access::mode::write> accessor;
}

void host_single_task() {
  auto q = sycl::queue{};
  auto range = sycl::range<1>{1};
  auto buf = sycl::buffer<int, 1>{nullptr, range};
  q.submit([&](sycl::handler &cgh) {
    auto A = buf.get_access<sycl::access::mode::write>(cgh);
    cgh.single_task<class kernel_single_task>([=]() {
      cons_5();
      A[0] = 42;
    });
  });
}

// Keep at the end of the file.
// CHECK-LLVM: attributes #[[FUNCATTRS1]] = { convergent mustprogress noinline norecurse nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="{{.*}}/polygeist/tools/cgeist/Test/Verification/sycl/sycl_extern_func.cpp" }
