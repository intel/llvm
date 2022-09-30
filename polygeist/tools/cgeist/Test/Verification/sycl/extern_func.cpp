#include <sycl/sycl.hpp>

// RUN: sycl-clang.py %s -S | FileCheck %s --check-prefix=CHECK-MLIR
// RUN: sycl-clang.py %s -S -emit-llvm | FileCheck %s --check-prefix=CHECK-LLVM

// CHECK-MLIR-LABEL: func.func @cons_5() attributes {llvm.cconv = #llvm.cconv<spir_funccc>, llvm.linkage = #llvm.linkage<external>, passthrough = ["norecurse", "nounwind", "convergent", "mustprogress"]} {
// CHECK-MLIR:           sycl.constructor(%{{.*}})
// CHECK-MLIR-NEXT:      return

// CHECK-LLVM-LABEL: define spir_func void @cons_5() #0 {
// CHECK-LLVM-NEXT:  %{{.*}} = alloca %"class.cl::sycl::accessor.1", i64 ptrtoint (%"class.cl::sycl::accessor.1"* getelementptr (%"class.cl::sycl::accessor.1", %"class.cl::sycl::accessor.1"* null, i32 1) to i64), align 8
// CHECK-LLVM-NEXT:  call void @_ZN4sycl3_V18accessorIiLi1ELNS0_6access4modeE1025ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEC1Ev(%"class.cl::sycl::accessor.1"* %1, %"class.cl::sycl::accessor.1"* %1, i64 0, i64 1, i64 1)
// CHECK-LLVM-NEXT:  ret void


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
