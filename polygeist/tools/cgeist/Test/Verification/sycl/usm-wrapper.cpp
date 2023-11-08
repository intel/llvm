// RUN: clang++ -fsycl -fsycl-device-only -S -emit-llvm -o - -fsycl-targets=spir64-unknown-unknown-syclmlir -O3 %s | FileCheck %s

// COM: Test `foo` is called with the right parameters. In order to do so, we need to add call-site attributes.

#include <sycl/sycl.hpp>

struct wrapper { std::size_t a; std::size_t b; float *c; };

SYCL_EXTERNAL void foo(sycl::id<1>, wrapper);

// CHECK-LABEL: define spir_func void @_Z4testN4sycl3_V12idILi1EEE7wrapper(
// CHECK-SAME:                                                             ptr nocapture noundef readonly byval(%"class.sycl::_V1::id.1") align 8 %[[#ARG0:]],
// CHECK-SAME:                                                             ptr nocapture noundef readonly byval({ i64, i64, ptr addrspace(4) }) align 8 %[[#ARG1:]])
// CHECK:         %[[#WRAPPER_ALLOCA:]] = alloca { i64, i64, ptr addrspace(4) }, align 8
// CHECK:         %[[#ID_ALLOCA:]] = alloca %"class.sycl::_V1::id.1", align 8
// CHECK:         %[[GS:.*]] = load i64, ptr %[[#ARG0]], align 1
// CHECK:         %[[M0:.*]] = load i64, ptr %[[#ARG1]], align 1
// CHECK:         %[[M1_PTR:.*]] = getelementptr inbounds i8, ptr %[[#ARG1]], i64 8
// CHECK:         %[[M1:.*]] = load i64, ptr %[[M1_PTR]], align 1
// CHECK:         %[[M2_PTR:.*]] = getelementptr inbounds i8, ptr %[[#ARG1]], i64 16
// CHECK:         %[[M2:.*]] = load ptr addrspace(4), ptr %[[M2_PTR]], align 1
// CHECK:         store i64 %[[GS]], ptr %[[#ID_ALLOCA]], align 8
// CHECK:         store i64 %[[M0]], ptr %[[#WRAPPER_ALLOCA]], align 8
// CHECK:         %[[CPY_M1_PTR:.*]] = getelementptr inbounds { i64, i64, ptr addrspace(4) }, ptr %[[#WRAPPER_ALLOCA]], i64 0, i32 1
// CHECK:         store i64 %[[M1]], ptr %[[CPY_M1_PTR]], align 8
// CHECK:         %[[CPY_M2_PTR:.*]] = getelementptr inbounds { i64, i64, ptr addrspace(4) }, ptr %[[#WRAPPER_ALLOCA]], i64 0, i32 2
// CHECK:         store ptr addrspace(4) %[[M2]], ptr %[[CPY_M2_PTR]], align 8
// CHECK:         tail call spir_func void @_Z3fooN4sycl3_V12idILi1EEE7wrapper(ptr noundef nonnull byval(%"class.sycl::_V1::id.1") align 8 %[[#ID_ALLOCA]], ptr noundef nonnull byval({ i64, i64, ptr addrspace(4) }) align 8 %[[#WRAPPER_ALLOCA]])
// CHECK:         ret void

// CHECK: declare spir_func void @_Z3fooN4sycl3_V12idILi1EEE7wrapper(ptr noundef byval(%"class.sycl::_V1::id.1") align 8, ptr noundef byval({ i64, i64, ptr addrspace(4) }) align 8)

SYCL_EXTERNAL void test(sycl::id<1> i, wrapper w) {
  foo(i, w);
}
