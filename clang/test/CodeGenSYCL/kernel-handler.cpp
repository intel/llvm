// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple nvptx64-unknown-unknown -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefixes=ALL,NONATIVESUPPORT
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefixes=ALL,NATIVESUPPORT

// This test checks IR generated when kernel_handler argument
// (used to handle SYCL 2020 specialization constants) is passed
// by kernel

#include "sycl.hpp"

using namespace cl::sycl;

void test(int val) {
  queue q;
  q.submit([&](handler &h) {
    int a;
    kernel_handler kh;
    h.single_task<class test_kernel_handler>(
        [=](auto) {
          int local = a;
        },
        kh);
  });
}

// ALL: define dso_local{{ spir_kernel | }}void @{{.*}}test_kernel_handler{{[^(]*}}
// ALL-SAME: (i32 noundef %_arg_, i8 addrspace(1)* noundef align 1 %_arg__specialization_constants_buffer)
// ALL: %kh = alloca %"class.cl::sycl::kernel_handler", align 1

// NONATIVESUPPORT: %[[KH:[0-9]+]] = load i8 addrspace(1)*, i8 addrspace(1)** %_arg__specialization_constants_buffer.addr, align 8
// NONATIVESUPPORT: %[[ADDRSPACECAST:[0-9]+]] = addrspacecast i8 addrspace(1)* %[[KH]] to i8*
// NONATIVESUPPORT: call void @{{.*}}__init_specialization_constants_buffer{{.*}}(%"class.cl::sycl::kernel_handler"* noundef nonnull align 1 dereferenceable(1) %kh, i8* noundef %[[ADDRSPACECAST]])

// NATIVESUPPORT-NOT: load i8 addrspace(1)*, i8 addrspace(1)** %_arg__specialization_constants_buffer.addr, align 8
// NATIVESUPPORT-NOT: addrspacecast i8 addrspace(1)* %{{[0-9]+}} to i8*
// NATIVESUPPORT-NOT: call void @{{.*}}__init_specialization_constants_buffer{{.*}}(%"class.cl::sycl::kernel_handler"* noundef align 4 nonnull align 1 dereferenceable(1) %kh, i8* noundef align 4 %{{[0-9]+}})

// ALL: call{{ spir_func | }}void @{{[a-zA-Z0-9_$]+}}kernel_handler{{[a-zA-Z0-9_$]+}}
// ALL-SAME: noundef byval(%"class.cl::sycl::kernel_handler")
