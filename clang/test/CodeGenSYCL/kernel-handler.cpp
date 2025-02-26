// RUN: %clang_cc1 -fno-sycl-force-inline-kernel-lambda -fno-sycl-decompose-functor -fsycl-is-device -internal-isystem %S/Inputs -triple nvptx64-unknown-unknown -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefixes=ALL,NONATIVESUPPORT
// RUN: %clang_cc1 -fno-sycl-force-inline-kernel-lambda -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefixes=ALL,NATIVESUPPORT

// This test checks IR generated when kernel_handler argument
// (used to handle SYCL 2020 specialization constants) is passed
// by kernel

#include "sycl.hpp"

using namespace sycl;

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
// NONATIVESUPPORT-SAME: (ptr noundef byval(%class.anon) align 4 %_arg__sycl_functor, ptr addrspace(1) noundef align 1 %_arg__specialization_constants_buffer)
// NATIVESUPPORT-SAME: (i32 noundef %_arg_a, ptr addrspace(1) noundef align 1 %_arg__specialization_constants_buffer)
// ALL: %kh = alloca %"class.sycl::_V1::kernel_handler", align 1

// NONATIVESUPPORT: %[[KH:[0-9]+]] = load ptr addrspace(1), ptr %_arg__specialization_constants_buffer.addr, align 8
// NONATIVESUPPORT: %[[ADDRSPACECAST:[0-9]+]] = addrspacecast ptr addrspace(1) %[[KH]] to ptr
// NONATIVESUPPORT: call void @{{.*}}__init_specialization_constants_buffer{{.*}}(ptr noundef nonnull align 1 dereferenceable(1) %kh, ptr noundef %[[ADDRSPACECAST]])

// NATIVESUPPORT-NOT: load ptr addrspace(1), ptr addrspace(1) %_arg__specialization_constants_buffer.addr, align 8
// NATIVESUPPORT-NOT: addrspacecast ptr addrspace(1) %{{[0-9]+}} to ptr
// NATIVESUPPORT-NOT: call void @{{.*}}__init_specialization_constants_buffer{{.*}}(ptr noundef align 4 nonnull align 1 dereferenceable(1) %kh, ptr noundef align 4 %{{[0-9]+}})

// ALL: call{{ spir_func | }}void @{{[a-zA-Z0-9_$]+}}kernel_handler{{[a-zA-Z0-9_$]+}}
// ALL-SAME: noundef byval(%"class.sycl::_V1::kernel_handler")
