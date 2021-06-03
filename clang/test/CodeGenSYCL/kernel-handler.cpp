// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple nvptx64-unknown-unknown -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefix=NONATIVESUPPORT
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefix=NATIVESUPPORT

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

// NONATIVESUPPORT: define dso_local void @{{.*}}test_kernel_handler{{[^(]*}}
// NONATIVESUPPORT-SAME: (i32 %_arg_, i8 addrspace(1)* %_arg__specialization_constants_buffer)
// NONATIVESUPPORT: %kh = alloca %"class.[[MANGLEDCLASS:[a-zA-Z0-9_]+]].cl::sycl::kernel_handler", align 1
// NONATIVESUPPORT: %[[KH:[0-9]+]] = load i8 addrspace(1)*, i8 addrspace(1)** %_arg__specialization_constants_buffer.addr, align 8
// NONATIVESUPPORT: %[[ADDRSPACECAST:[0-9]+]] = addrspacecast i8 addrspace(1)* %[[KH]] to i8*
// NONATIVESUPPORT: call void @{{.*}}__init_specialization_constants_buffer{{.*}}(%"class.[[MANGLEDCLASS]].cl::sycl::kernel_handler"* nonnull align 1 dereferenceable(1) %kh, i8* %[[ADDRSPACECAST]])
// NONATIVESUPPORT: void @[[MANGLEDKERNELCALL:[a-zA-Z0-9_$]+]]
// NONATIVESUPPORT-SAME: byval(%"class.[[MANGLEDCLASS]].cl::sycl::kernel_handler")

// NATIVESUPPORT: define dso_local spir_kernel void @{{.*}}test_kernel_handler{{[^(]*}}
// NATIVESUPPORT-SAME: (i32 %_arg_)
// NATIVESUPPORT: %kh = alloca %"class.[[MANGLEDCLASS:[a-zA-Z0-9_]+]].cl::sycl::kernel_handler"
// NATIVESUPPORT-NOT: __init_specialization_constants_buffer
// NATIVE-SUPPORT: call spir_func void @"[[MANGLEDKERNELCALL:[a-zA-Z0-9_$]+]]"
// NATIVE-SUPPORT-SAME: byval(%"class.[[MANGLEDCLASS]].cl::sycl::kernel_handler")
