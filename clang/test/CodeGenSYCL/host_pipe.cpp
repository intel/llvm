// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown-sycldevice -fsycl-unique-prefix=THE_PREFIX -opaque-pointers -emit-llvm %s -o - | FileCheck %s
#include "sycl.hpp"

// Test cases below show that 'sycl-unique-id' LLVM IR attribute is attached to the
// global variable whose type is decorated with host_pipe attribute, and that a
// unique string is generated.

using namespace sycl::ext::intel::experimental;
using namespace sycl;
queue q;

// check that "sycl-unique-id" attribute is created for host pipes
// CHECK: @_ZN4sycl3_V13ext5intel12experimental9host_pipeIZZZ3foovENKUlRNS0_7handlerEE_clES6_ENKUlvE_clEvE5HPIntiE6__pipeE = internal addrspace(1) constant %"struct.sycl::_V1::ext::intel::experimental::host_pipe<HPInt, int>::__pipeType" zeroinitializer, align 1 #[[HPINT_ATTRS:[0-9]+]]
// CHECK: @_ZN4sycl3_V13ext5intel12experimental9host_pipeIZZZ3foovENKUlRNS0_7handlerEE_clES6_ENKUlvE_clEvE7HPFloatiE6__pipeE = internal addrspace(1) constant %"struct.sycl::_V1::ext::intel::experimental::host_pipe<HPFloat, int>::__pipeType" zeroinitializer, align 1 #[[HPFLOAT_ATTRS:[0-9]+]]

void foo() {
  q.submit([&](handler &h) {
    h.single_task<class kernel_name_1>([=]() {
      host_pipe<class HPInt, int>::read();
      host_pipe<class HPFloat, int>::read();
    });
  });
}

// CHECK: attributes #[[HPINT_ATTRS]] = { "sycl-unique-id"="THE_PREFIX____ZN4sycl3_V13ext5intel12experimental9host_pipeIZZZ3foovENKUlRNS0_7handlerEE_clES6_ENKUlvE_clEvE5HPIntiE6__pipeE" }
// CHECK: attributes #[[HPFLOAT_ATTRS]] = { "sycl-unique-id"="THE_PREFIX____ZN4sycl3_V13ext5intel12experimental9host_pipeIZZZ3foovENKUlRNS0_7handlerEE_clES6_ENKUlvE_clEvE7HPFloatiE6__pipeE" 

