// RUN: %clang_cc1 -fsycl-is-device \
// RUN: -internal-isystem %S/Inputs -triple spir64-unknown-unknown \
// RUN:   -disable-llvm-passes -emit-llvm %s -o - %s | FileCheck --enable-var-scope %s

#include "sycl.hpp"
using namespace cl::sycl;
void test() {

  queue q;

  q.submit([&](handler &h) {
    cl::sycl::sampler smplr;
    cl::sycl::stream Stream(1024, 128, h);

    // CHECK: define {{.*}}spir_kernel void @{{.*}}esimd_kernel({{.*}}) #0 !kernel_arg_addr_space ![[NUM:[0-9]+]] !kernel_arg_access_qual ![[NUM1:[0-9]+]] !kernel_arg_type ![[NUM2:[0-9]+]] !kernel_arg_base_type ![[NUM2:[0-9]+]] !kernel_arg_type_qual ![[NUM3:[0-9]+]] !kernel_arg_accessor_ptr ![[NUM4:[0-9]+]] !sycl_explicit_simd ![[NUM5:[0-9]+]] !intel_reqd_sub_group_size ![[NUM6:[0-9]+]]
    h.single_task<class esimd_kernel>(
        [=]() [[intel::sycl_explicit_simd]] { smplr.use(); });
    // CHECK: define {{.*}}spir_kernel void @{{.*}}StreamTester({{.*}}) #0 !kernel_arg_addr_space ![[NUM:[0-9]+]] !kernel_arg_access_qual ![[NUM1:[0-9]+]] !kernel_arg_type ![[NUM2:[0-9]+]] !kernel_arg_base_type ![[NUM2:[0-9]+]] !kernel_arg_type_qual ![[NUM3:[0-9]+]] !kernel_arg_accessor_ptr ![[NUM4:[0-9]+]] !sycl_explicit_simd ![[NUM5:[0-9]+]] !intel_reqd_sub_group_size ![[NUM6:[0-9]+]]
    h.single_task<class StreamTester>([=]()
                                          [[intel::sycl_explicit_simd]] { Stream << "one"
                                                                                 << "two"; });
  });
}
