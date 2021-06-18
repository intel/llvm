// RUN: %clang_cc1 -S -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -emit-llvm %s -o -  | FileCheck %s 

#include "Inputs/sycl.hpp"

cl::sycl::queue myQueue;
cl::sycl::handler SH;

class AccessorBase {
  int A;
public:
  cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                 cl::sycl::access::target::local>
  acc;
};

class accessor {
public:
  int field;
};

class stream {
public:
  int field;
};

class sampler {
public:
  int field;
  };

int main() {

  AccessorBase Accessor1;
  accessor Accessor2 = {1};
  stream Stream2;
  cl::sycl::sampler Sampler1;
  sampler Sampler2;

  myQueue.submit([&](cl::sycl::handler &h) {
    h.single_task<class kernel_function1>([=]() {
      Accessor1.acc.use();
    });
    h.single_task<class kernel_function2>([=]() {
      int a = Accessor2.field;
    });

    cl::sycl::stream Stream1{0, 0, SH};
    h.single_task<class kernel_function3>([=]() {
      int a = Stream2.field;
    });

    h.single_task<class kernelfunction4>([=] {
      Sampler1.use();
    });

    h.single_task<class kernelfunction5>([=] {
      int a = Sampler2.field;
    });

  });

  return 0;
}

// CHECK: %[[RANGE_TYPE:"struct.*cl::sycl::range"]]
// CHECK: %[[ID_TYPE:"struct.*cl::sycl::id"]]
// CHECK: define dso_local spir_kernel void @{{.*}}kernel_function1
// CHECK-SAME: i32 [[ARG_A:%[a-zA-Z0-9_]+]],
// CHECK-SAME: i32 addrspace(1)* [[ACC1_DATA:%[a-zA-Z0-9_]+]],
// CHECK-SAME: %[[RANGE_TYPE]]* byval(%[[RANGE_TYPE]]) align 4 [[ACC1_DATA:%[a-zA-Z0-9_]+]],
// CHECK-SAME: %[[RANGE_TYPE]]* byval(%[[RANGE_TYPE]]) align 4 [[ACC2_DATA:%[a-zA-Z0-9_]+]],
// CHECK-SAME: %[[ID_TYPE]]* byval(%[[ID_TYPE]]) align 4 [[ACC3_DATA:%[a-zA-Z0-9_]+]])

// CHECK: [[ACC_FIELD:%[a-zA-Z0-9_]+]] = getelementptr inbounds %class.{{.*}}.AccessorBase, %class.{{.*}}.AccessorBase addrspace(4)* %3, i32 0, i32 1
// CHECK: call spir_func void @_ZN2cl4sycl8accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2016ELNS2_11placeholderE0EEC1Ev(%"class.{{.*}}.cl::sycl::accessor" addrspace(4)* align 4 dereferenceable_or_null(12) [[ACC_FIELD]])

// CHECK: [[ACC1_FIELD:%[a-zA-Z0-9_]+]] = getelementptr inbounds %class{{.*}}.AccessorBase, %class{{.*}}.AccessorBase addrspace(4)* %5, i32 0, i32 1
// CHECK: [[ACC1_DATA_LOAD:%[a-zA-Z0-9_]+]] = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(4)* %_arg_acc.addr.ascast, align 8
// CHECK: call spir_func void @{{.*}}__init{{.*}}(%"class{{.*}}cl::sycl::accessor" addrspace(4)* {{.*}} [[ACC1_FIELD]], i32 addrspace(1)* [[ACC1_DATA_LOAD]]

