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

    h.single_task<class kernel_function4>([=] {
      Sampler1.use();
    });

    h.single_task<class kernel_function5>([=] {
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
// CHECK: call spir_func void @_ZN2cl4sycl8accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2016ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEC1Ev(%"class.{{.*}}.cl::sycl::accessor" addrspace(4)* align 4 dereferenceable_or_null(12) [[ACC_FIELD]])

// CHECK: [[ACC1_FIELD:%[a-zA-Z0-9_]+]] = getelementptr inbounds %class{{.*}}.AccessorBase, %class{{.*}}.AccessorBase addrspace(4)* %5, i32 0, i32 1
// CHECK: [[ACC1_DATA_LOAD:%[a-zA-Z0-9_]+]] = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(4)* %_arg_acc.addr.ascast, align 8
// CHECK: call spir_func void @{{.*}}__init{{.*}}(%"class{{.*}}cl::sycl::accessor" addrspace(4)* {{.*}} [[ACC1_FIELD]], i32 addrspace(1)* [[ACC1_DATA_LOAD]]

// CHECK: define dso_local spir_kernel void @{{.*}}kernel_function2(%class.{{.*}}.accessor* byval(%class.{{.*}}.accessor) align 4 [[ARG_F2:%[a-zA-Z0-9_]+]])
// CHECK: [[KERNEL_F2:%[a-zA-Z0-9_]+]] = alloca %class.{{.*}}.anon
// CHECK: [[KERNEL_OBJ_F2:%[a-zA-Z0-9_]+]] = addrspacecast %class.{{.*}}.anon* [[KERNEL_F2]] to %class.{{.*}}.anon addrspace(4)*
// CHECK: call spir_func void @_ZZZ4mainENKUlRN2cl4sycl7handlerEE_clES2_ENKUlvE0_clEv(%class.{{.*}}.anon addrspace(4)* align 4 dereferenceable_or_null(4) [[KERNEL_OBJ_F2]]

// CHECK: define dso_local spir_kernel void @{{.*}}kernel_function3(%class.{{.*}}.stream* byval(%class.{{.*}}.stream) align 4 [[ARG_F3:%[a-zA-Z0-9_]+]])
// CHECK: [[KERNEL_F3:%[a-zA-Z0-9_]+]] = alloca %class.{{.*}}.anon
// CHECK: [[KERNEL_OBJ_F3:%[a-zA-Z0-9_]+]] = addrspacecast %class.{{.*}}.anon* [[KERNEL_F3]] to %class.{{.*}}.anon addrspace(4)*
// CHECK: call spir_func void @_ZZZ4mainENKUlRN2cl4sycl7handlerEE_clES2_ENKUlvE1_clEv(%class.{{.*}}.anon addrspace(4)* align 4 dereferenceable_or_null(4) [[KERNEL_OBJ_F3]])

// CHECK: define dso_local spir_kernel void @_{{.*}}kernel_function4(%opencl.sampler_t addrspace(2)* [[ARG_F3:%[a-zA-Z0-9_]+]])
// CHECK: [[ARG_F3]].addr = alloca %opencl.sampler_t addrspace(2)*, align 8
// CHECK: [[ARG_F3]].addr.ascast = addrspacecast %opencl.sampler_t addrspace(2)** [[ARG_F3]].addr to %opencl.sampler_t addrspace(2)* addrspace(4)*
// CHECK: [[ANON_F4:%[0-9]+]] = alloca %class.{{.*}}.anon, align 8
// CHECK: [[ANON_CAST_F4:%[0-9]+]] = addrspacecast %class.{{.*}}.anon* [[ANON_F4]] to %class.{{.*}}.anon addrspace(4)*
// CHECK: [[GEP_F3:%[0-9]+]] = getelementptr inbounds %class.{{.*}}.anon, %class.{{.*}}.anon addrspace(4)* [[ANON_CAST_F4]], i32 0, i32 0
// CHECK: [[LOAD_ARG_F4:%[0-9]+]] = load %opencl.sampler_t addrspace(2)*, %opencl.sampler_t addrspace(2)* addrspace(4)* [[ARG_F3]].addr.ascast, align 8
// CHECK:  call spir_func void @_ZN2cl4sycl7sampler6__initE11ocl_sampler(%"class.{{.*}}.cl::sycl::sampler" addrspace(4)* align 8 dereferenceable_or_null(8) [[GEP_F3]], %opencl.sampler_t addrspace(2)* [[LOAD_ARG_F4]])

// CHECK: define dso_local spir_kernel void @_{{.*}}kernel_function5(%class.{{.*}}.sampler* byval(%class.{{.*}}.sampler) align 4 [[ARG_F5:%[a-zA-Z0-9_]+]])
// CHECK: [[KERNEL_F5:%[0-9]+]] = alloca %class.{{.*}}.anon, align 4
// CHECK: [[KERNEL_OBJ_F5:%[0-9]+]] = addrspacecast %class.{{.*}}.anon* [[KERNEL_F5]] to %class.{{.*}}.anon addrspace(4)*
// CHECK: call spir_func void @_ZZZ4mainENKUlRN2cl4sycl7handlerEE_clES2_ENKUlvE3_clEv(%class.{{.*}}.anon addrspace(4)* align 4 dereferenceable_or_null(4) [[KERNEL_OBJ_F5]])
