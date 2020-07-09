// RUN: %clang_cc1 -fsycl -fsycl-is-device -I %S/Inputs -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

// This test checks a kernel with an argument that is a POD array.

#include <sycl.hpp>

using namespace cl::sycl;

template <typename name, typename Func>
__attribute__((sycl_kernel)) void a_kernel(Func kernelFunc) {
  kernelFunc();
}

struct foo_inner {
  int foo_inner_x;
  int foo_inner_y;
};

struct foo {
  int foo_a;
  foo_inner foo_b[2];
  int foo_c;
};

int main() {

  int a[2];
  foo struct_array[2];

  a_kernel<class kernel_B>(
      [=]() {
        int local = a[1];
      });

  a_kernel<class kernel_C>(
      [=]() {
        foo local = struct_array[1];
      });
}

// Check kernel_B parameters
// CHECK: define spir_kernel void @{{.*}}kernel_B
// CHECK-SAME: i32 [[ELEM_ARG0:%[a-zA-Z0-9_]+]],
// CHECK-SAME: i32 [[ELEM_ARG1:%[a-zA-Z_]+_[0-9]+]])

// Check local lambda object alloca
// CHECK: [[LOCAL_OBJECT:%[0-9]+]] = alloca %"class.{{.*}}.anon", align 4

// Check local variables created for parameters
// CHECK: store i32 [[ELEM_ARG0]], i32* [[ELEM_L0:%[a-zA-Z_]+.addr]], align 4
// CHECK: store i32 [[ELEM_ARG1]], i32* [[ELEM_L1:%[a-zA-Z_]+.addr[0-9]*]], align 4

// Check init of local array
// CHECK: [[ARRAY:%[0-9]*]] = getelementptr inbounds %"class.{{.*}}.anon", %"class.{{.*}}.anon"* [[LOCAL_OBJECT]], i32 0, i32 0
// CHECK: [[ARRAY_BEGIN:%[a-zA-Z_.]+]] = getelementptr inbounds [2 x i32], [2 x i32]* [[ARRAY]], i64 0, i64 0
// CHECK: [[ARRAY0:%[0-9]*]] = load i32, i32* [[ELEM_L0]], align 4
// CHECK: store i32 [[ARRAY0]], i32* [[ARRAY_BEGIN]], align 4
// CHECK: [[ARRAY_ELEMENT:%[a-zA-Z_.]+]] = getelementptr inbounds i32, i32* %arrayinit.begin, i64 1
// CHECK: [[ARRAY1:%[0-9]*]] = load i32, i32* [[ELEM_L1]], align 4
// CHECK: store i32 [[ARRAY1]], i32* [[ARRAY_ELEMENT]], align 4

// Check kernel_C parameters
// CHECK: define spir_kernel void @{{.*}}kernel_C
// CHECK-SAME: i32 [[FOO1_A:%[a-zA-Z0-9_]+]], i32 [[FOO1_B1_X:%[a-zA-Z0-9_]+]], i32 [[FOO1_B1_Y:%[a-zA-Z0-9_]+]], i32 [[FOO1_B2_X:%[a-zA-Z0-9_]+]], i32 [[FOO1_B2_Y:%[a-zA-Z0-9_]+]], i32 [[FOO1_C:%[a-zA-Z0-9_]+]],
// CHECK-SAME: i32 [[FOO2_A:%[a-zA-Z0-9_]+]], i32 [[FOO2_B1_X:%[a-zA-Z0-9_]+]], i32 [[FOO2_B1_Y:%[a-zA-Z0-9_]+]], i32 [[FOO2_B2_X:%[a-zA-Z0-9_]+]], i32 [[FOO2_B2_Y:%[a-zA-Z0-9_]+]], i32 [[FOO2_C:%[a-zA-Z0-9_]+]]

// Check local lambda object alloca
// CHECK: [[KERNEL_OBJ:%[0-9]+]] = alloca %"class.{{.*}}.anon.0", align 4

// Check local stores
// CHECK: store i32 [[FOO1_A]], i32* [[FOO1_A_LOCAL:%[a-zA-Z_]+.addr[0-9]*]], align 4
// CHECK: store i32 [[FOO1_B1_X]], i32* [[FOO1_B1_X_LOCAL:%[a-zA-Z_]+.addr[0-9]*]], align 4
// CHECK: store i32 [[FOO1_B1_Y]], i32* [[FOO1_B1_Y_LOCAL:%[a-zA-Z_]+.addr[0-9]*]], align 4
// CHECK: store i32 [[FOO1_B2_X]], i32* [[FOO1_B2_X_LOCAL:%[a-zA-Z_]+.addr[0-9]*]], align 4
// CHECK: store i32 [[FOO1_B2_Y]], i32* [[FOO1_B2_Y_LOCAL:%[a-zA-Z_]+.addr[0-9]*]], align 4
// CHECK: store i32 [[FOO1_C]], i32* [[FOO1_C_LOCAL:%[a-zA-Z_]+.addr[0-9]*]], align 4
// CHECK: store i32 [[FOO2_A]], i32* [[FOO2_A_LOCAL:%[a-zA-Z_]+.addr[0-9]*]], align 4
// CHECK: store i32 [[FOO2_B1_X]], i32* [[FOO2_B1_X_LOCAL:%[a-zA-Z_]+.addr[0-9]*]], align 4
// CHECK: store i32 [[FOO2_B1_Y]], i32* [[FOO2_B1_Y_LOCAL:%[a-zA-Z_]+.addr[0-9]*]], align 4
// CHECK: store i32 [[FOO2_B2_X]], i32* [[FOO2_B2_X_LOCAL:%[a-zA-Z_]+.addr[0-9]*]], align 4
// CHECK: store i32 [[FOO2_B2_Y]], i32* [[FOO2_B2_Y_LOCAL:%[a-zA-Z_]+.addr[0-9]*]], align 4
// CHECK: store i32 [[FOO2_C]], i32* [[FOO2_C_LOCAL:%[a-zA-Z_]+.addr[0-9]*]], align 4

// Check initialization of local array

// Initialize struct_array[0].foo_a
// CHECK: [[GEP:%[a-zA-Z0-9_]+]] = getelementptr inbounds %"class.{{.*}}.anon.0", %"class.{{.*}}.anon.0"* [[KERNEL_OBJ]], i32 0, i32 0
// CHECK: [[FOO_ARRAY_0:%[a-zA-Z_.]+]] = getelementptr inbounds [2 x %struct.{{.*}}.foo], [2 x %struct.{{.*}}.foo]* [[GEP]], i64 0, i64 0
// CHECK: [[GEP_FOO1_A:%[a-zA-Z0-9_]+]] = getelementptr inbounds %struct.{{.*}}.foo, %struct.{{.*}}.foo* [[FOO_ARRAY_0]], i32 0, i32 0
// CHECK: [[LOAD_FOO1_A:%[a-zA-Z0-9_]+]] = load i32, i32* [[FOO1_A_LOCAL]], align 4
// CHECK: store i32 [[LOAD_FOO1_A]], i32* [[GEP_FOO1_A]], align 4

// Initialize struct_array[0].foo_b[0].x
// CHECK: [[GEP_FOO1_B:%[a-zA-Z0-9_]+]] = getelementptr inbounds %struct.{{.*}}.foo, %struct.{{.*}}.foo* [[FOO_ARRAY_0]], i32 0, i32 1
// CHECK: [[B_ARRAY_0:%[a-zA-Z0-9_.]+]] = getelementptr inbounds [2 x %struct.{{.*}}foo_inner.foo_inner], [2 x %struct.{{.*}}foo_inner.foo_inner]* [[GEP_FOO1_B]], i64 0, i64 0
// CHECK: [[GEP_FOO1_B1_X:%[a-zA-Z0-9_]+]] = getelementptr inbounds %struct.{{.*}}foo_inner.foo_inner, %struct.{{.*}}foo_inner.foo_inner* [[B_ARRAY_0]], i32 0, i32 0
// CHECK: [[LOAD_FOO1_B1_X:%[a-zA-Z0-9_]+]] = load i32, i32* [[FOO1_B1_X_LOCAL]], align 4
// store i32 [[LOAD_FOO1_B1_X]], i32* [[GEP_FOO1_B1_X]]

// Initialize struct_array[0].foo_b[0].y
// CHECK: [[GEP_FOO1_B1_Y:%[a-zA-Z0-9_]+]] = getelementptr inbounds %struct.{{.*}}foo_inner.foo_inner, %struct.{{.*}}foo_inner.foo_inner* [[B_ARRAY_0]], i32 0, i32 1
// CHECK: [[LOAD_FOO1_B1_Y:%[a-zA-Z0-9_]+]] = load i32, i32* [[FOO1_B1_Y_LOCAL]], align 4
// CHECK: store i32 [[LOAD_FOO1_B1_Y]], i32* [[GEP_FOO1_B1_Y]], align 4

// Initialize struct_array[0].foo_b[1].x
// CHECK: [[B_ARRAY_1:%[a-zA-Z0-9_.]+]] = getelementptr inbounds %struct.{{.*}}foo_inner.foo_inner, %struct.{{.*}}foo_inner.foo_inner* [[B_ARRAY_0]], i64 1
// CHECK: [[GEP_FOO1_B2_X:%[a-zA-Z0-9_]+]] = getelementptr inbounds %struct.{{.*}}foo_inner.foo_inner, %struct.{{.*}}foo_inner.foo_inner* [[B_ARRAY_1]], i32 0, i32 0
// CHECK: [[LOAD_FOO1_B2_X:%[a-zA-Z0-9_]+]] = load i32, i32* [[FOO1_B2_X_LOCAL]], align 4
// store i32 [[LOAD_FOO1_B2_X]], i32* [[GEP_FOO1_B2_X]], align 4

// Initialize struct_array[0].foo_b[1].y
// CHECK: [[GEP_FOO1_B2_Y:%[a-zA-Z0-9_]+]] = getelementptr inbounds %struct.{{.*}}foo_inner.foo_inner, %struct.{{.*}}foo_inner.foo_inner* [[B_ARRAY_1]], i32 0, i32 1
// CHECK: [[LOAD_FOO1_B2_Y:%[a-zA-Z0-9_]+]] = load i32, i32* [[FOO1_B2_Y_LOCAL]], align 4
// CHECK: store i32 [[LOAD_FOO1_B2_Y]], i32* [[GEP_FOO1_B2_Y]], align 4

// Initialize struct_array[0].foo_c
// CHECK: [[GEP_FOO1_C:%[a-zA-Z0-9_]+]] = getelementptr inbounds %struct.{{.*}}foo.foo, %struct.{{.*}}foo.foo* [[FOO_ARRAY_0]], i32 0, i32 2
// CHECK: [[LOAD_FOO1_C:%[a-zA-Z0-9_]+]] = load i32, i32* [[FOO1_C_LOCAL]], align 4
// CHECK: store i32 [[LOAD_FOO1_C]], i32* [[GEP_FOO1_C]], align 4

// Initialize struct_array[1].foo_a
// CHECK: [[FOO_ARRAY_1:%[a-zA-Z0-9_.]+]] = getelementptr inbounds %struct._ZTS3foo.foo, %struct._ZTS3foo.foo* [[FOO_ARRAY_0]], i64 1
// CHECK: [[GEP_FOO2_A:%[a-zA-Z0-9_]+]] = getelementptr inbounds %struct.{{.*}}foo.foo, %struct.{{.*}}foo.foo* [[FOO_ARRAY_1]], i32 0, i32 0
// CHECK: [[LOAD_FOO2_A:%[a-zA-Z0-9_]+]] = load i32, i32* [[FOO2_A_LOCAL]], align 4
// CHECK: store i32 [[LOAD_FOO2_A]], i32* [[GEP_FOO2_A]], align 4

// Initialize struct_array[1].foo_b[0].x
// CHECK: [[GEP_FOO2_B:%[a-zA-Z0-9_]+]] = getelementptr inbounds %struct.{{.*}}.foo, %struct.{{.*}}.foo* [[FOO_ARRAY_1]], i32 0, i32 1
// CHECK: [[FOO2_B_ARRAY_0:%[a-zA-Z0-9_.]+]] = getelementptr inbounds [2 x %struct.{{.*}}foo_inner.foo_inner], [2 x %struct.{{.*}}foo_inner.foo_inner]* [[GEP_FOO2_B]], i64 0, i64 0
// CHECK: [[GEP_FOO2_B1_X:%[a-zA-Z0-9_]+]] = getelementptr inbounds %struct.{{.*}}foo_inner.foo_inner, %struct.{{.*}}foo_inner.foo_inner* [[FOO2_B_ARRAY_0]], i32 0, i32 0
// CHECK: [[LOAD_FOO2_B1_X:%[a-zA-Z0-9_]+]] = load i32, i32* [[FOO2_B1_X_LOCAL]], align 4
// CHECK: store i32 [[LOAD_FOO2_B1_X]], i32* [[GEP_FOO2_B1_X]]

// Initialize struct_array[1].foo_b[0].y
// CHECK: [[GEP_FOO2_B1_Y:%[a-zA-Z0-9_]+]] = getelementptr inbounds %struct.{{.*}}foo_inner.foo_inner, %struct.{{.*}}foo_inner.foo_inner* [[FOO2_B_ARRAY_0]], i32 0, i32 1
// CHECK: [[LOAD_FOO2_B1_Y:%[a-zA-Z0-9_]+]] = load i32, i32* [[FOO2_B1_Y_LOCAL]], align 4
// CHECK: store i32 [[LOAD_FOO2_B1_Y]], i32* [[GEP_FOO2_B1_Y]], align 4

// Initialize struct_array[1].foo_b[1].x
// CHECK: [[FOO2_B_ARRAY_1:%[a-zA-Z0-9_.]+]] = getelementptr inbounds %struct.{{.*}}foo_inner.foo_inner, %struct.{{.*}}foo_inner.foo_inner* [[FOO2_B_ARRAY_0]], i64 1
// CHECK: [[GEP_FOO2_B2_X:%[a-zA-Z0-9_]+]] = getelementptr inbounds %struct.{{.*}}foo_inner.foo_inner, %struct.{{.*}}foo_inner.foo_inner* [[FOO2_B_ARRAY_1]], i32 0, i32 0
// CHECK: [[LOAD_FOO2_B2_X:%[a-zA-Z0-9_]+]] = load i32, i32* [[FOO2_B2_X_LOCAL]], align 4
// store i32 [[LOAD_FOO2_B2_X]], i32* [[GEP_FOO2_B2_X]], align 4

// Initialize struct_array[1].foo_b[1].y
// CHECK: [[GEP_FOO2_B2_Y:%[a-zA-Z0-9_]+]] = getelementptr inbounds %struct.{{.*}}foo_inner.foo_inner, %struct.{{.*}}foo_inner.foo_inner* [[FOO2_B_ARRAY_1]], i32 0, i32 1
// CHECK: [[LOAD_FOO2_B2_Y:%[a-zA-Z0-9_]+]] = load i32, i32* [[FOO2_B2_Y_LOCAL]], align 4
// CHECK: store i32 [[LOAD_FOO2_B2_Y]], i32* [[GEP_FOO2_B2_Y]], align 4

// Initialize struct_array[1].foo_c
// CHECK: [[GEP_FOO2_C:%[a-zA-Z0-9_]+]] = getelementptr inbounds %struct.{{.*}}foo.foo, %struct.{{.*}}foo.foo* [[FOO_ARRAY_1]], i32 0, i32 2
// CHECK: [[LOAD_FOO2_C:%[a-zA-Z0-9_]+]] = load i32, i32* [[FOO2_C_LOCAL]], align 4
// CHECK: store i32 [[LOAD_FOO2_C]], i32* [[GEP_FOO2_C]], align 4
