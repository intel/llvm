// RUN:  %clang_cc1 -fsycl -fsycl-is-device -I %S/Inputs -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

#include <sycl.hpp>

class second_base {
public:
  int e;
};

class InnerFieldBase {
public:
  int d;
};
class InnerField : public InnerFieldBase {
  int c;
};

struct base {
public:
  int b;
  InnerField obj;
};

struct derived : base, second_base {
  int a;

  void operator()() {
  }
};

int main() {
  cl::sycl::queue q;

  q.submit([&](cl::sycl::handler &cgh) {
    derived f{};
    cgh.single_task(f);
  });

  return 0;
}

// Check kernel paramters
// CHECK: define spir_kernel void @{{.*}}derived(i32 %_arg_b, i32 %_arg_d, i32 %_arg_c, i32 %_arg_e, i32 %_arg_a)

// Check alloca for kernel paramters
// CHECK: %[[ARG_B:[a-zA-Z0-9_.]+]] = alloca i32, align 4
// CHECK: %[[ARG_D:[a-zA-Z0-9_.]+]] = alloca i32, align 4
// CHECK: %[[ARG_C:[a-zA-Z0-9_.]+]] = alloca i32, align 4
// CHECK: %[[ARG_E:[a-zA-Z0-9_.]+]] = alloca i32, align 4
// CHECK: %[[ARG_A:[a-zA-Z0-9_.]+]] = alloca i32, align 4

// Check alloca for local functor object
// CHECK: %[[LOCAL_OBJECT:[a-zA-Z0-9_.]+]] = alloca %struct.{{.*}}.derived, align 4

// Initialize field 'b'
// CHECK: %[[BITCAST1:[0-9]+]] = bitcast %struct.{{.*}}.derived* %[[LOCAL_OBJECT]] to %struct.{{.*}}.base*
// CHECK: %[[GEP_B:[a-zA-Z0-9]+]] = getelementptr inbounds %struct.{{.*}}.base, %struct.{{.*}}.base* %[[BITCAST1]], i32 0, i32 0
// CHECK: %[[LOAD_B:[0-9]+]] = load i32, i32* %[[ARG_B]], align 4
// CHECK: store i32 %[[LOAD_B]], i32* %[[GEP_B]], align 4

// Initialize field 'd'
// CHECK: %[[GEP_OBJ:[a-zA-Z0-9]+]] = getelementptr inbounds %struct.{{.*}}.base, %struct.{{.*}}.base* %[[BITCAST1]], i32 0, i32 1
// CHECK: %[[BITCAST2:[0-9]+]] = bitcast %class.{{.*}}.InnerField* %[[GEP_OBJ]] to %class.{{.*}}.InnerFieldBase*
// CHECK: %[[GEP_D:[a-zA-Z0-9]+]] = getelementptr inbounds %class.{{.*}}.InnerFieldBase, %class.{{.*}}.InnerFieldBase* %[[BITCAST2]], i32 0, i32 0
// CHECK: %[[LOAD_D:[0-9]+]] = load i32, i32* %[[ARG_D]], align 4
// CHECK: store i32 %[[LOAD_D]], i32* %[[GEP_D]], align 4

// Initialize field 'c'
// CHECK: %[[GEP_C:[a-zA-Z0-9]+]] = getelementptr inbounds %class.{{.*}}.InnerField, %class.{{.*}}.InnerField* %[[GEP_OBJ]], i32 0, i32 1
// CHECK: %[[LOAD_C:[0-9]+]] = load i32, i32* %[[ARG_C]], align 4
// CHECK: store i32 %[[LOAD_C]], i32* %[[GEP_C]], align 4

// Initialize field 'e'
// CHECK: %[[BITCAST3:[0-9]+]] = bitcast %struct.{{.*}}.derived* %[[LOCAL_OBJECT]] to i8*
// CHECK: %[[GEP_DERIVED:[a-zA-Z0-9]+]] = getelementptr inbounds i8, i8* %[[BITCAST3]], i64 12
// CHECK: %[[BITCAST4:[0-9]+]] = bitcast i8* %[[GEP_DERIVED]] to %class.{{.*}}.second_base*
// CHECK: %[[GEP_E:[a-zA-Z0-9]+]] = getelementptr inbounds %class.{{.*}}.second_base, %class.{{.*}}.second_base* %[[BITCAST4]], i32 0, i32 0
// CHECK: %[[LOAD_E:[0-9]+]] = load i32, i32* %[[ARG_E]], align 4
// CHECK: store i32 %[[LOAD_E]], i32* %[[GEP_E]], align 4

// Initialize field 'a'
// CHECK: %[[GEP_A:[a-zA-Z0-9]+]] = getelementptr inbounds %struct.{{.*}}.derived, %struct.{{.*}}.derived* %[[LOCAL_OBJECT]], i32 0, i32 2
// CHECK: %[[LOAD_A:[0-9]+]] = load i32, i32* %[[ARG_A]], align 4
// CHECK: store i32 %[[LOAD_A]], i32* %[[GEP_A]], align 4
