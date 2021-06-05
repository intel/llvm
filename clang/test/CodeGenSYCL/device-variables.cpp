// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

enum class test_type { value1, value2, value3 };

constexpr test_type global_value = test_type::value1;
static constexpr int my_array[1] = {42};

// CHECK: @{{.*}}global_value = internal addrspace(1) constant i32 0
// CHECK: @{{.*}}my_array = internal addrspace(1) constant [1 x i32] [i32 42]

void foo(const test_type &) {}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(const Func &kernelFunc) {
  kernelFunc();
}

int main() {
  constexpr test_type local_value = test_type::value2;
  const int some_const = 1;
  int some_local_var = 10;

  kernel<class test_kernel>([=]() {
    // Global variables used directly
    foo(global_value);
    // CHECK: call spir_func void @{{.*}}foo{{.*}}(i32 addrspace(4)* align 4 dereferenceable(4) addrspacecast (i32 addrspace(1)* @{{.*}}global_value to i32 addrspace(4)*))
    int a = my_array[0];
    // CHECK: [[LOAD:%[0-9]+]] = load i32, i32 addrspace(4)* getelementptr inbounds ([1 x i32], [1 x i32] addrspace(4)* addrspacecast ([1 x i32] addrspace(1)* @{{.*}}my_array to [1 x i32] addrspace(4)*), i64 0, i64 0)
    // CHECK: store i32 [[LOAD]], i32 addrspace(4)* %a
    int b = some_const;
    // Constant used directly
    // CHECK: store i32 1, i32 addrspace(4)* %b
    foo(local_value);
    // Local variables and constexprs captured by lambda
    // CHECK:  [[GEP:%[0-9]+]] = getelementptr inbounds %class.{{.*}}.anon, %class.{{.*}}.anon addrspace(4)* %{{.*}}, i32 0, i32 0
    // CHECK: call spir_func void @{{.*}}foo{{.*}}(i32 addrspace(4)* align 4 dereferenceable(4) [[GEP]])
    int some_device_local_var = some_local_var;
    // CHECK:  [[GEP1:%[0-9]+]] = getelementptr inbounds %class.{{.*}}.anon, %class.{{.*}}.anon addrspace(4)* %{{.*}}, i32 0, i32 1
    // CHECK:  [[LOAD1:%[0-9]+]] = load i32, i32 addrspace(4)* [[GEP1]]
    // CHECK:  store i32 [[LOAD1]], i32 addrspace(4)* %some_device_local_var
  });

  return 0;
}
