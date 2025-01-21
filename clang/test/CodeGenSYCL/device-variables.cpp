// RUN: %clang_cc1 -fno-sycl-force-inline-kernel-lambda -fsycl-is-device -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

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
    // CHECK: call spir_func void @{{.*}}foo{{.*}}(ptr addrspace(4) noundef align 4 dereferenceable(4) addrspacecast (ptr addrspace(1) @{{.*}}global_value to ptr addrspace(4)))
    int a = my_array[0];
    // CHECK: [[LOAD:%[0-9]+]] = load i32, ptr addrspace(4)
    // CHECK: store i32 [[LOAD]], ptr addrspace(4) %a
    int b = some_const;
    // Constant used directly
    // CHECK: store i32 1, ptr addrspace(4) %b
    foo(local_value);
    // Local variables and constexprs captured by lambda
    // CHECK:  [[GEP:%[a-z_]+]] = getelementptr inbounds nuw %class.anon, ptr addrspace(4) %{{.*}}, i32 0, i32 0
    // CHECK: call spir_func void @{{.*}}foo{{.*}}(ptr addrspace(4) noundef align 4 dereferenceable(4) [[GEP]])
    int some_device_local_var = some_local_var;
    // CHECK:  [[GEP1:%[a-z_]+]] = getelementptr inbounds nuw %class.anon, ptr addrspace(4) %{{.*}}, i32 0, i32 1
    // CHECK:  [[LOAD1:%[0-9]+]] = load i32, ptr addrspace(4) [[GEP1]]
    // CHECK:  store i32 [[LOAD1]], ptr addrspace(4) %some_device_local_var
  });

  return 0;
}
