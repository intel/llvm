// REQUIRES: native_cpu_ock
// This test checks that the clang-linker-wrapper generates the expected output
// for Native CPU kernels
// RUN: %clangxx --offload-new-driver -fsycl -fsycl-targets=native_cpu %s -Xlinker --print-wrapped-module 2>&1 | FileCheck %s

#include <sycl/sycl.hpp>
using namespace sycl;

class Test;
class Test1;

int main() {
  sycl::queue q;
  sycl::nd_range<1> r(1, 1);
  q.submit([&](handler &h) {
    h.parallel_for<Test>(r, [=](nd_item<1> it) { it.barrier(); });
  });

  q.submit([&](handler &h) {
    h.parallel_for<Test1>(1, [=](item<1> it) { it.get_id(); });
  });
}

// check the declarations for the kernels
// CHECK-DAG: @__sycl_native_cpu_decls = internal constant [{{.*}} x %__nativecpu_entry]
// CHECK-DAG: @__ncpu_function_[[NAME1:.*]] = internal unnamed_addr constant [10 x i8] c"_ZTS4Test\00"
// CHECK-DAG: @__ncpu_function_[[NAME2:.*]] = internal unnamed_addr constant [11 x i8] c"_ZTS5Test1\00"
// CHECK-DAG: {{.*}}%__nativecpu_entry { ptr @__ncpu_function_[[NAME1]], ptr @_ZTS4Test.SYCLNCPU }
// CHECK-DAG: {{.*}}%__nativecpu_entry { ptr @__ncpu_function_[[NAME2]], ptr @_ZTS5Test1.SYCLNCPU }

// check the expected nd_range property attribute for the kernels
// CHECK-DAG: @__sycl_offload_prop_sets_arr.[[PROPS:.*]] = internal constant [{{.*}} x %_pi_device_binary_property_struct]
// CHECK-DAG: @prop.[[WITHNDRANGE:.*]] = internal unnamed_addr constant [22 x i8] c"_ZTS4Test@is_nd_range\00"
// CHECK-DAG: @prop.[[NONDRANGE:.*]] = internal unnamed_addr constant [23 x i8] c"_ZTS5Test1@is_nd_range\00"
// CHECK-DAG: {{.*}}%_pi_device_binary_property_struct { ptr @prop.[[WITHNDRANGE]], ptr null, i32 1, i64 1 }
// CHECK-DAG: {{.*}}%_pi_device_binary_property_struct { ptr @prop.[[NONDRANGE]], ptr null, i32 1, i64 0 }
