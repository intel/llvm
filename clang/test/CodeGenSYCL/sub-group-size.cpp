// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2020 -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefixes=NONE,ALL
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -fsycl-default-sub-group-size=primary -sycl-std=2020 -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefixes=PRIM_DEF,ALL
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -fsycl-default-sub-group-size=10 -sycl-std=2020 -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefixes=TEN_DEF,ALL

// Ensure that both forms of the new sub_group_size properly emit their metadata
// on sycl-kernel and sycl-external functions.

#include "Inputs/sycl.hpp"
using namespace cl::sycl;

[[intel::named_sub_group_size(primary)]] SYCL_EXTERNAL void external_primary() {}
// ALL-DAG: define {{.*}}spir_func void @{{.*}}external_primary{{.*}}() #{{[0-9]+}} !intel_reqd_sub_group_size ![[PRIMARY:[0-9]+]]

[[intel::sub_group_size(10)]] SYCL_EXTERNAL void external_10() {}
// ALL-DAG: define {{.*}}spir_func void @{{.*}}external_10{{.*}}() #{{[0-9]+}} !intel_reqd_sub_group_size ![[TEN:[0-9]+]]

SYCL_EXTERNAL void external_default_behavior() {}
// NONE-DAG: define {{.*}}spir_func void @{{.*}}external_default_behavior{{.*}}() #{{[0-9]+}} {
// PRIM_DEF-DAG: define {{.*}}spir_func void @{{.*}}external_default_behavior{{.*}}() #{{[0-9]+}} !intel_reqd_sub_group_size ![[PRIMARY]] {
// TEN_DEF-DAG: define {{.*}}spir_func void @{{.*}}external_default_behavior{{.*}}() #{{[0-9]+}} !intel_reqd_sub_group_size ![[TEN]] {

void default_behavior() {
  kernel_single_task<class Kernel1>([]() {
  });
}
// NONE-DAG: define {{.*}}spir_kernel void @{{.*}}Kernel1() #{{[0-9]+}} !kernel_arg_buffer_location !{{[0-9]+}} {
// PRIM_DEF-DAG: define {{.*}}spir_kernel void @{{.*}}Kernel1() #{{[0-9]+}} !intel_reqd_sub_group_size ![[PRIMARY]]
// TEN_DEF-DAG: define {{.*}}spir_kernel void @{{.*}}Kernel1() #{{[0-9]+}} !intel_reqd_sub_group_size ![[TEN]]

void primary() {
  kernel_single_task<class Kernel2>([]() [[intel::named_sub_group_size(primary)]]{});
}
// ALL-DAG: define {{.*}}spir_kernel void @{{.*}}Kernel2() #{{[0-9]+}} !intel_reqd_sub_group_size ![[PRIMARY]]

void ten() {
  kernel_single_task<class Kernel3>([]() [[intel::sub_group_size(10)]]{});
}
// ALL-DAG: define {{.*}}spir_kernel void @{{.*}}Kernel3() #{{[0-9]+}} !intel_reqd_sub_group_size ![[TEN]]

// PRIM_DEF: ![[PRIMARY]] = !{!"primary"}
// TEN_DEF: ![[TEN]] = !{i32 10}
