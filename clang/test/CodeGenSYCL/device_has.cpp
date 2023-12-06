// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

// Tests for IR of device_has(aspect, ...) attribute
#include "sycl.hpp"

using namespace sycl;
queue q;

// CHECK-DAG: define dso_local spir_kernel void @{{.*}}kernel_name_1{{.*}} !sycl_declared_aspects ![[ASPECTS1:[0-9]+]] !srcloc ![[SRCLOC1:[0-9]+]]

// CHECK-DAG: define {{.*}}spir_func void @{{.*}}func1{{.*}} !sycl_declared_aspects ![[ASPECTS1]] !srcloc ![[SRCLOC2:[0-9]+]] {
[[sycl::device_has(sycl::aspect::cpu)]] void func1() {}

// CHECK-DAG: define {{.*}}spir_func void @{{.*}}func2{{.*}} !sycl_declared_aspects ![[ASPECTS2:[0-9]+]] !srcloc ![[SRCLOC3:[0-9]+]] {
[[sycl::device_has(sycl::aspect::fp16, sycl::aspect::gpu)]] void func2() {}

// CHECK-DAG: define {{.*}}spir_func void @{{.*}}func3{{.*}} !sycl_declared_aspects ![[EMPTYASPECTS:[0-9]+]] !srcloc ![[SRCLOC4:[0-9]+]] {
[[sycl::device_has()]] void func3() {}

// CHECK-DAG: define {{.*}}spir_func void @{{.*}}func4{{.*}} !sycl_declared_aspects ![[ASPECTS3:[0-9]+]] !srcloc ![[SRCLOC5:[0-9]+]] {
template <sycl::aspect Aspect>
[[sycl::device_has(Aspect)]] void func4() {}

// CHECK-DAG: define {{.*}}spir_func void @{{.*}}func5{{.*}} !sycl_declared_aspects ![[ASPECTS1]] !srcloc ![[SRCLOC6:[0-9]+]] {
[[sycl::device_has(sycl::aspect::cpu)]] void func5();
void func5() {}

constexpr sycl::aspect getAspect() { return sycl::aspect::cpu; }
// CHECK-DAG: define {{.*}}spir_func void @{{.*}}func6{{.*}} !sycl_declared_aspects ![[ASPECTS1]] !srcloc ![[SRCLOC7:[0-9]+]] {
[[sycl::device_has(getAspect())]] void func6() {}

// CHECK-DAG: define {{.*}}spir_func void @{{.*}}func7{{.*}} !sycl_declared_aspects ![[ASPECTS1]]
// CHECK-DAG: define {{.*}}spir_func void @{{.*}}func7{{.*}} !sycl_declared_aspects ![[ASPECTS5:[0-9]+]]
template <sycl::aspect... Asp>
[[sycl::device_has(Asp...)]] void func7() {}

// CHECK-DAG: define {{.*}}spir_func void @{{.*}}func8{{.*}} !sycl_declared_aspects ![[ASPECTS5]]
template <sycl::aspect Asp, sycl::aspect... AspPack>
[[sycl::device_has(Asp, AspPack...)]] void func8() {}

// CHECK-DAG: declare !sycl_declared_aspects ![[ASPECTS6:[0-9]+]] spir_func void @{{.*}}func9{{.*}}
[[sycl::device_has(sycl::aspect::fp16)]]
SYCL_EXTERNAL void func9();

// CHECK-DAG: define dso_local spir_func void @{{.*}}func10{{.*}} !sycl_declared_aspects ![[ASPECTS6]]
[[sycl::device_has(sycl::aspect::fp16)]]
SYCL_EXTERNAL void func10() {}

class KernelFunctor {
public:
  [[sycl::device_has(sycl::aspect::cpu)]] void operator()() const {
    func1();
    func2();
    func3();
    func4<sycl::aspect::host>();
    func5();
    func6();
    func7<sycl::aspect::cpu>();
    func7<sycl::aspect::cpu, sycl::aspect::host>();
    func8<sycl::aspect::cpu, sycl::aspect::host>();
    func9();
    func10();
  }
};

void foo() {
  q.submit([&](handler &h) {
    KernelFunctor f1;
    h.single_task<class kernel_name_1>(f1);
    // CHECK-DAG: define dso_local spir_kernel void @{{.*}}kernel_name_2{{.*}} !sycl_declared_aspects ![[ASPECTS4:[0-9]+]] !srcloc ![[SRCLOC8:[0-9]+]]
    h.single_task<class kernel_name_2>([]() [[sycl::device_has(sycl::aspect::gpu)]] {});
  });
}

// CHECK-DAG: [[ASPECTS1]] = !{i32 1}
// CHECK-DAG: [[SRCLOC1]] = !{i32 {{[0-9]+}}}
// CHECK-DAG: [[EMPTYASPECTS]] = !{}
// CHECK-DAG: [[SRCLOC2]] = !{i32 {{[0-9]+}}}
// CHECK-DAG: [[ASPECTS2]] = !{i32 5, i32 2}
// CHECK-DAG: [[SRCLOC3]] = !{i32 {{[0-9]+}}}
// CHECK-DAG: [[SRCLOC4]] = !{i32 {{[0-9]+}}}
// CHECK-DAG: [[ASPECTS3]] = !{i32 0}
// CHECK-DAG: [[SRCLOC5]] = !{i32 {{[0-9]+}}}
// CHECK-DAG: [[SRCLOC6]] = !{i32 {{[0-9]+}}}
// CHECK-DAG: [[SRCLOC7]] = !{i32 {{[0-9]+}}}
// CHECK-DAG: [[ASPECTS5]] = !{i32 1, i32 0}
// CHECK-DAG: [[ASPECTS6]] = !{i32 5}
// CHECK-DAG: [[ASPECTS4]] = !{i32 2}
// CHECK-DAG: [[SRCLOC8]] = !{i32 {{[0-9]+}}}
