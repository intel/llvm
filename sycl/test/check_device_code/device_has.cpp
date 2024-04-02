// RUN: %clangxx -fsycl -Xclang -fsycl-is-device -fsycl-device-only -Xclang -fno-sycl-early-optimizations -S -emit-llvm %s -o %t.ll
// RUN: FileCheck %s --input-file %t.ll --check-prefix=CHECK-ASPECTS
// RUN: FileCheck %s --input-file %t.ll --check-prefix=CHECK-SRCLOC

// Tests for IR of device_has(aspect, ...) attribute and
// !sycl_used_aspects metadata.
// We run FileCheck for 2 times to break metadata order dependency since
// compiler has no guarantee for meta data order.
#include <sycl/sycl.hpp>

using namespace sycl;
queue q;

// CHECK-ASPECTS: define weak_odr dso_local spir_kernel void @{{.*}}kernel_name_1{{.*}} !sycl_declared_aspects ![[ASPECTS1:[0-9]+]] {{.*}}
// CHECK-SRCLOC: define weak_odr dso_local spir_kernel void @{{.*}}kernel_name_1{{.*}} !srcloc ![[SRCLOC1:[0-9]+]] {{.*}}

// CHECK-ASPECTS: define {{.*}}spir_func void @{{.*}}func1{{.*}} !sycl_declared_aspects ![[ASPECTS1]]
// CHECK-ASPECTS-SAME: !sycl_used_aspects ![[ASPECTS1]]
// CHECK-SRCLOC: define {{.*}}spir_func void @{{.*}}func1{{.*}} !srcloc ![[SRCLOC2:[0-9]+]]
[[sycl::device_has(sycl::aspect::cpu)]] void func1() {}

// CHECK-ASPECTS: define {{.*}}spir_func void @{{.*}}func2{{.*}} !sycl_declared_aspects ![[ASPECTS2:[0-9]+]]
// CHECK-ASPECTS-SAME: !sycl_used_aspects ![[ASPECTS2]]
// CHECK-SRCLOC: define {{.*}}spir_func void @{{.*}}func2{{.*}} !srcloc ![[SRCLOC3:[0-9]+]]
[[sycl::device_has(sycl::aspect::fp16, sycl::aspect::gpu)]] void func2() {}

// CHECK-ASPECTS: define {{.*}}spir_func void @{{.*}}func3{{.*}} !sycl_declared_aspects ![[EMPTYASPECTS:[0-9]+]]
// CHECK-SRCLOC: define {{.*}}spir_func void @{{.*}}func3{{.*}} !srcloc ![[SRCLOC4:[0-9]+]]
[[sycl::device_has()]] void func3() {}

// CHECK-ASPECTS: define {{.*}}spir_func void @{{.*}}func4{{.*}} !sycl_declared_aspects ![[ASPECTS3:[0-9]+]]
// CHECK-ASPECTS-SAME: !sycl_used_aspects ![[ASPECTS3]]
// CHECK-SRCLOC: define {{.*}}spir_func void @{{.*}}func4{{.*}} !srcloc ![[SRCLOC5:[0-9]+]]
template <sycl::aspect Aspect> [[sycl::device_has(Aspect)]] void func4() {}

// CHECK-ASPECTS: define {{.*}}spir_func void @{{.*}}func5{{.*}} !sycl_declared_aspects ![[ASPECTS1]]
// CHECK-ASPECTS-SAME: !sycl_used_aspects ![[ASPECTS1]]
// CHECK-SRCLOC: define {{.*}}spir_func void @{{.*}}func5{{.*}} !srcloc ![[SRCLOC6:[0-9]+]]
[[sycl::device_has(sycl::aspect::cpu)]] void func5();
void func5() {}

constexpr sycl::aspect getAspect() { return sycl::aspect::cpu; }
// CHECK-ASPECTS: define {{.*}}spir_func void @{{.*}}func6{{.*}} !sycl_declared_aspects ![[ASPECTS1]]
// CHECK-ASPECTS-SAME: !sycl_used_aspects ![[ASPECTS1]]
// CHECK-SRCLOC: define {{.*}}spir_func void @{{.*}}func6{{.*}} !srcloc ![[SRCLOC7:[0-9]+]]
[[sycl::device_has(getAspect())]] void func6() {}

class KernelFunctor {
public:
  [[sycl::device_has(sycl::aspect::cpu)]] void operator()() const {
    func1();
    func2();
    func3();
    func4<sycl::aspect::host>();
    func5();
    func6();
  }
};

void foo() {
  q.submit([&](handler &h) {
    KernelFunctor f1;
    h.single_task<class kernel_name_1>(f1);
    // CHECK-ASPECTS: define weak_odr dso_local spir_kernel void @{{.*}}kernel_name_2{{.*}} !sycl_declared_aspects ![[ASPECTS4:[0-9]+]]
    // CHECK-SRCLOC: define weak_odr dso_local spir_kernel void @{{.*}}kernel_name_2{{.*}} !srcloc ![[SRCLOC8:[0-9]+]] {{.*}}
    h.single_task<class kernel_name_2>(
        []() [[sycl::device_has(sycl::aspect::gpu)]] {});
  });
}

// CHECK-ASPECTS-DAG: [[ASPECTS1]] = !{i32 1}
// CHECK-SRCLOC-DAG: [[SRCLOC1]] = !{i32 {{[0-9]+}}}
// CHECK-ASPECTS-DAG: [[EMPTYASPECTS]] = !{}
// CHECK-SRCLOC-DAG: [[SRCLOC2]] = !{i32 {{[0-9]+}}}
// CHECK-ASPECTS-DAG: [[ASPECTS2]] = !{i32 5, i32 2}
// CHECK-SRCLOC-DAG: [[SRCLOC3]] = !{i32 {{[0-9]+}}}
// CHECK-SRCLOC-DAG: [[SRCLOC4]] = !{i32 {{[0-9]+}}}
// CHECK-ASPECTS-DAG: [[ASPECTS3]] = !{i32 0}
// CHECK-SRCLOC-DAG: [[SRCLOC5]] = !{i32 {{[0-9]+}}}
// CHECK-SRCLOC-DAG: [[SRCLOC6]] = !{i32 {{[0-9]+}}}
// CHECK-SRCLOC-DAG: [[SRCLOC7]] = !{i32 {{[0-9]+}}}
// CHECK-ASPECTS-DAG: [[ASPECTS4]] = !{i32 2}
// CHECK-SRCLOC-DAG: [[SRCLOC8]] = !{i32 {{[0-9]+}}}
