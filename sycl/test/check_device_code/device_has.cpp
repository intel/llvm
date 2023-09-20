// RUN: %clangxx -fsycl -Xclang -fsycl-is-device -fsycl-device-only -Xclang -fno-sycl-early-optimizations -S -emit-llvm %s -o - | FileCheck %s

// Tests for IR of device_has(aspect, ...) attribute and
// !sycl_used_aspects metadata
#include <sycl/sycl.hpp>

using namespace sycl;
queue q;

// CHECK: define weak_odr dso_local spir_kernel void @{{.*}}kernel_name_1
// CHECK-SAME: !sycl_declared_aspects ![[ASPECTS1:[0-9]+]]
// CHECK-SAME: !srcloc ![[SRCLOC1:[0-9]+]]

// CHECK: define {{.*}}spir_func void @{{.*}}func1
// CHECK-SAME: !sycl_declared_aspects ![[ASPECTS1]]
// CHECK-SAME: !srcloc ![[SRCLOC2:[0-9]+]]
// CHECK-SAME: !sycl_used_aspects ![[ASPECTS1]]
[[sycl::device_has(sycl::aspect::cpu)]] void func1() {}

// CHECK: define {{.*}}spir_func void @{{.*}}func2
// CHECK-SAME: !sycl_declared_aspects ![[ASPECTS2:[0-9]+]]
// CHECK-SAME: !srcloc ![[SRCLOC3:[0-9]+]]
// CHECK-SAME: !sycl_used_aspects ![[ASPECTS2]]
[[sycl::device_has(sycl::aspect::fp16, sycl::aspect::gpu)]] void func2() {}

// CHECK: define {{.*}}spir_func void @{{.*}}func3
// CHECK-SAME: !sycl_declared_aspects ![[EMPTYASPECTS:[0-9]+]]
// CHECK-SAME: !srcloc ![[SRCLOC4:[0-9]+]]
[[sycl::device_has()]] void func3() {}

// CHECK: define {{.*}}spir_func void @{{.*}}func4
// CHECK-SAME: !sycl_declared_aspects ![[ASPECTS3:[0-9]+]]
// CHECK-SAME: !srcloc ![[SRCLOC5:[0-9]+]]
// CHECK-SAME: !sycl_used_aspects ![[ASPECTS3]]
template <sycl::aspect Aspect> [[sycl::device_has(Aspect)]] void func4() {}

// CHECK: define {{.*}}spir_func void @{{.*}}func5
// CHECK-SAME: !sycl_declared_aspects ![[ASPECTS1]]
// CHECK-SAME: !srcloc ![[SRCLOC6:[0-9]+]]
// CHECK-SAME: !sycl_used_aspects ![[ASPECTS1]]
[[sycl::device_has(sycl::aspect::cpu)]] void func5();
void func5() {}

constexpr sycl::aspect getAspect() { return sycl::aspect::cpu; }
// CHECK: define {{.*}}spir_func void @{{.*}}func6
// CHECK-SAME: !sycl_declared_aspects ![[ASPECTS1]]
// CHECK-SAME: !srcloc ![[SRCLOC7:[0-9]+]]
// CHECK-SAME: !sycl_used_aspects ![[ASPECTS1]]
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
    // CHECK: define weak_odr dso_local spir_kernel void @{{.*}}kernel_name_2
    // CHECK-SAME: !sycl_declared_aspects ![[ASPECTS4:[0-9]+]]
    // CHECK-SAME: !srcloc ![[SRCLOC8:[0-9]+]] {{.*}}
    h.single_task<class kernel_name_2>(
        []() [[sycl::device_has(sycl::aspect::gpu)]] {});
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
// CHECK-DAG: [[ASPECTS4]] = !{i32 2}
// CHECK-DAG: [[SRCLOC8]] = !{i32 {{[0-9]+}}}
