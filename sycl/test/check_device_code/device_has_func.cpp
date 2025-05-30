// RUN: %clangxx -fsycl -Xclang -fsycl-is-device -fsycl-device-only -Xclang -fno-sycl-early-optimizations -S -emit-llvm %s -o %t.ll
// RUN: FileCheck %s --input-file %t.ll --check-prefix=CHECK-ASPECTS

// Tests for IR of device_has(aspect, ...) attribute and
// !sycl_used_aspects metadata.
// We run FileCheck for 2 times to break metadata order dependency since
// compiler has no guarantee for meta data order.
#include <sycl/sycl.hpp>

using namespace sycl;

// CHECK-ASPECTS: define dso_local spir_func void @{{.*}}kernel_name_1{{.*}} !sycl_declared_aspects ![[ASPECTS1:[0-9]+]] {{.*}}

// CHECK-ASPECTS: define {{.*}}spir_func void @{{.*}}func1{{.*}} !sycl_declared_aspects ![[ASPECTS1]]
// CHECK-ASPECTS-SAME: !sycl_used_aspects ![[ASPECTS1]]
[[sycl::device_has(sycl::aspect::cpu)]] void func1() {}

// CHECK-ASPECTS: define {{.*}}spir_func void @{{.*}}func2{{.*}} !sycl_declared_aspects ![[ASPECTS2:[0-9]+]]
// CHECK-ASPECTS-SAME: !sycl_used_aspects ![[ASPECTS2]]
[[sycl::device_has(sycl::aspect::fp16, sycl::aspect::gpu)]] void func2() {}

// CHECK-ASPECTS: define {{.*}}spir_func void @{{.*}}func3{{.*}} !sycl_declared_aspects ![[EMPTYASPECTS:[0-9]+]]
[[sycl::device_has()]] void func3() {}

// CHECK-ASPECTS: define {{.*}}spir_func void @{{.*}}func4{{.*}} !sycl_declared_aspects ![[ASPECTS3:[0-9]+]]
// CHECK-ASPECTS-SAME: !sycl_used_aspects ![[ASPECTS3]]
template <sycl::aspect Aspect> [[sycl::device_has(Aspect)]] void func4() {}

// CHECK-ASPECTS: define {{.*}}spir_func void @{{.*}}func5{{.*}} !sycl_declared_aspects ![[ASPECTS1]]
// CHECK-ASPECTS-SAME: !sycl_used_aspects ![[ASPECTS1]]
[[sycl::device_has(sycl::aspect::cpu)]] void func5();
void func5() {}

constexpr sycl::aspect getAspect() { return sycl::aspect::cpu; }
// CHECK-ASPECTS: define {{.*}}spir_func void @{{.*}}func6{{.*}} !sycl_declared_aspects ![[ASPECTS1]]
// CHECK-ASPECTS-SAME: !sycl_used_aspects ![[ASPECTS1]]
[[sycl::device_has(getAspect())]] void func6() {}

SYCL_EXTERNAL [[sycl::device_has(sycl::aspect::cpu)]] void kernel_name_1() {
  func1();
  func2();
  func3();
  func4<sycl::aspect::host>();
  func5();
  func6();
}

// CHECK-ASPECTS: define dso_local spir_func void @{{.*}}kernel_name_2{{.*}} !sycl_declared_aspects ![[ASPECTS4:[0-9]+]]
SYCL_EXTERNAL [[sycl::device_has(sycl::aspect::gpu)]] void kernel_name_2() {}

// CHECK-ASPECTS-DAG: [[ASPECTS1]] = !{![[ASPECTCPU:[0-9]+]]}
// CHECK-ASPECTS-DAG: [[ASPECTCPU]] = !{!"cpu", i32 1}
// CHECK-ASPECTS-DAG: [[EMPTYASPECTS]] = !{}
// CHECK-ASPECTS-DAG: [[ASPECTS2]] = !{![[ASPECTFP16:[0-9]+]], ![[ASPECTGPU:[0-9]+]]}
// CHECK-ASPECTS-DAG: [[ASPECTFP16]] = !{!"fp16", i32 5}
// CHECK-ASPECTS-DAG: [[ASPECTGPU]] = !{!"gpu", i32 2}
// CHECK-ASPECTS-DAG: [[ASPECTS3]] = !{![[ASPECTHOST:[0-9]+]]}
// CHECK-ASPECTS-DAG: [[ASPECTHOST]] = !{!"host", i32 0}
// CHECK-ASPECTS-DAG: [[ASPECTS4]] = !{![[ASPECTGPU]]}
