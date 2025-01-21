// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

// Tests for IR of __uses_aspects__(aspect, ...) attribute
#include "sycl.hpp"

using namespace sycl;
queue q;

class [[__sycl_detail__::__uses_aspects__(sycl::aspect::cpu)]] Type1WithAspect{};
class [[__sycl_detail__::__uses_aspects__(sycl::aspect::fp16, sycl::aspect::cpu)]] Type2WithAspect{};
class [[__sycl_detail__::__uses_aspects__(sycl::aspect::host)]] UnusedType3WithAspect{};

// CHECK: define {{.*}}spir_func void @{{.*}}func1
// CHECK-SAME: !sycl_used_aspects ![[ASPECTS1:[0-9]+]]
[[__sycl_detail__::__uses_aspects__(sycl::aspect::cpu)]] void func1() {}

// CHECK: define {{.*}}spir_func void @{{.*}}func2
// CHECK-SAME: !sycl_used_aspects ![[ASPECTS2:[0-9]+]]
[[__sycl_detail__::__uses_aspects__(sycl::aspect::fp16, sycl::aspect::gpu)]] void func2() {}

// CHECK: define {{.*}}spir_func void @{{.*}}func3
// CHECK-SAME: !sycl_used_aspects ![[EMPTYASPECTS:[0-9]+]]
[[__sycl_detail__::__uses_aspects__()]] void func3() {}

// CHECK: define {{.*}}spir_func void @{{.*}}func4
// CHECK-SAME: !sycl_used_aspects ![[ASPECTS3:[0-9]+]]
template <sycl::aspect Aspect>
[[__sycl_detail__::__uses_aspects__(Aspect)]] void func4() {}

// CHECK: define {{.*}}spir_func void @{{.*}}func5
// CHECK-SAME: !sycl_used_aspects ![[ASPECTS1]]
[[__sycl_detail__::__uses_aspects__(sycl::aspect::cpu)]] void func5();
void func5() {}

// CHECK: define {{.*}}spir_func void @{{.*}}func6
// CHECK-SAME: !sycl_used_aspects ![[ASPECTS1]]
[[__sycl_detail__::__uses_aspects__(sycl::aspect::cpu)]] void func6();
void func6() {
  Type1WithAspect TestObj1;
  Type2WithAspect TestObj2;
}

constexpr sycl::aspect getAspect() { return sycl::aspect::cpu; }
// CHECK: define {{.*}}spir_func void @{{.*}}func7
// CHECK-SAME: !sycl_used_aspects ![[ASPECTS1]]
[[__sycl_detail__::__uses_aspects__(getAspect())]] void func7() {}

// CHECK: declare !sycl_used_aspects ![[ASPECTS1]] spir_func void @{{.*}}func8
[[__sycl_detail__::__uses_aspects__(sycl::aspect::cpu)]] SYCL_EXTERNAL void func8();

class KernelFunctor {
public:
  void operator()() const {
    func1();
    func2();
    func3();
    func4<sycl::aspect::host>();
    func5();
    func6();
    func7();
    func8();
  }
};

void foo() {
  q.submit([&](handler &h) {
    KernelFunctor f1;
    h.single_task<class kernel_name_1>(f1);
  });
}
// CHECK: !sycl_types_that_use_aspects = !{![[TYPE1:[0-9]+]], ![[TYPE2:[0-9]+]]}
// CHECK: [[TYPE1]] = !{!"class.Type1WithAspect", i32 1}
// CHECK: [[TYPE2]] = !{!"class.Type2WithAspect", i32 5, i32 1}
// CHECK: [[EMPTYASPECTS]] = !{}
// CHECK: [[ASPECTS1]] = !{i32 1}
// CHECK: [[ASPECTS2]] = !{i32 5, i32 2}
// CHECK: [[ASPECTS3]] = !{i32 0}
