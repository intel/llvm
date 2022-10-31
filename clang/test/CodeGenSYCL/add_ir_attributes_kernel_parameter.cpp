
// RUN: %clang_cc1 -internal-isystem %S/Inputs -disable-llvm-passes \
// RUN:    -triple spir64-unknown-unknown -fsycl-is-device -S \
// RUN:    -opaque-pointers -emit-llvm %s -o - | FileCheck %s

// Tests the generation of IR attributes when using
// __sycl_detail__::add_ir_attributes_kernel_parameter

#include "mock_properties.hpp"
#include "sycl.hpp"

// One __init parameter with add_ir_attributes_kernel_parameter attribute.

template <typename... Properties> class __attribute__((sycl_special_class)) __SYCL_TYPE(annotated_arg) g {
public:
  int *x;

  g() : x(nullptr) {}
  g(int *_x) : x(_x) {}

#ifdef __SYCL_DEVICE_ONLY__
  void __init(
      [[__sycl_detail__::add_ir_attributes_kernel_parameter(
          Properties::name..., Properties::value...)]] int *_x) {
    x = _x;
  }
#endif
};

class __attribute__((sycl_special_class)) __SYCL_TYPE(annotated_arg) h {
public:
  int *x;

  h() : x(nullptr) {}
  h(int *_x) : x(_x) {}

#ifdef __SYCL_DEVICE_ONLY__
  void __init(
      [[__sycl_detail__::add_ir_attributes_kernel_parameter(
          "Prop11", "Prop12", "Prop13", "Prop14", "Prop15", "Prop16", "Prop17", "Prop18",
          "Another property string", 2, false, TestEnum::Enum1, nullptr, nullptr, ScopedTestEnum::ScopedEnum2, PropertyValue8)]] int *_x) {
    x = _x;
  }
#endif
};

template <typename... Properties> class __attribute__((sycl_special_class)) __SYCL_TYPE(annotated_arg) gh {
public:
  int *x;

  gh() : x(nullptr) {}
  gh(int *_x) : x(_x) {}

#ifdef __SYCL_DEVICE_ONLY__
  void __init(
      [[__sycl_detail__::add_ir_attributes_kernel_parameter(
          Properties::name..., "Prop11", "Prop12", "Prop13", "Prop14", "Prop15", "Prop16", "Prop17", "Prop18",
          Properties::value..., "Another property string", 2, false, TestEnum::Enum1, nullptr, nullptr, ScopedTestEnum::ScopedEnum2, PropertyValue8)]] int *_x) {
    x = _x;
  }
#endif
};

template <typename... Properties> class __attribute__((sycl_special_class)) __SYCL_TYPE(annotated_arg) hg {
public:
  int *x;

  hg() : x(nullptr) {}
  hg(int *_x) : x(_x) {}

#ifdef __SYCL_DEVICE_ONLY__
  void __init(
      [[__sycl_detail__::add_ir_attributes_kernel_parameter(
          "Prop11", "Prop12", "Prop13", "Prop14", "Prop15", "Prop16", "Prop17", "Prop18", Properties::name...,
          "Another property string", 2, false, TestEnum::Enum1, nullptr, nullptr, ScopedTestEnum::ScopedEnum2, PropertyValue8, Properties::value...)]] int *_x) {
    x = _x;
  }
#endif
};

// Two __init parameters, one with add_ir_attributes_kernel_parameter attribute.

template <typename... Properties> class __attribute__((sycl_special_class)) __SYCL_TYPE(annotated_arg) k {
public:
  int *x;
  float *y;

  k() : x(nullptr), y(nullptr) {}
  k(int *_x, float *_y) : x(_x), y(_y) {}

#ifdef __SYCL_DEVICE_ONLY__
  void __init(
      int *_x,
      [[__sycl_detail__::add_ir_attributes_kernel_parameter(
          Properties::name..., Properties::value...)]] float *_y) {
    x = _x;
    y = _y;
  }
#endif
};

class __attribute__((sycl_special_class)) __SYCL_TYPE(annotated_arg) l {
public:
  int *x;
  float *y;

  l() : x(nullptr), y(nullptr) {}
  l(int *_x, float *_y) : x(_x), y(_y) {}

#ifdef __SYCL_DEVICE_ONLY__
  void __init(
      int *_x,
      [[__sycl_detail__::add_ir_attributes_kernel_parameter(
          "Prop11", "Prop12", "Prop13", "Prop14", "Prop15", "Prop16", "Prop17", "Prop18",
          "Another property string", 2, false, TestEnum::Enum1, nullptr, nullptr, ScopedTestEnum::ScopedEnum2, PropertyValue8)]] float *_y) {
    x = _x;
    y = _y;
  }
#endif
};

template <typename... Properties> class __attribute__((sycl_special_class)) __SYCL_TYPE(annotated_arg) kl {
public:
  int *x;
  float *y;

  kl() : x(nullptr), y(nullptr) {}
  kl(int *_x, float *_y) : x(_x), y(_y) {}

#ifdef __SYCL_DEVICE_ONLY__
  void __init(
      int *_x,
      [[__sycl_detail__::add_ir_attributes_kernel_parameter(
          Properties::name..., "Prop11", "Prop12", "Prop13", "Prop14", "Prop15", "Prop16", "Prop17", "Prop18",
          Properties::value..., "Another property string", 2, false, TestEnum::Enum1, nullptr, nullptr, ScopedTestEnum::ScopedEnum2, PropertyValue8)]] float *_y) {
    x = _x;
    y = _y;
  }
#endif
};

template <typename... Properties> class __attribute__((sycl_special_class)) __SYCL_TYPE(annotated_arg) lk {
public:
  int *x;
  float *y;

  lk() : x(nullptr), y(nullptr) {}
  lk(int *_x, float *_y) : x(_x), y(_y) {}

#ifdef __SYCL_DEVICE_ONLY__
  void __init(
      int *_x,
      [[__sycl_detail__::add_ir_attributes_kernel_parameter(
          "Prop11", "Prop12", "Prop13", "Prop14", "Prop15", "Prop16", "Prop17", "Prop18", Properties::name...,
          "Another property string", 2, false, TestEnum::Enum1, nullptr, nullptr, ScopedTestEnum::ScopedEnum2, PropertyValue8, Properties::value...)]] float *_y) {
    x = _x;
    y = _y;
  }
#endif
};

// Two __init parameters, both with add_ir_attributes_kernel_parameter attribute.

template <typename... Properties> class __attribute__((sycl_special_class)) __SYCL_TYPE(annotated_arg) m {
public:
  int *x;
  float *y;

  m() : x(nullptr), y(nullptr) {}
  m(int *_x, float *_y) : x(_x), y(_y) {}

#ifdef __SYCL_DEVICE_ONLY__
  void __init(
      [[__sycl_detail__::add_ir_attributes_kernel_parameter(
          Properties::name..., Properties::value...)]] int *_x,
      [[__sycl_detail__::add_ir_attributes_kernel_parameter(
          Properties::name..., Properties::value...)]] float *_y) {
    x = _x;
    y = _y;
  }
#endif
};

class __attribute__((sycl_special_class)) __SYCL_TYPE(annotated_arg) n {
public:
  int *x;
  float *y;

  n() : x(nullptr), y(nullptr) {}
  n(int *_x, float *_y) : x(_x), y(_y) {}

#ifdef __SYCL_DEVICE_ONLY__
  void __init(
      [[__sycl_detail__::add_ir_attributes_kernel_parameter(
          "Prop11", "Prop12", "Prop13", "Prop14", "Prop15", "Prop16", "Prop17", "Prop18",
          "Another property string", 2, false, TestEnum::Enum1, nullptr, nullptr, ScopedTestEnum::ScopedEnum2, PropertyValue8)]] int *_x,
      [[__sycl_detail__::add_ir_attributes_kernel_parameter(
          "Prop11", "Prop12", "Prop13", "Prop14", "Prop15", "Prop16", "Prop17", "Prop18",
          "Another property string", 2, false, TestEnum::Enum1, nullptr, nullptr, ScopedTestEnum::ScopedEnum2, PropertyValue8)]] float *_y) {
    x = _x;
    y = _y;
  }
#endif
};

template <typename... Properties> class __attribute__((sycl_special_class)) __SYCL_TYPE(annotated_arg) mn {
public:
  int *x;
  float *y;

  mn() : x(nullptr), y(nullptr) {}
  mn(int *_x, float *_y) : x(_x), y(_y) {}

#ifdef __SYCL_DEVICE_ONLY__
  void __init(
      [[__sycl_detail__::add_ir_attributes_kernel_parameter(
          Properties::name..., "Prop11", "Prop12", "Prop13", "Prop14", "Prop15", "Prop16", "Prop17", "Prop18",
          Properties::value..., "Another property string", 2, false, TestEnum::Enum1, nullptr, nullptr, ScopedTestEnum::ScopedEnum2, PropertyValue8)]] int *_x,
      [[__sycl_detail__::add_ir_attributes_kernel_parameter(
          Properties::name..., "Prop11", "Prop12", "Prop13", "Prop14", "Prop15", "Prop16", "Prop17", "Prop18",
          Properties::value..., "Another property string", 2, false, TestEnum::Enum1, nullptr, nullptr, ScopedTestEnum::ScopedEnum2, PropertyValue8)]] float *_y) {
    x = _x;
    y = _y;
  }
#endif
};

template <typename... Properties> class __attribute__((sycl_special_class)) __SYCL_TYPE(annotated_arg) nm {
public:
  int *x;
  float *y;

  nm() : x(nullptr), y(nullptr) {}
  nm(int *_x, float *_y) : x(_x), y(_y) {}

#ifdef __SYCL_DEVICE_ONLY__
  void __init(
      [[__sycl_detail__::add_ir_attributes_kernel_parameter(
          "Prop11", "Prop12", "Prop13", "Prop14", "Prop15", "Prop16", "Prop17", "Prop18", Properties::name...,
          "Another property string", 2, false, TestEnum::Enum1, nullptr, nullptr, ScopedTestEnum::ScopedEnum2, PropertyValue8, Properties::value...)]] int *_x,
      [[__sycl_detail__::add_ir_attributes_kernel_parameter(
          "Prop11", "Prop12", "Prop13", "Prop14", "Prop15", "Prop16", "Prop17", "Prop18", Properties::name...,
          "Another property string", 2, false, TestEnum::Enum1, nullptr, nullptr, ScopedTestEnum::ScopedEnum2, PropertyValue8, Properties::value...)]] float *_y) {
    x = _x;
    y = _y;
  }
#endif
};

// Empty attribute names.

class __attribute__((sycl_special_class)) __SYCL_TYPE(annotated_arg) np {
public:
  int *x;

  np() : x(nullptr) {}
  np(int *_x) : x(_x) {}

#ifdef __SYCL_DEVICE_ONLY__
  void __init(
      [[__sycl_detail__::add_ir_attributes_kernel_parameter(
          "", "", "", "", "", "", "", "",
          "Another property string", 2, false, TestEnum::Enum1, nullptr, nullptr, ScopedTestEnum::ScopedEnum2, PropertyValue8)]] int *_x) {
    x = _x;
  }
#endif
};

class __attribute__((sycl_special_class)) __SYCL_TYPE(annotated_arg) mp {
public:
  int *x;

  mp() : x(nullptr) {}
  mp(int *_x) : x(_x) {}

#ifdef __SYCL_DEVICE_ONLY__
  void __init(
      [[__sycl_detail__::add_ir_attributes_kernel_parameter(
          "", "Prop12", "", "", "", "Prop16", "Prop17", "Prop18",
          "Another property string", 2, false, TestEnum::Enum1, nullptr, nullptr, ScopedTestEnum::ScopedEnum2, PropertyValue8)]] int *_x) {
    x = _x;
  }
#endif
};

int main() {
  sycl::queue q;
  g<prop1, prop2, prop3, prop4, prop5, prop6, prop7, prop8> a1;
  q.submit([&](sycl::handler &h) {
    h.single_task<class test_kernel1>(
        [=]() {
          (void)a1;
        });
  });
  h b1;
  q.submit([&](sycl::handler &h) {
    h.single_task<class test_kernel2>(
        [=]() {
          (void)b1;
        });
  });
  gh<prop1, prop2, prop3, prop4, prop5, prop6, prop7, prop8> c1;
  q.submit([&](sycl::handler &h) {
    h.single_task<class test_kernel3>(
        [=]() {
          (void)c1;
        });
  });
  hg<prop1, prop2, prop3, prop4, prop5, prop6, prop7, prop8> d1;
  q.submit([&](sycl::handler &h) {
    h.single_task<class test_kernel4>(
        [=]() {
          (void)d1;
        });
  });
  k<prop1, prop2, prop3, prop4, prop5, prop6, prop7, prop8> a2;
  q.submit([&](sycl::handler &h) {
    h.single_task<class test_kernel5>(
        [=]() {
          (void)a2;
        });
  });
  l b2;
  q.submit([&](sycl::handler &h) {
    h.single_task<class test_kernel6>(
        [=]() {
          (void)b2;
        });
  });
  kl<prop1, prop2, prop3, prop4, prop5, prop6, prop7, prop8> c2;
  q.submit([&](sycl::handler &h) {
    h.single_task<class test_kernel7>(
        [=]() {
          (void)c2;
        });
  });
  lk<prop1, prop2, prop3, prop4, prop5, prop6, prop7, prop8> d2;
  q.submit([&](sycl::handler &h) {
    h.single_task<class test_kernel8>(
        [=]() {
          (void)d2;
        });
  });
  m<prop1, prop2, prop3, prop4, prop5, prop6, prop7, prop8> a3;
  q.submit([&](sycl::handler &h) {
    h.single_task<class test_kernel9>(
        [=]() {
          (void)a3;
        });
  });
  n b3;
  q.submit([&](sycl::handler &h) {
    h.single_task<class test_kernel10>(
        [=]() {
          (void)b3;
        });
  });
  mn<prop1, prop2, prop3, prop4, prop5, prop6, prop7, prop8> c3;
  q.submit([&](sycl::handler &h) {
    h.single_task<class test_kernel11>(
        [=]() {
          (void)c3;
        });
  });
  nm<prop1, prop2, prop3, prop4, prop5, prop6, prop7, prop8> d3;
  q.submit([&](sycl::handler &h) {
    h.single_task<class test_kernel12>(
        [=]() {
          (void)d3;
        });
  });
  np e;
  q.submit([&](sycl::handler &h) {
    h.single_task<class test_kernel13>(
        [=]() {
          (void)e;
        });
  });
  mp f;
  q.submit([&](sycl::handler &h) {
    h.single_task<class test_kernel14>(
        [=]() {
          (void)f;
        });
  });
}

// One __init parameter with add_ir_attributes_kernel_parameter attribute.
// CHECK-DAG: define {{.*}}spir_kernel void @{{.*}}test_kernel1({{.*}}ptr addrspace({{.*}}) {{.*}}"Prop1"="Property string"{{.*}}"Prop2"="1"{{.*}}"Prop3"="true"{{.*}}"Prop4"="2"{{.*}}"Prop5"{{.*}}"Prop6"{{.*}}"Prop7"="1"{{.*}}"Prop8"="Property"{{.*}} %{{.*}})
// CHECK-DAG: define {{.*}}spir_kernel void @{{.*}}test_kernel2({{.*}}ptr addrspace({{.*}}) {{.*}}"Prop11"="Another property string"{{.*}}"Prop12"="2"{{.*}}"Prop13"="false"{{.*}}"Prop14"="1"{{.*}}"Prop15"{{.*}}"Prop16"{{.*}}"Prop17"="2"{{.*}}"Prop18"="Property"{{.*}} %{{.*}})
// CHECK-DAG: define {{.*}}spir_kernel void @{{.*}}test_kernel3({{.*}}ptr addrspace({{.*}}) {{.*}}"Prop1"="Property string"{{.*}}"Prop11"="Another property string"{{.*}}"Prop12"="2"{{.*}}"Prop13"="false"{{.*}}"Prop14"="1"{{.*}}"Prop15"{{.*}}"Prop16"{{.*}}"Prop17"="2"{{.*}}"Prop18"="Property"{{.*}}"Prop2"="1"{{.*}}"Prop3"="true"{{.*}}"Prop4"="2"{{.*}}"Prop5"{{.*}}"Prop6"{{.*}}"Prop7"="1"{{.*}}"Prop8"="Property"{{.*}} %{{.*}})
// CHECK-DAG: define {{.*}}spir_kernel void @{{.*}}test_kernel4({{.*}}ptr addrspace({{.*}}) {{.*}}"Prop1"="Property string"{{.*}}"Prop11"="Another property string"{{.*}}"Prop12"="2"{{.*}}"Prop13"="false"{{.*}}"Prop14"="1"{{.*}}"Prop15"{{.*}}"Prop16"{{.*}}"Prop17"="2"{{.*}}"Prop18"="Property"{{.*}}"Prop2"="1"{{.*}}"Prop3"="true"{{.*}}"Prop4"="2"{{.*}}"Prop5"{{.*}}"Prop6"{{.*}}"Prop7"="1"{{.*}}"Prop8"="Property"{{.*}} %{{.*}})

// Two __init parameters, one with add_ir_attributes_kernel_parameter attribute.
// CHECK-DAG: define {{.*}}spir_kernel void @{{.*}}test_kernel5({{.*}}ptr addrspace({{.*}}) {{[^"]*}} %{{.*}}, {{.*}}ptr addrspace({{.*}}) {{.*}}"Prop1"="Property string"{{.*}}"Prop2"="1"{{.*}}"Prop3"="true"{{.*}}"Prop4"="2"{{.*}}"Prop5"{{.*}}"Prop6"{{.*}}"Prop7"="1"{{.*}}"Prop8"="Property"{{.*}} %{{.*}})
// CHECK-DAG: define {{.*}}spir_kernel void @{{.*}}test_kernel6({{.*}}ptr addrspace({{.*}}) {{[^"]*}} %{{.*}}, {{.*}}ptr addrspace({{.*}}) {{.*}}"Prop11"="Another property string"{{.*}}"Prop12"="2"{{.*}}"Prop13"="false"{{.*}}"Prop14"="1"{{.*}}"Prop15"{{.*}}"Prop16"{{.*}}"Prop17"="2"{{.*}}"Prop18"="Property"{{.*}} %{{.*}})
// CHECK-DAG: define {{.*}}spir_kernel void @{{.*}}test_kernel7({{.*}}ptr addrspace({{.*}}) {{[^"]*}} %{{.*}}, {{.*}}ptr addrspace({{.*}}) {{.*}}"Prop1"="Property string"{{.*}}"Prop11"="Another property string"{{.*}}"Prop12"="2"{{.*}}"Prop13"="false"{{.*}}"Prop14"="1"{{.*}}"Prop15"{{.*}}"Prop16"{{.*}}"Prop17"="2"{{.*}}"Prop18"="Property"{{.*}}"Prop2"="1"{{.*}}"Prop3"="true"{{.*}}"Prop4"="2"{{.*}}"Prop5"{{.*}}"Prop6"{{.*}}"Prop7"="1"{{.*}}"Prop8"="Property"{{.*}} %{{.*}})
// CHECK-DAG: define {{.*}}spir_kernel void @{{.*}}test_kernel8({{.*}}ptr addrspace({{.*}}) {{[^"]*}} %{{.*}}, {{.*}}ptr addrspace({{.*}}) {{.*}}"Prop1"="Property string"{{.*}}"Prop11"="Another property string"{{.*}}"Prop12"="2"{{.*}}"Prop13"="false"{{.*}}"Prop14"="1"{{.*}}"Prop15"{{.*}}"Prop16"{{.*}}"Prop17"="2"{{.*}}"Prop18"="Property"{{.*}}"Prop2"="1"{{.*}}"Prop3"="true"{{.*}}"Prop4"="2"{{.*}}"Prop5"{{.*}}"Prop6"{{.*}}"Prop7"="1"{{.*}}"Prop8"="Property"{{.*}} %{{.*}})

// Two __init parameters, both with add_ir_attributes_kernel_parameter attribute.
// CHECK-DAG: define {{.*}}spir_kernel void @{{.*}}test_kernel9({{.*}}ptr addrspace({{.*}}) {{.*}}"Prop1"="Property string"{{.*}}"Prop2"="1"{{.*}}"Prop3"="true"{{.*}}"Prop4"="2"{{.*}}"Prop5"{{.*}}"Prop6"{{.*}}"Prop7"="1"{{.*}}"Prop8"="Property"{{.*}} %{{.*}}, {{.*}}ptr addrspace({{.*}}) {{.*}}"Prop1"="Property string"{{.*}}"Prop2"="1"{{.*}}"Prop3"="true"{{.*}}"Prop4"="2"{{.*}}"Prop5"{{.*}}"Prop6"{{.*}}"Prop7"="1"{{.*}}"Prop8"="Property"{{.*}} %{{.*}})
// CHECK-DAG: define {{.*}}spir_kernel void @{{.*}}test_kernel10({{.*}}ptr addrspace({{.*}}) {{.*}}"Prop11"="Another property string"{{.*}}"Prop12"="2"{{.*}}"Prop13"="false"{{.*}}"Prop14"="1"{{.*}}"Prop15"{{.*}}"Prop16"{{.*}}"Prop17"="2"{{.*}}"Prop18"="Property"{{.*}} %{{.*}}, {{.*}}ptr addrspace({{.*}}) {{.*}}"Prop11"="Another property string"{{.*}}"Prop12"="2"{{.*}}"Prop13"="false"{{.*}}"Prop14"="1"{{.*}}"Prop15"{{.*}}"Prop16"{{.*}}"Prop17"="2"{{.*}}"Prop18"="Property"{{.*}} %{{.*}})
// CHECK-DAG: define {{.*}}spir_kernel void @{{.*}}test_kernel11({{.*}}ptr addrspace({{.*}}) {{.*}}"Prop1"="Property string"{{.*}}"Prop11"="Another property string"{{.*}}"Prop12"="2"{{.*}}"Prop13"="false"{{.*}}"Prop14"="1"{{.*}}"Prop15"{{.*}}"Prop16"{{.*}}"Prop17"="2"{{.*}}"Prop18"="Property"{{.*}}"Prop2"="1"{{.*}}"Prop3"="true"{{.*}}"Prop4"="2"{{.*}}"Prop5"{{.*}}"Prop6"{{.*}}"Prop7"="1"{{.*}}"Prop8"="Property"{{.*}} %{{.*}}, {{.*}}ptr addrspace({{.*}}) {{.*}}"Prop1"="Property string"{{.*}}"Prop11"="Another property string"{{.*}}"Prop12"="2"{{.*}}"Prop13"="false"{{.*}}"Prop14"="1"{{.*}}"Prop15"{{.*}}"Prop16"{{.*}}"Prop17"="2"{{.*}}"Prop18"="Property"{{.*}}"Prop2"="1"{{.*}}"Prop3"="true"{{.*}}"Prop4"="2"{{.*}}"Prop5"{{.*}}"Prop6"{{.*}}"Prop7"="1"{{.*}}"Prop8"="Property"{{.*}} %{{.*}})
// CHECK-DAG: define {{.*}}spir_kernel void @{{.*}}test_kernel12({{.*}}ptr addrspace({{.*}}) {{.*}}"Prop1"="Property string"{{.*}}"Prop11"="Another property string"{{.*}}"Prop12"="2"{{.*}}"Prop13"="false"{{.*}}"Prop14"="1"{{.*}}"Prop15"{{.*}}"Prop16"{{.*}}"Prop17"="2"{{.*}}"Prop18"="Property"{{.*}}"Prop2"="1"{{.*}}"Prop3"="true"{{.*}}"Prop4"="2"{{.*}}"Prop5"{{.*}}"Prop6"{{.*}}"Prop7"="1"{{.*}}"Prop8"="Property"{{.*}} %{{.*}}, {{.*}}ptr addrspace({{.*}}) {{.*}}"Prop1"="Property string"{{.*}}"Prop11"="Another property string"{{.*}}"Prop12"="2"{{.*}}"Prop13"="false"{{.*}}"Prop14"="1"{{.*}}"Prop15"{{.*}}"Prop16"{{.*}}"Prop17"="2"{{.*}}"Prop18"="Property"{{.*}}"Prop2"="1"{{.*}}"Prop3"="true"{{.*}}"Prop4"="2"{{.*}}"Prop5"{{.*}}"Prop6"{{.*}}"Prop7"="1"{{.*}}"Prop8"="Property"{{.*}} %{{.*}})

// Empty attribute names.
// CHECK-DAG: define {{.*}}spir_kernel void @{{.*}}test_kernel13({{.*}}ptr addrspace({{.*}}) {{.*}} %{{.*}})
// CHECK-DAG: define {{.*}}spir_kernel void @{{.*}}test_kernel14({{.*}}) {{.*}}"Prop12"="2"{{.*}}"Prop16"{{.*}}"Prop17"="2"{{.*}}"Prop18"="Property"{{.*}} %{{.*}})
// CHECK-NOT: ""="Another property string"
// CHECK-NOT: ""="1"
// CHECK-NOT: ""="2"
// CHECK-NOT: ""="false"
