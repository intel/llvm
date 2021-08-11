// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -ast-dump %s | FileCheck %s

#include "sycl.hpp"

class AccessorBase {
  int A;

public:
  sycl::accessor<int, 1, sycl::access::mode::read_write,
                 sycl::access::target::local>
      acc;
};

class accessor {
public:
  int field;
};

class stream {
public:
  int field;
};

class sampler {
public:
  int field;
};

int main() {

  sycl::queue myQueue;
  sycl::handler H;
  AccessorBase Accessor1;
  accessor Accessor2 = {1};
  sycl::stream Stream1{0, 0, H};
  stream Stream2;
  sycl::sampler Sampler1;
  sampler Sampler2;

  myQueue.submit([&](sycl::handler &h) {
    h.single_task<class kernel_function1>([=]() {
      Accessor1.acc.use();
    });
    h.single_task<class kernel_function2>([=]() {
      int a = Accessor2.field;
    });

    h.single_task<class kernel_function3>([=]() {
      Stream1.use();
    });
    h.single_task<class kernel_function4>([=]() {
      int a = Stream2.field;
    });

    h.single_task<class kernelfunction5>([=] {
      Sampler1.use();
    });

    h.single_task<class kernelfunction6>([=] {
      int a = Sampler2.field;
    });
  });

  return 0;
}

// CHECK: ClassTemplateDecl {{.*}} accessor
// CHECK: CXXRecordDecl {{.*}} class accessor definition
// CHECK: SYCLSpecialClassAttr {{.*}}
// CHECK: CXXRecordDecl {{.*}} implicit class accessor

// CHECK: ClassTemplateSpecializationDecl {{.*}} class accessor definition
// CHECK: SYCLSpecialClassAttr{{.*}}
// CHECK: CXXRecordDecl {{.*}} prev {{.*}} implicit class accessor

// CHECK: ClassTemplateSpecializationDecl {{.*}} class accessor definition
// CHECK: SYCLSpecialClassAttr{{.*}}
// CHECK: CXXRecordDecl {{.*}} prev {{.*}} implicit class accessor

// CHECK: CXXRecordDecl {{.*}} referenced class sampler definition
// CHECK: SYCLSpecialClassAttr {{.*}}
// CHECK: CXXRecordDecl {{.*}} implicit class sampler

// CHECK: CXXRecordDecl {{.*}} prev {{.*}} referenced class stream definition
// CHECK: SYCLSpecialClassAttr {{.*}}
// CHECK: CXXRecordDecl {{.*}} implicit referenced class stream
