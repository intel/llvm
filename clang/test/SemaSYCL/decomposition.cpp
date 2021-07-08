// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -ast-dump -sycl-std=2020 %s | FileCheck %s

// This test checks that the compiler decomposes structs containing special types only
// (i.e. accessor/stream/sampler etc) and all others are passed without decomposition
// thus optimizing the number of kernel arguments.

#include "sycl.hpp"

sycl::queue myQueue;

struct StructWithAccessor {
  sycl::accessor<char, 1, sycl::access::mode::read> acc;
};

struct StructInheritedAccessor : sycl::accessor<char, 1, sycl::access::mode::read> {
  int i;
};

struct StructWithSampler {
  sycl::sampler sampl;
};

struct StructWithSpecConst {
  sycl::ext::oneapi::experimental::spec_constant<int, class f1> SC;
};

sycl::handler H;

struct StructWithStream {
  sycl::stream s1{0, 0, H};
};

struct StructWithHalf {
  sycl::half h;
};

struct StructNonDecomposed {
  int i;
  float f;
  double d;
};

struct StructWithNonDecomposedStruct : StructNonDecomposed {
  StructNonDecomposed member;
  float f;
  double d;
};

template <typename T>
struct StructWithArray {
  T a;
  T b[2];
  StructNonDecomposed SimpleStruct;
  int i;
};

template <typename T>
struct DerivedStruct : T {
  StructNonDecomposed SimpleStruct;
  int i;
};

int main() {

  StructNonDecomposed SimpleStruct;
  StructNonDecomposed ArrayOfSimpleStruct[5];
  StructWithNonDecomposedStruct NonDecompStruct;
  StructWithNonDecomposedStruct ArrayOfNonDecompStruct[5];
  // Check to ensure that these are not decomposed.
  myQueue.submit([&](sycl::handler &h) {
    h.single_task<class NonDecomposed>([=]() { return SimpleStruct.i + ArrayOfSimpleStruct[0].i + NonDecompStruct.i + ArrayOfNonDecompStruct[0].i; });
  });
  // CHECK: FunctionDecl {{.*}}NonDecomposed{{.*}} 'void (StructNonDecomposed, __wrapper_class, StructWithNonDecomposedStruct, __wrapper_class)'

  {
    StructWithArray<StructWithAccessor> t1;
    myQueue.submit([&](sycl::handler &h) {
      h.single_task<class Acc1>([=]() { return t1.i; });
    });
    // CHECK: FunctionDecl {{.*}}Acc1{{.*}} 'void (__global char *, sycl::range<1>, sycl::range<1>, sycl::id<1>, __global char *, sycl::range<1>, sycl::range<1>, sycl::id<1>, __global char *, sycl::range<1>, sycl::range<1>, sycl::id<1>, StructNonDecomposed, int)'

    DerivedStruct<StructWithAccessor> t2;
    myQueue.submit([&](sycl::handler &h) {
      h.single_task<class Acc2>([=]() { return t2.i; });
    });
    // CHECK: FunctionDecl {{.*}}Acc2{{.*}} 'void (__global char *, sycl::range<1>, sycl::range<1>, sycl::id<1>, StructNonDecomposed, int)'

    StructWithArray<StructInheritedAccessor> t3;
    myQueue.submit([&](sycl::handler &h) {
      h.single_task<class Acc3>([=]() { return t3.i; });
    });
    // CHECK: FunctionDecl {{.*}}Acc3{{.*}} 'void (__global char *, sycl::range<1>, sycl::range<1>, sycl::id<1>, int, __global char *, sycl::range<1>, sycl::range<1>, sycl::id<1>, int, __global char *, sycl::range<1>, sycl::range<1>, sycl::id<1>, int, StructNonDecomposed, int)'

    DerivedStruct<StructInheritedAccessor> t4;
    myQueue.submit([&](sycl::handler &h) {
      h.single_task<class Acc4>([=]() { return t4.i; });
    });
    // CHECK: FunctionDecl {{.*}}Acc4{{.*}} 'void (__global char *, sycl::range<1>, sycl::range<1>, sycl::id<1>, int, StructNonDecomposed, int)'
  }

  {
    StructWithArray<StructWithSampler> t1;
    myQueue.submit([&](sycl::handler &h) {
      h.single_task<class Sampl1>([=]() { return t1.i; });
    });
    // CHECK: FunctionDecl {{.*}}Sampl1{{.*}} 'void (sampler_t, sampler_t, sampler_t, StructNonDecomposed, int)'

    DerivedStruct<StructWithSampler> t2;
    myQueue.submit([&](sycl::handler &h) {
      h.single_task<class Sampl2>([=]() { return t2.i; });
    });
    // CHECK: FunctionDecl {{.*}}Sampl2{{.*}} 'void (sampler_t, StructNonDecomposed, int)'
  }

  {
    StructWithArray<StructWithSpecConst> t1;
    myQueue.submit([&](sycl::handler &h) {
      h.single_task<class SpecConst1>([=]() { return t1.i; });
    });
    // CHECK: FunctionDecl {{.*}}SpecConst{{.*}} 'void (StructNonDecomposed, int)'

    DerivedStruct<StructWithSpecConst> t2;
    myQueue.submit([&](sycl::handler &h) {
      h.single_task<class SpecConst2>([=]() { return t2.i; });
    });
    // CHECK: FunctionDecl {{.*}}SpecConst2{{.*}} 'void (StructNonDecomposed, int)'
  }

  {
    StructWithArray<StructWithStream> t1;
    myQueue.submit([&](sycl::handler &h) {
      h.single_task<class Stream1>([=]() { return t1.i; });
    });
    // CHECK: FunctionDecl {{.*}}Stream1{{.*}} 'void (__global char *, sycl::range<1>, sycl::range<1>, sycl::id<1>, int, __global char *, sycl::range<1>, sycl::range<1>, sycl::id<1>, int, __global char *, sycl::range<1>, sycl::range<1>, sycl::id<1>, int, StructNonDecomposed, int)'
    DerivedStruct<StructWithStream> t2;
    myQueue.submit([&](sycl::handler &h) {
      h.single_task<class Stream2>([=]() { return t2.i; });
    });
    // CHECK: FunctionDecl {{.*}}Stream2{{.*}} 'void (__global char *, sycl::range<1>, sycl::range<1>, sycl::id<1>, int, StructNonDecomposed, int)'
  }

  {
    StructWithArray<StructWithHalf> t1;
    myQueue.submit([&](sycl::handler &h) {
      h.single_task<class Half1>([=]() { return t1.i; });
    });
    // CHECK: FunctionDecl {{.*}}Half1{{.*}} 'void (sycl::half, sycl::half, sycl::half, StructNonDecomposed, int)'

    DerivedStruct<StructWithHalf> t2;
    myQueue.submit([&](sycl::handler &h) {
      h.single_task<class Half2>([=]() { return t2.i; });
    });
    // CHECK: FunctionDecl {{.*}}Half2{{.*}} 'void (sycl::half, StructNonDecomposed, int)'
  }
}
