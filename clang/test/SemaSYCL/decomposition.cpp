// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -ast-dump -sycl-std=2020 %s | FileCheck %s

// This test checks that the compiler decomposes structs containing special types only
// (i.e. accessor/stream/sampler etc) and all others are passed without decomposition
// thus optimizing the number of kernel arguments.

#include "sycl.hpp"

sycl::queue myQueue;

struct StructWithAccessor {
  sycl::accessor<char, 1, sycl::access::mode::read> acc;
  int *ptr;
};

struct StructInheritedAccessor : sycl::accessor<char, 1, sycl::access::mode::read> {
  int i;
};

struct StructWithSampler {
  sycl::sampler sampl;
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

struct StructWithPtr {
  StructNonDecomposed member;
  int *ptr;
  int *ptrArr[2];
  int i;
};

struct Nested {
typedef StructWithPtr TDStrWithPTR;
};

struct NonTrivialType {
  int *Ptr;
  int i;
  NonTrivialType(int i){}
};

struct NonTrivialDerived : NonTrivialType {
  int a;
  NonTrivialDerived(int i) : NonTrivialType(i) {}
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
  // CHECK: FunctionDecl {{.*}}NonDecomposed{{.*}} 'void (StructNonDecomposed, __wrapper_class, StructWithNonDecomposedStruct, __wrapper_class) __attribute__((device_kernel))'

  {
    StructWithArray<StructWithAccessor> t1;
    myQueue.submit([&](sycl::handler &h) {
      h.single_task<class Acc1>([=]() { return t1.i; });
    });
    // CHECK: FunctionDecl {{.*}}Acc1{{.*}} 'void (__global char *, sycl::range<1>, sycl::range<1>, sycl::id<1>, __wrapper_class, __global char *, sycl::range<1>, sycl::range<1>, sycl::id<1>, __wrapper_class, __global char *, sycl::range<1>, sycl::range<1>, sycl::id<1>, __wrapper_class, StructNonDecomposed, int) __attribute__((device_kernel))'

    DerivedStruct<StructWithAccessor> t2;
    myQueue.submit([&](sycl::handler &h) {
      h.single_task<class Acc2>([=]() { return t2.i; });
    });
    // CHECK: FunctionDecl {{.*}}Acc2{{.*}} 'void (__global char *, sycl::range<1>, sycl::range<1>, sycl::id<1>, __wrapper_class, StructNonDecomposed, int) __attribute__((device_kernel))'

    StructWithArray<StructInheritedAccessor> t3;
    myQueue.submit([&](sycl::handler &h) {
      h.single_task<class Acc3>([=]() { return t3.i; });
    });
    // CHECK: FunctionDecl {{.*}}Acc3{{.*}} 'void (__global char *, sycl::range<1>, sycl::range<1>, sycl::id<1>, int, __global char *, sycl::range<1>, sycl::range<1>, sycl::id<1>, int, __global char *, sycl::range<1>, sycl::range<1>, sycl::id<1>, int, StructNonDecomposed, int) __attribute__((device_kernel))'

    DerivedStruct<StructInheritedAccessor> t4;
    myQueue.submit([&](sycl::handler &h) {
      h.single_task<class Acc4>([=]() { return t4.i; });
    });
    // CHECK: FunctionDecl {{.*}}Acc4{{.*}} 'void (__global char *, sycl::range<1>, sycl::range<1>, sycl::id<1>, int, StructNonDecomposed, int) __attribute__((device_kernel))'
  }

  {
    StructWithArray<StructWithSampler> t1;
    myQueue.submit([&](sycl::handler &h) {
      h.single_task<class Sampl1>([=]() { return t1.i; });
    });
    // CHECK: FunctionDecl {{.*}}Sampl1{{.*}} 'void (sampler_t, sampler_t, sampler_t, StructNonDecomposed, int) __attribute__((device_kernel))'

    DerivedStruct<StructWithSampler> t2;
    myQueue.submit([&](sycl::handler &h) {
      h.single_task<class Sampl2>([=]() { return t2.i; });
    });
    // CHECK: FunctionDecl {{.*}}Sampl2{{.*}} 'void (sampler_t, StructNonDecomposed, int) __attribute__((device_kernel))'
  }

  {
    StructWithArray<StructWithStream> t1;
    myQueue.submit([&](sycl::handler &h) {
      h.single_task<class Stream1>([=]() { return t1.i; });
    });
    // CHECK: FunctionDecl {{.*}}Stream1{{.*}} 'void (__global char *, sycl::range<1>, sycl::range<1>, sycl::id<1>, int, __global char *, sycl::range<1>, sycl::range<1>, sycl::id<1>, int, __global char *, sycl::range<1>, sycl::range<1>, sycl::id<1>, int, StructNonDecomposed, int) __attribute__((device_kernel))'
    DerivedStruct<StructWithStream> t2;
    myQueue.submit([&](sycl::handler &h) {
      h.single_task<class Stream2>([=]() { return t2.i; });
    });
    // CHECK: FunctionDecl {{.*}}Stream2{{.*}} 'void (__global char *, sycl::range<1>, sycl::range<1>, sycl::id<1>, int, StructNonDecomposed, int) __attribute__((device_kernel))'
  }

  {
    StructWithArray<StructWithHalf> t1;
    myQueue.submit([&](sycl::handler &h) {
      h.single_task<class Half1>([=]() { return t1.i; });
    });
    // CHECK: FunctionDecl {{.*}}Half1{{.*}} 'void (StructWithArray<StructWithHalf>) __attribute__((device_kernel))'

    DerivedStruct<StructWithHalf> t2;
    myQueue.submit([&](sycl::handler &h) {
      h.single_task<class Half2>([=]() { return t2.i; });
    });
    // CHECK: FunctionDecl {{.*}}Half2{{.*}} 'void (DerivedStruct<StructWithHalf>) __attribute__((device_kernel))'
  }

  {
    StructWithPtr SimpleStructWithPtr;
    myQueue.submit([&](sycl::handler &h) {
      h.single_task<class Pointer>([=]() { return SimpleStructWithPtr.i; });
    });
    // CHECK: FunctionDecl {{.*}}Pointer{{.*}} 'void (__generated_StructWithPtr) __attribute__((device_kernel))'

    Nested::TDStrWithPTR TDStructWithPtr;
    myQueue.submit([&](sycl::handler &h) {
      h.single_task<class TDStr>([=]() { return TDStructWithPtr.i; });
    });
    // CHECK: FunctionDecl {{.*}}TDStr{{.*}} 'void (__generated_StructWithPtr) __attribute__((device_kernel))'

    StructWithArray<StructWithPtr> t1;
    myQueue.submit([&](sycl::handler &h) {
      h.single_task<class NestedArrayOfStructWithPointer>([=]() { return t1.i; });
    });
    // CHECK: FunctionDecl {{.*}}NestedArrayOfStructWithPointer{{.*}} 'void (__generated_StructWithArray) __attribute__((device_kernel))'

    DerivedStruct<StructWithPtr> t2;
    myQueue.submit([&](sycl::handler &h) {
      h.single_task<class PointerInBase>([=]() { return t2.i; });
    });
    // CHECK: FunctionDecl {{.*}}PointerInBase{{.*}} 'void (__generated_DerivedStruct) __attribute__((device_kernel))'
  }

  {
    NonTrivialType NonTrivialStructWithPtr(10);
    myQueue.submit([&](sycl::handler &h) {
      h.single_task<class NonTrivial>([=]() { return NonTrivialStructWithPtr.i;});
    });
    // CHECK: FunctionDecl {{.*}}NonTrivial{{.*}} 'void (__generated_NonTrivialType) __attribute__((device_kernel))'

    NonTrivialType NonTrivialTypeArray[2]{0,0};
    myQueue.submit([&](sycl::handler &h) {
      h.single_task<class ArrayOfNonTrivialStruct>([=]() { return NonTrivialTypeArray[0].i;});
    });
    // CHECK: FunctionDecl {{.*}}ArrayOfNonTrivialStruct{{.*}} 'void (__wrapper_class) __attribute__((device_kernel))'

    NonTrivialDerived NonTrivialDerivedStructWithPtr(10);
    myQueue.submit([&](sycl::handler &h) {
      h.single_task<class NonTrivialStructInBase>([=]() { return NonTrivialDerivedStructWithPtr.i;});
    });
    // CHECK: FunctionDecl {{.*}}NonTrivialStructInBase{{.*}} 'void (__generated_NonTrivialDerived) __attribute__((device_kernel))'
  }
}
