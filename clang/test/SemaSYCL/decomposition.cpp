// RUN: %clang_cc1 -fsycl -fsycl-is-device -ast-dump %s | FileCheck %s

#include "Inputs/sycl.hpp"

using namespace cl::sycl;

struct has_acc {
  accessor<char, 1, access::mode::read> acc;
};

struct acc_base : accessor<char, 1, access::mode::read> {
  int i;
};

struct has_sampler {
  sampler sampl;
};

struct has_spec_const {
  ONEAPI::experimental::spec_constant<int, class f1> SC;
};

handler H;

struct has_stream {
  stream s1{0, 0, H};
};

struct has_half {
  half h;
};

struct non_decomposed {
  int i;
  float f;
  double d;
};

struct use_non_decomposed : non_decomposed {
  non_decomposed member;
  float f;
  double d;
};

template <typename T>
struct Test1 {
  T a;
  T b[2];
  non_decomposed d;
  int i;
};

template <typename T>
struct Test2 : T {
  non_decomposed d;
  int i;
};

template <typename Name, typename Func>
__attribute__((sycl_kernel)) void kernel(const Func &kernelFunc) {
  kernelFunc();
}

int main() {

  non_decomposed d;
  non_decomposed ds[5];
  use_non_decomposed d2;
  use_non_decomposed d2s[5];
  // Check to ensure that these are not decomposed.
  kernel<class NonDecomp>([=]() { return d.i + ds[0].i + d2.i + d2s[0].i; });
  // CHECK: FunctionDecl {{.*}}NonDecomp{{.*}} 'void (non_decomposed, __wrapper_class, use_non_decomposed, __wrapper_class)'

  {
    Test1<has_acc> t1;
    kernel<class Acc1>([=]() { return t1.i; });
    // CHECK: FunctionDecl {{.*}}Acc1{{.*}} 'void (__global char *, cl::sycl::range<1>, cl::sycl::range<1>, cl::sycl::id<1>, __global char *, cl::sycl::range<1>, cl::sycl::range<1>, cl::sycl::id<1>, __global char *, cl::sycl::range<1>, cl::sycl::range<1>, cl::sycl::id<1>, non_decomposed, int)'
    Test2<has_acc> t2;
    kernel<class Acc2>([=]() { return t2.i; });
    // CHECK: FunctionDecl {{.*}}Acc2{{.*}} 'void (__global char *, cl::sycl::range<1>, cl::sycl::range<1>, cl::sycl::id<1>, non_decomposed, int)'
    Test1<acc_base> t3;
    kernel<class Acc3>([=]() { return t3.i; });
    // CHECK: FunctionDecl {{.*}}Acc3{{.*}} 'void (__global char *, cl::sycl::range<1>, cl::sycl::range<1>, cl::sycl::id<1>, int, __global char *, cl::sycl::range<1>, cl::sycl::range<1>, cl::sycl::id<1>, int, __global char *, cl::sycl::range<1>, cl::sycl::range<1>, cl::sycl::id<1>, int, non_decomposed, int)'
    Test2<acc_base> t4;
    kernel<class Acc4>([=]() { return t4.i; });
    // CHECK: FunctionDecl {{.*}}Acc4{{.*}} 'void (__global char *, cl::sycl::range<1>, cl::sycl::range<1>, cl::sycl::id<1>, int, non_decomposed, int)'
  }

  {
    Test1<has_sampler> t1;
    kernel<class Sampl1>([=]() { return t1.i; });
    // CHECK: FunctionDecl {{.*}}Sampl1{{.*}} 'void (sampler_t, sampler_t, sampler_t, non_decomposed, int)'
    Test2<has_sampler> t2;
    kernel<class Sampl2>([=]() { return t2.i; });
    // CHECK: FunctionDecl {{.*}}Sampl2{{.*}} 'void (sampler_t, non_decomposed, int)'
  }

  {
    Test1<has_spec_const> t1;
    kernel<class SpecConst1>([=]() { return t1.i; });
    // CHECK: FunctionDecl {{.*}}SpecConst{{.*}} 'void (non_decomposed, int)'
    Test2<has_spec_const> t2;
    kernel<class SpecConst2>([=]() { return t2.i; });
    // CHECK: FunctionDecl {{.*}}SpecConst2{{.*}} 'void (non_decomposed, int)'
  }

  {
    Test1<has_stream> t1;
    kernel<class Stream1>([=]() { return t1.i; });
    // CHECK: FunctionDecl {{.*}}Stream1{{.*}} 'void (cl::sycl::stream, __global int *, cl::sycl::range<1>, cl::sycl::range<1>, cl::sycl::id<1>, cl::sycl::stream, __global int *, cl::sycl::range<1>, cl::sycl::range<1>, cl::sycl::id<1>, cl::sycl::stream, __global int *, cl::sycl::range<1>, cl::sycl::range<1>, cl::sycl::id<1>, non_decomposed, int)'
    Test2<has_stream> t2;
    kernel<class Stream2>([=]() { return t2.i; });
    // CHECK: FunctionDecl {{.*}}Stream2{{.*}} 'void (cl::sycl::stream, __global int *, cl::sycl::range<1>, cl::sycl::range<1>, cl::sycl::id<1>, non_decomposed, int)'
  }

  {
    Test1<has_half> t1;
    kernel<class Half1>([=]() { return t1.i; });
    // CHECK: FunctionDecl {{.*}}Half1{{.*}} 'void (cl::sycl::half, cl::sycl::half, cl::sycl::half, non_decomposed, int)'
    Test2<has_half> t2;
    kernel<class Half2>([=]() { return t2.i; });
    // CHECK: FunctionDecl {{.*}}Half2{{.*}} 'void (cl::sycl::half, non_decomposed, int)'
  }
}
