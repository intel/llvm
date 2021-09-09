// RUN:  %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -disable-llvm-passes -O0 -emit-llvm %s -o - | FileCheck %s

// This test checks SYCL device compiler code generation for the __regcall
// functions. This calling convention makes return values and function arguments
// passed as values (through virtual registers) in most cases.

// CHECK-DAG: target triple = "spir64-unknown-unknown"

template <class T, int N> using raw_vector =
    T __attribute__((ext_vector_type(N)));

template <class T, int N>
struct simd {
  raw_vector<T, N> val;
};

#ifdef __SYCL_DEVICE_ONLY__
#define SYCL_DEVICE __attribute__((sycl_device))
#else
#define SYCL_DEVICE
#endif

SYCL_DEVICE simd<float, 8> __regcall SCALE(simd<float, 8> v);
// CHECK-DAG: declare x86_regcallcc <8 x float> @_Z17__regcall3__SCALE4simdIfLi8EE(<8 x float>)

SYCL_DEVICE simd<float, 8> __regcall foo(simd<float, 8> x) {
  return SCALE(x);
  // CHECK-DAG: %{{[0-9a-zA-Z_.]+}} = call x86_regcallcc <8 x float> @_Z17__regcall3__SCALE4simdIfLi8EE(<8 x float> %{{[0-9a-zA-Z_.]+}})
}

struct C {
  float x, y;
};
// CHECK-DAG: %struct._ZTS1C.C = type { float, float }

struct PassAsByval {
  C a;
  int *b;
  raw_vector<float, 3> c;
};
// CHECK-DAG: %struct._ZTS11PassAsByval.PassAsByval = type { %struct._ZTS1C.C, i32 addrspace(4)*, <3 x float> }

SYCL_DEVICE PassAsByval __regcall bar(PassAsByval x) {
  // CHECK-DAG: dso_local x86_regcallcc %struct._ZTS11PassAsByval.PassAsByval @_Z15__regcall3__bar11PassAsByval(
  //   CHECK-DAG-NEXT: %struct._ZTS11PassAsByval.PassAsByval addrspace(4)* noalias sret
  //     CHECK-DAG-NEXT: (%struct._ZTS11PassAsByval.PassAsByval) align {{[0-9]+}} %{{[0-9a-zA-Z_.]+}},
  //   CHECK-DAG-NEXT: %struct._ZTS11PassAsByval.PassAsByval* byval
  //     CHECK-DAG-NEXT: (%struct._ZTS11PassAsByval.PassAsByval) align {{[0-9]+}} %{{[0-9a-zA-Z_.]+}})
  x.a.x += 1;
  return x;
}

struct PassAsSingle32bitInt {
  char a, b, c, d;
  PassAsSingle32bitInt operator++(int) {
    auto sav = *this;
    a++;
    return sav;
  }
};
// CHECK-DAG: %struct._ZTS20PassAsSingle32bitInt.PassAsSingle32bitInt = type { i8, i8, i8, i8 }

struct PassAsTwo32bitInts {
  char a, b, c, d, e, f, g, h;
  PassAsTwo32bitInts operator++(int) {
    auto sav = *this;
    a++;
    return sav;
  }
};
// CHECK-DAG: %struct._ZTS18PassAsTwo32bitInts.PassAsTwo32bitInts = type { i8, i8, i8, i8, i8, i8, i8, i8 }

struct StillPassThroughRegisters {
  int a, a01, a02, a03, a04, a05, a06, a07,
      a08, a09, a10, a11, a12, a13, a14, a15,
      a16, a17, a18, a19, a20, a21, a22, a23;

  StillPassThroughRegisters operator++(int) {
    auto sav = *this;
    a++;
    return sav;
  }
};
// CHECK-DAG: %struct._ZTS25StillPassThroughRegisters.StillPassThroughRegisters = type {
//   CHECK-DAG-NEXT: i32, i32, i32, i32, i32, i32, i32, i32,
//   CHECK-DAG-NEXT: i32, i32, i32, i32, i32, i32, i32, i32,
//   CHECK-DAG-NEXT: i32, i32, i32, i32, i32, i32, i32, i32 }

template <class T> T __regcall func(T x) {
  return x++;
}

template SYCL_DEVICE PassAsSingle32bitInt __regcall func<PassAsSingle32bitInt>(PassAsSingle32bitInt x);
// CHECK-DAG: define weak_odr x86_regcallcc i32 @_Z16__regcall3__funcI20PassAsSingle32bitIntET_S1_(i32 %{{[0-9a-zA-Z_.]+}})
template SYCL_DEVICE PassAsTwo32bitInts __regcall func<PassAsTwo32bitInts>(PassAsTwo32bitInts x);
// CHECK-DAG: define weak_odr x86_regcallcc [2 x i32] @_Z16__regcall3__funcI18PassAsTwo32bitIntsET_S1_([2 x i32] %{{[0-9a-zA-Z_.]+}})

template SYCL_DEVICE StillPassThroughRegisters __regcall func<StillPassThroughRegisters>(StillPassThroughRegisters x);
// CHECK-DAG: define weak_odr x86_regcallcc %struct._ZTS25StillPassThroughRegisters.StillPassThroughRegisters
//   CHECK-DAG-NEXT: @_Z16__regcall3__funcI25StillPassThroughRegistersET_S1_(
//     CHECK-DAG-NEXT: i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}},
//     CHECK-DAG-NEXT: i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}},
//     CHECK-DAG-NEXT: i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}},
//     CHECK-DAG-NEXT: i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}},
//     CHECK-DAG-NEXT: i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}},
//     CHECK-DAG-NEXT: i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}})
