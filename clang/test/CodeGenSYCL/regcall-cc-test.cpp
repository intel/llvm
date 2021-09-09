// clang-format off
// RUN:  %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -disable-llvm-passes -O0 -emit-llvm %s -o - | FileCheck %s

// This test checks SYCL device compiler code generation for the __regcall
// functions. This calling convention makes return values and function arguments
// passed as values (through virtual registers) in most cases.

// CHECK-DAG: target triple = "spir64-unknown-unknown"

// ------------------- Positive test cases (pass by value)

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

template <class T> T __regcall func(T x) { return x.foo(); }

// === TEST CASE: invoke_simd scenario, when sycl::ext::intel::esimd::simd
// objects used as return value and parameters

SYCL_DEVICE simd<float, 8> __regcall SCALE(simd<float, 8> v);
// CHECK-DAG: declare x86_regcallcc <8 x float> @_Z17__regcall3__SCALE4simdIfLi8EE(<8 x float>)

SYCL_DEVICE simd<float, 8> __regcall foo(simd<float, 8> x) {
  return SCALE(x);
// CHECK-DAG: %{{[0-9a-zA-Z_.]+}} = call x86_regcallcc <8 x float> @_Z17__regcall3__SCALE4simdIfLi8EE(<8 x float> %{{[0-9a-zA-Z_.]+}})
}

// === TEST CASE: nested struct with different types of fields

struct C {
  float x, y;
};
// CHECK-DAG: %struct.C = type { float, float }

struct PassAsByval {
  C a;
  int *b;
  raw_vector<float, 3> c;
};
// CHECK-DAG: %struct.PassAsByval = type { %struct.C, i32 addrspace(4)*, <3 x float> }

SYCL_DEVICE PassAsByval __regcall bar(PassAsByval x) {
// CHECK-DAG: define dso_local x86_regcallcc %struct.PassAsByval @_Z15__regcall3__bar11PassAsByval(%struct.C %{{[0-9a-zA-Z_.]+}}, i32 addrspace(4)* %{{[0-9a-zA-Z_.]+}}, <3 x float> %{{[0-9a-zA-Z_.]+}})
  x.a.x += 1;
  return x;
}

// === TEST CASE: multi-level nested structs with single primitive type element at the bottom

struct A1 { char x; };
struct B1 { A1 a; };
struct C1 {
  B1 b;
  C1 foo() { return *this; }
};
// CHECK-DAG: %struct.C1 = type { %struct.B1 }
// CHECK-DAG: %struct.B1 = type { %struct.A1 }
// CHECK-DAG: %struct.A1 = type { i8 }

template SYCL_DEVICE C1 __regcall func<C1>(C1 x);
// CHECK-DAG: define weak_odr x86_regcallcc i8 @_Z16__regcall3__funcI2C1ET_S1_(i8 %{{[0-9a-zA-Z_.]+}})

// === TEST CASE: multi-level nested structs with multiple elements at all levels

struct A2 { char x; };
struct B2 { A2 a; int* ptr; };
struct C2 {
  B2 b;
  double c;

  C2 foo() { return *this; }
};

// CHECK-DAG: %struct.C2 = type { %struct.B2, double }
// CHECK-DAG: %struct.B2 = type { %struct.A2, i32 addrspace(4)* }
// CHECK-DAG: %struct.A2 = type { i8 }

template SYCL_DEVICE C2 __regcall func<C2>(C2 x);
// CHECK-DAG: define weak_odr x86_regcallcc %struct.C2 @_Z16__regcall3__funcI2C2ET_S1_(%struct.B2 %{{[0-9a-zA-Z_.]+}}, double %{{[0-9a-zA-Z_.]+}})

// === TEST CASE: multi-level nested structs with one primitive type element at
// the bottom, and one - at the top. The nested struct at the top is expected to
// get "unwraped" by the compiler evaporating to the single element at the
// bottom.

struct A3 { char x; };
struct B3 { A3 a; }; // unwrapped
struct C3 { // unwrapped
  B3 b;
  char c;

  C3 foo() { return *this; }
};

// CHECK-DAG: %struct.C3 = type { %struct.B3, i8 }
// CHECK-DAG: %struct.B3 = type { %struct.A3 }
// CHECK-DAG: %struct.A3 = type { i8 }

template SYCL_DEVICE C3 __regcall func<C3>(C3 x);
// CHECK-DAG: define weak_odr x86_regcallcc i16 @_Z16__regcall3__funcI2C3ET_S1_(i16 %{{[0-9a-zA-Z_.]+}})

// === TEST CASE: multi-level nested structs with one primitive type element at
// the bottom, and one - at the top. The nested struct at the top is expected to
// get "unwraped" by the compiler evaporating to the single element at the
// bottom.

struct A4 { char x; };
struct B4 { A4 a; };
struct C4 {
  B4 b;
  int *ptr;

  C4 foo() { return *this; }
};

// CHECK-DAG: %struct.C4 = type { %struct.B4, i32 addrspace(4)* }
// CHECK-DAG: %struct.B4 = type { %struct.A4 }
// CHECK-DAG: %struct.A4 = type { i8 }

template SYCL_DEVICE C4 __regcall func<C4>(C4 x);
// CHECK-DAG: define weak_odr x86_regcallcc %struct.C4 @_Z16__regcall3__funcI2C4ET_S1_(%struct.B4 %{{[0-9a-zA-Z_.]+}}, i32 addrspace(4)* %{{[0-9a-zA-Z_.]+}})

// === TEST CASE: multi-level nested structs with only leaf fields of primitive
// types. Unwrapping and merging should yield 2 32-bit integers

struct A5a { char x; char y; };
struct A5b { char x; char y; };
struct B5 { A5a a; A5b b; };
struct C5 {
  B5 b1;
  B5 b2;

  C5 foo() { return *this; }
};

// CHECK-DAG: %struct.C5 = type { %struct.B5, %struct.B5 }
// CHECK-DAG: %struct.B5 = type { %struct.A5a, %struct.A5b }
// CHECK-DAG: %struct.A5a = type { i8, i8 }
// CHECK-DAG: %struct.A5b = type { i8, i8 }

template SYCL_DEVICE C5 __regcall func<C5>(C5 x);
// CHECK-DAG: define weak_odr x86_regcallcc [2 x i32] @_Z16__regcall3__funcI2C5ET_S1_([2 x i32] %{{[0-9a-zA-Z_.]+}})

// === TEST CASE: multi-level nested structs with only leaf fields of primitive
// types. Unwrapping and merging should yield 2 32-bit integers

struct B6 { int *a; int b; };
struct C6 {
  B6 b;
  char x;
  char y;

  C6 foo() { return *this; }
};

// CHECK-DAG: %struct.C6 = type { %struct.B6, i8, i8 }
// CHECK-DAG: %struct.B6 = type { i32 addrspace(4)*, i32 }

template SYCL_DEVICE C6 __regcall func<C6>(C6 x);
// CHECK-DAG: define weak_odr x86_regcallcc %struct.C6 @_Z16__regcall3__funcI2C6ET_S1_(%struct.B6 %{{[0-9a-zA-Z_.]+}}, i8 %{{[0-9a-zA-Z_.]+}}, i8 %{{[0-9a-zA-Z_.]+}})

// === TEST CASE: a struct with sizeof() <= 2 is passed as a single 16-bit integer

struct CharChar {
  char a;
  char b;

  CharChar foo() { return *this; }
};
// CHECK-DAG: %struct.CharChar = type { i8, i8 }

template SYCL_DEVICE CharChar __regcall func<CharChar>(CharChar x);
// CHECK-DAG: define weak_odr x86_regcallcc i16 @_Z16__regcall3__funcI8CharCharET_S1_(i16 %{{[0-9a-zA-Z_.]+}})

// === TEST CASE: a struct with sizeof() == 3-4 is passed as single 32-bit integer

struct ShortShort {
  short a;
  short b;

  ShortShort foo() { return *this; }
};
// CHECK-DAG: %struct.ShortShort = type { i16, i16 }

template SYCL_DEVICE ShortShort __regcall func<ShortShort>(ShortShort x);
// CHECK-DAG: define weak_odr x86_regcallcc i32 @_Z16__regcall3__funcI10ShortShortET_S1_(i32 %{{[0-9a-zA-Z_.]+}})

struct CharShort {
  char a;
  short b;

  CharShort foo() { return *this; }
};
// CHECK-DAG: %struct.CharShort = type { i8, i16 }

template SYCL_DEVICE CharShort __regcall func<CharShort>(CharShort x);
// CHECK-DAG: define weak_odr x86_regcallcc i32 @_Z16__regcall3__funcI9CharShortET_S1_(i32 %{{[0-9a-zA-Z_.]+}})

// === TEST CASE: a struct with primitive single field element is just unwrapped

struct Char {
  char a;

  Char foo() { return *this; }
};
// CHECK-DAG: %struct.Char = type { i8 }

template SYCL_DEVICE Char __regcall func<Char>(Char x);
// CHECK-DAG: define weak_odr x86_regcallcc i8 @_Z16__regcall3__funcI4CharET_S1_(i8 %{{[0-9a-zA-Z_.]+}})

struct Float {
  float a;

  Float foo() { return *this; }
};
// CHECK-DAG: %struct.Float = type { float }

template SYCL_DEVICE Float __regcall func<Float>(Float x);
// CHECK-DAG: define weak_odr x86_regcallcc float @_Z16__regcall3__funcI5FloatET_S1_(float %{{[0-9a-zA-Z_.]+}})

// === TEST CASE: a struct with sizeof() == 5-8 is passed as two 32-bit integers
// 32-bit integers

struct CharCharShortFloat {
  char a, b;
  short c;
  float d;

  CharCharShortFloat foo() { return *this; }
};
// CHECK-DAG: %struct.CharCharShortFloat = type { i8, i8, i16, float }

template SYCL_DEVICE CharCharShortFloat __regcall func<CharCharShortFloat>(CharCharShortFloat x);
// CHECK-DAG: define weak_odr x86_regcallcc [2 x i32] @_Z16__regcall3__funcI18CharCharShortFloatET_S1_([2 x i32] %{{[0-9a-zA-Z_.]+}})

// === TEST CASE: a struct with some of the fields padded and sizeof() > 8
//   * when passed as argument, it is broken into constituents
//   * is returned by value

struct CharFloatCharShort {
  char a;
  float b;
  char c;
  short d;

  CharFloatCharShort foo() { return *this; }
};

// CHECK-DAG: %struct.CharFloatCharShort = type { i8, float, i8, i16 }

template SYCL_DEVICE CharFloatCharShort __regcall func<CharFloatCharShort>(CharFloatCharShort x);
// CHECK-DAG: define weak_odr x86_regcallcc %struct.CharFloatCharShort @_Z16__regcall3__funcI18CharFloatCharShortET_S1_(i8 %{{[0-9a-zA-Z_.]+}}, float %{{[0-9a-zA-Z_.]+}}, i8 %{{[0-9a-zA-Z_.]+}}, i16 %{{[0-9a-zA-Z_.]+}})

struct CharDoubleCharLonglong {
  char a;
  double b;
  char c;
  long long d;

  CharDoubleCharLonglong foo() { return *this; }
};

// CHECK-DAG: %struct.CharDoubleCharLonglong = type { i8, double, i8, i64 }

template SYCL_DEVICE CharDoubleCharLonglong __regcall func<CharDoubleCharLonglong>(CharDoubleCharLonglong x);
// CHECK-DAG: define weak_odr x86_regcallcc %struct.CharDoubleCharLonglong @_Z16__regcall3__funcI22CharDoubleCharLonglongET_S1_(i8 %{{[0-9a-zA-Z_.]+}}, double %{{[0-9a-zA-Z_.]+}}, i8 %{{[0-9a-zA-Z_.]+}}, i64 %{{[0-9a-zA-Z_.]+}})


// === TEST CASE: a struct of 130x4-byte elements is still passed by value

struct StillPassThroughRegisters {
  // 130 total:
  int a, a01, a02, a03, a04, a05, a06, a07, a08, a09,
    a10, a11, a12, a13, a14, a15, a16, a17, a18, a19,
    a20, a21, a22, a23, a24, a25, a26, a27, a28, a29,
    a30, a31, a32, a33, a34, a35, a36, a37, a38, a39,
    a40, a41, a42, a43, a44, a45, a46, a47, a48, a49,
    a50, a51, a52, a53, a54, a55, a56, a57, a58, a59,
    a60, a61, a62, a63, a64, a65, a66, a67, a68, a69,
    a70, a71, a72, a73, a74, a75, a76, a77, a78, a79,
    a80, a81, a82, a83, a84, a85, a86, a87, a88, a89,
    a90, a91, a92, a93, a94, a95, a96, a97, a98, a99,
    aa0, aa1, aa2, aa3, aa4, aa5, aa6, aa7, aa8, aa9,
    ab0, ab1, ab2, ab3, ab4, ab5, ab6, ab7, ab8, ab9,
    ac0, ac1, ac2, ac3, ac4, ac5, ac6, ac7, ac8, ac9;

  StillPassThroughRegisters foo() { return *this; }
};
// CHECK-DAG: %struct.StillPassThroughRegisters = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }


template SYCL_DEVICE StillPassThroughRegisters __regcall func<StillPassThroughRegisters>(StillPassThroughRegisters x);
// CHECK-DAG: define weak_odr x86_regcallcc %struct.StillPassThroughRegisters @_Z16__regcall3__funcI25StillPassThroughRegistersET_S1_(i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}})

// === TEST CASE: class with multiple inheritance is passed by value

class Class0 { int x0; };
class Class1 { int x1; };
class ClassX : public Class0, public Class1 { int x; };
class ClassY { int y; };
class ClassXY : public ClassX, public ClassY {
  int xy;
public:
  ClassXY foo() { return *this; }
};
// CHECK-DAG: %class.ClassXY = type { %class.ClassX, %class.ClassY, i32 }
// CHECK-DAG: %class.ClassX = type { %class.Class0, %class.Class1, i32 }
// CHECK-DAG: %class.Class0 = type { i32 }
// CHECK-DAG: %class.Class1 = type { i32 }
// CHECK-DAG: %class.ClassY = type { i32 }

template SYCL_DEVICE ClassXY __regcall func<ClassXY>(ClassXY x);
// CHECK-DAG: define weak_odr x86_regcallcc %class.ClassXY @_Z16__regcall3__funcI7ClassXYET_S1_(%class.ClassX %{{[0-9a-zA-Z_.]+}}, %class.ClassY %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}})

// ------------------- Negative test cases (pass via memory)

// === TEST CASE: no copy constructor -> pass by pointer
struct NonCopyable {
  NonCopyable(int a) : a(a) {}
  NonCopyable(const NonCopyable&) = delete;
  int a;
};
// CHECK-DAG: %struct.NonCopyable = type { i32 }

SYCL_DEVICE int __regcall bar(NonCopyable x) {
// CHECK-DAG: define dso_local x86_regcallcc noundef i32 @_Z15__regcall3__bar11NonCopyable(%struct.NonCopyable* noundef %x)
  return x.a;
}

// === TEST CASE: empty struct -> optimize out
struct Empty {};
// CHECK-DAG: %struct.Empty = type

SYCL_DEVICE int __regcall bar(Empty x) {
// CHECK-DAG: define dso_local x86_regcallcc noundef i32 @_Z15__regcall3__bar5Empty()
  return 10;
}

// === TEST CASE: struct ends with flexible array -> pass by pointer
struct EndsWithFlexArray {
  int a;
  int x[];
};
// CHECK-DAG: %struct.EndsWithFlexArray = type { i32, [0 x i32] } 

SYCL_DEVICE int __regcall bar(EndsWithFlexArray x) {
// CHECK-DAG: define dso_local x86_regcallcc noundef i32 @_Z15__regcall3__bar17EndsWithFlexArray(%struct.EndsWithFlexArray* noundef byval(%struct.EndsWithFlexArray) align 4 %x)
  return x.a;
}
