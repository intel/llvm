// RUN: not %clangxx -fsycl -fsyntax-only -Wno-unused-command-line-argument %s 2>&1 | FileCheck %s --implicit-check-not="warning:" --implicit-check-not="error:"

#include <sycl/ext/intel/experimental/esimd.hpp>

using namespace sycl::ext::intel::experimental::esimd;

// simd implicit conversion if there is only one element in the vector
void test_simd_implicit_conv() SYCL_ESIMD_FUNCTION {
  simd<int, 1> v1 = 1;
  int scalar = v1;
}

// Since simd<T, 1> implicitly converts to T, and also
// simd class can be constructed from a scalar, operations like
// simd<T, 1> + T can be ambiguous.
// This test checks that such case can be resolved.
void test_simd1_binop() SYCL_ESIMD_FUNCTION {
  simd<int, 1> v1 = 1;
  simd<int, 1> v2 = 2;
  simd<int, 1> v3 = v1 + 1;
}

void test_simd_binop() SYCL_ESIMD_FUNCTION {
  simd<int, 4> v1 = 1;
  simd<int, 1> v2 = v1[0] + v1[1];
}

// simd relop to return bool if there is only one element in the vector
// TODO: introduce implicit conversion of simd_mask<1> to bool
void test_simd1_relop() SYCL_ESIMD_FUNCTION {
  simd<int, 1> v1 = 1;
  bool res1 = v1 == 2;
}

// simd_view implicit conversion if there is only one element in the view
void test_simd_view_implicit_conv() SYCL_ESIMD_FUNCTION {
  simd<int, 4> v1 = 1;
  int val1 = v1[1];
}

// simd_view relop to return bool if there is only one element
// TODO: introduce implicit conversion of simd_mask<1> to bool
void test_simd_view_relop() SYCL_ESIMD_FUNCTION {
  simd<int, 4> v1 = 1;
  bool res1 = v1[0] == 2;
  bool res2 = v1[0] == v1[1];
}

// simd::select can accept simd_view of size 1 as Offset argument
void test_simd_select() SYCL_ESIMD_FUNCTION {
  simd<int, 4> v1 = 1;
  simd<int, 4> v2 = 2;
  v1.select<1, 1>(v2[2]) += 1;
}

// simd::replicate can accept simd_view of size 1 as Offset argument
void test_simd_replicate() SYCL_ESIMD_FUNCTION {
  simd<int, 4> v = 1;
  v.replicate_w<2, 1>(v[0]);
}

void test_simd_writable_subscript() SYCL_ESIMD_FUNCTION {
  simd<int, 4> v = 1;
  int val1 = v[0]; // simd_view -> int
  v[1] = 0;        // returns simd_view
}

void test_simd_const_subscript() SYCL_ESIMD_FUNCTION {
  const simd<int, 4> cv = 1;
  int val2 = cv[0]; // returns int instead of simd_view
  // CHECK: simd_subscript.cpp:72{{.*}}error: expression is not assignable
  cv[1] = 0;
}

void test_simd_view_assign_op() SYCL_ESIMD_FUNCTION {
  simd<int, 4> A = 1;
  simd<int, 2> B = 2;
  A.select<2, 1>(0) = A.select<2, 1>(2);
  A.select<2, 1>(0) = B;
  simd<int, 1> C = 3;
  A[0] = A[1];
  A[0] = C;
}
