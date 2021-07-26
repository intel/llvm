// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -fsycl-int-header=%t.h %s -fsyntax-only
// RUN: FileCheck -input-file=%t.h %s
// This test checks the integration header generated for a kernel
// with an argument that is a VLA

// CHECK: #include <CL/sycl/detail/kernel_desc.hpp>

// CHECK: class Canonical;
// CHECK: class FLA;
// CHECK: class SD;
// CHECK: class MD3;
// CHECK: class MDFLA;
// CHECK: class MDFLAVLA;

// CHECK: __SYCL_INLINE_NAMESPACE(cl) {
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: namespace detail {

// CHECK: static constexpr
// CHECK-NEXT: const char* const kernel_names[] = {
// CHECK-NEXT:   "_ZTSZ3fooPiiiE9Canonical",
// CHECK-NEXT:   "_ZTSZ3fooPiiiE3FLA",
// CHECK-NEXT:   "_ZTSZ3fooPiiiE2SD",
// CHECK-NEXT:   "_ZTSZ3fooPiiiE3MD3",
// CHECK-NEXT:   "_ZTSZ3fooPiiiE5MDFLA",
// CHECK-NEXT:   "_ZTSZ3fooPiiiE8MDFLAVLA"
// CHECK-NEXT: };

// CHECK: static constexpr
// CHECK-NEXT: const kernel_param_desc_t kernel_signatures[] = {
// CHECK-NEXT:   //--- _ZTSZ3fooPiiiE9Canonical
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 8, 0 },
// CHECK-NEXT:   { kernel_param_kind_t::kind_vla, 0x1000004, 8 },
// CHECK-EMPTY:
// CHECK-NEXT:   //--- _ZTSZ3fooPiiiE3FLA
// CHECK-EMPTY:
// CHECK-NEXT:   //--- _ZTSZ3fooPiiiE2SD
// CHECK-NEXT:   { kernel_param_kind_t::kind_pointer, 8, 0 },
// CHECK-EMPTY:
// CHECK-NEXT:   //--- _ZTSZ3fooPiiiE3MD3
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 8, 0 },
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 8, 8 },
// CHECK-NEXT:   { kernel_param_kind_t::kind_vla, 0x2000004, 16 },
// CHECK-EMPTY:
// CHECK-NEXT:   //--- _ZTSZ3fooPiiiE5MDFLA
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 8, 0 },
// CHECK-NEXT:   { kernel_param_kind_t::kind_vla, 0x1000010, 8 },
// CHECK-EMPTY:
// CHECK-NEXT:   //--- _ZTSZ3fooPiiiE8MDFLAVLA
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 8, 0 },
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 8, 8 },
// CHECK-NEXT:   { kernel_param_kind_t::kind_vla, 0x2000004, 16 },
// CHECK-EMPTY:
// CHECK-NEXT:   { kernel_param_kind_t::kind_invalid, -987654321, -987654321 },
// CHECK-NEXT: };

#include "Inputs/sycl.hpp"

using namespace cl::sycl;

template <typename name, typename Func>
__attribute__((sycl_kernel)) void a_kernel(const Func &kernelFunc) {
  kernelFunc();
}

void foo(int *i, int x, int y) {
  using intarray = int(*)[x]; // Motivating example
  intarray ia = reinterpret_cast<intarray>(i);
  a_kernel<class Canonical>([=]() {
    ia[1][2] = 9;
  });
  a_kernel<class FLA>([=]() { // Not a VLA
    int fla[4];
    fla[0] = 9;
  });
  a_kernel<class SD>([=]() {
    i[0] = 9; // single-level pointer
  });
  using intmdarray = int(*)[x][y];
  intmdarray imda = reinterpret_cast<intmdarray>(i);
  a_kernel<class MD3>([=]() {
    imda[1][2][3] = 9; // Multi-dimensional VLA
  });
  using intflavlaarray = int(*)[x][4]; // VLA then a fixed-length array
  intflavlaarray ifva = reinterpret_cast<intflavlaarray>(i);
  a_kernel<class MDFLA>([=]() {
    ifva[1][2][3] = 9;
  });
  using intvlaflaarray = int(*)[4][y]; // A fix-length array dimension followed by a VLA
  intvlaflaarray ivfa = reinterpret_cast<intvlaflaarray>(i);
  a_kernel<class MDFLAVLA>([=]() {
    ivfa[1][2][3] = 9;
  });
}
