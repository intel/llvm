// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -fsycl-int-header=%t.h %s -fsyntax-only
// RUN: FileCheck -input-file=%t.h %s
// This test checks the integration header generated for a kernel
// with an argument that is a VLA

#include "Inputs/sycl.hpp"

using namespace cl::sycl;

// CHECK: #include <CL/sycl/detail/kernel_desc.hpp>

class Canonical;
class FLA;
class SD;
class MD3;
class MDVLAFLA;
class MDFLAVLA;

// CHECK: class Canonical;
// CHECK: class FLA;
// CHECK: class SD;
// CHECK: class MD3;
// CHECK: class MDVLAFLA;
// CHECK: class MDFLAVLA;

// CHECK: __SYCL_INLINE_NAMESPACE(cl) {
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: namespace detail {
// CHECK: static constexpr
// CHECK-NEXT: const char* const kernel_names[] = {
// CHECK-NEXT:   "_ZTS9Canonical",
// CHECK-NEXT:   "_ZTS3FLA",
// CHECK-NEXT:   "_ZTS2SD",
// CHECK-NEXT:   "_ZTS3MD3",
// CHECK-NEXT:   "_ZTS8MDVLAFLA",
// CHECK-NEXT:   "_ZTS8MDFLAVLA"
// CHECK-NEXT: };

void foo(int *i, int x, int y) {
  sycl::queue q;

  using intarray = int(*)[x]; // Canonical example for VLA usage
  intarray ia = reinterpret_cast<intarray>(i);

  int fla[4]; // Not a VLA

  using intmdarray = int(*)[x][y]; // Multi-dimensional VLA
  intmdarray imda = reinterpret_cast<intmdarray>(i);

  using intflavlaarray = int(*)[x][4]; // VLA then a fixed-length array
  intflavlaarray ifva = reinterpret_cast<intflavlaarray>(i);

  using intvlaflaarray = int(*)[4][y]; // A fix-length array dimension followed by a VLA
  intvlaflaarray ivfa = reinterpret_cast<intvlaflaarray>(i);

// CHECK: static constexpr
// CHECK-NEXT: const kernel_param_desc_t kernel_signatures[] = {
  q.submit([&](cl::sycl::handler &h) {
    h.single_task<Canonical>(
      [=]() {
        ia[1][2] = 9;
      });
// CHECK-NEXT:   //--- _ZTS9Canonical
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 8, 0 },
// CHECK-NEXT:   { kernel_param_kind_t::kind_vla, 0x1000004, 8 },
// CHECK-EMPTY:

    h.single_task<FLA>(
      [=]() {
        int i;
        i = fla[0];
      });
// CHECK-NEXT:   //--- _ZTS3FLA
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 16, 0 },
// CHECK-EMPTY:

    h.single_task<SD>(
      [=]() {
        i[0] = 9; // single-level pointer
      });
// CHECK-NEXT:   //--- _ZTS2SD
// CHECK-NEXT:   { kernel_param_kind_t::kind_pointer, 8, 0 },
// CHECK-EMPTY:

    h.single_task<MD3>(
      [=]() {
        imda[1][2][3] = 9; // Multi-dimensional VLA
      });
// CHECK-NEXT:   //--- _ZTS3MD3
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 8, 0 },
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 8, 8 },
// CHECK-NEXT:   { kernel_param_kind_t::kind_vla, 0x2000004, 16 },
// CHECK-EMPTY:

    h.single_task<class MDVLAFLA>(
      [=]() {
        ifva[1][2][3] = 9;
      });
// CHECK-NEXT:   //--- _ZTS8MDVLAFLA
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 8, 0 },
// CHECK-NEXT:   { kernel_param_kind_t::kind_vla, 0x1000010, 8 },
// CHECK-EMPTY:

    h.single_task<class MDFLAVLA>(
      [=]() {
        ivfa[1][2][3] = 9;
      });
// CHECK-NEXT:   //--- _ZTS8MDFLAVLA
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 8, 0 },
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 8, 8 },
// CHECK-NEXT:   { kernel_param_kind_t::kind_vla, 0x2000004, 16 },
// CHECK-EMPTY:
  });
// CHECK-NEXT:   { kernel_param_kind_t::kind_invalid, -987654321, -987654321 },
// CHECK-NEXT: };
}
