// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -fsycl-int-header=%t.h %s -fsyntax-only
// RUN: FileCheck -input-file=%t.h %s
// This test validates that we don't generate an empty 'kernel_signatures' in
// the case where there are no kernel fields. This is to avoid a warning in the
// integration header.

// CHECK: static constexpr
// CHECK-NEXT: const char* const kernel_names[] = {
// CHECK-NEXT:   "_ZTSZ4mainE1K",
// CHECK-NEXT:   ""
// CHECK-NEXT: };

// CHECK: static constexpr
// CHECK-NEXT: const kernel_param_desc_t kernel_signatures[] = {
// CHECK-NEXT: //--- _ZTSZ4mainE1K
// CHECK-EMPTY:
// CHECK-NEXT:   { kernel_param_kind_t::kind_invalid, -987654321, -987654321 },
// CHECK-NEXT: };

#include "Inputs/sycl.hpp"

using namespace sycl;

int main() {
  // captureless kernel lambda results in no fields.
  kernel_single_task<class K>([]{});
}
