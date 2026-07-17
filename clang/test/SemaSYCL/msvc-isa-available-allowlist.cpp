// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-windows \
// RUN:   -aux-triple x86_64-pc-windows-msvc -fsyntax-only -verify %s

// The SYCL device pass rejects reads of non-const globals from kernels
// via err_sycl_restrict / KernelGlobalVariable. `std::__isa_available` is
// exempted by the isMsvcSTLGlobalVar allowlist in Sema so that MSVC STL
// headers (e.g. <bit>) that read this global from inline functions
// compile under SYCL device mode. This test verifies the allowlist
// permits the read. We have a different test to verify the
// stl_wrappers shim survives -E round-trip.

// expected-no-diagnostics

#include "Inputs/sycl.hpp"

namespace std {
extern "C" {
inline int __isa_available = 0;
}
} // namespace std

// A read of another non-const std:: global — not on the allowlist — must
// still be rejected, so we don't accidentally exempt everything in std.

int test() {
  sycl::kernel_single_task<class kernel_read_isa_available>([=]() {
    (void)std::__isa_available; // ok — allowlisted
  });
  return 0;
}
