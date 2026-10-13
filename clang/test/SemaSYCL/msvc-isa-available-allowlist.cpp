// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-windows \
// RUN:   -aux-triple x86_64-pc-windows-msvc -fsyntax-only -verify %s

// expected-no-diagnostics

// The SYCL device pass rejects reads of non-const globals from device
// code via err_sycl_restrict / KernelGlobalVariable. `std::__isa_available`
// is exempted by the isMsvcSTLGlobalVar allowlist in Sema so that MSVC STL
// headers (e.g. <bit>) that read this global from inline functions compile
// under SYCL device mode. This test verifies the allowlist permits the
// read. A separate regression test verifies the stl_wrappers shim
// survives -E round-trip.

namespace std {
extern "C" {
inline int __isa_available = 0;
}
} // namespace std

__attribute__((sycl_device)) void kern() {
  (void)std::__isa_available; // ok — allowlisted
}
