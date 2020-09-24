// RUN: %clangxx -fsycl -fsycl-explicit-simd -fsycl-device-only -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics

#include <CL/sycl/INTEL/esimd.hpp>

// This test checks that DPC++ compiler in ESIMD mode understands
// the ESIMD_PRIVATE and ESIMD_REGISTER macros

ESIMD_PRIVATE ESIMD_REGISTER(17) int vc;

SYCL_EXTERNAL void init_vc(int x) {
  vc = x;
}
