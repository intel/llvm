// RUN: %clang_cc1 -fsycl-is-device -fintelfpga -verify=device-intelfpga -fsyntax-only %s
// RUN: %clang_cc1 -fsycl-is-host -fintelfpga -verify=host-intelfpga -fsyntax-only %s
// RUN: %clang_cc1 -fsycl-is-device -verify=device -fsyntax-only %s
// RUN: %clang_cc1 -fsycl-is-host -verify=host -fsyntax-only %s

// Tests that we do not issue errors for _Bitints of size greater than 128
// when -fintelfpga is enabled.  The backend is expected to be able to handle
// this.  When -fintelfpga is not passed, we continue to diagnose.

// device-intelfpga-no-diagnostics
// host-intelfpga-no-diagnostics
// device-error@+2 3{{signed _BitInt of bit sizes greater than 128 not supported}}
// host-error@+1 3{{signed _BitInt of bit sizes greater than 128 not supported}}
signed _BitInt(211) foo(signed  _BitInt(211) a, signed _BitInt(211) b) {
  return a/b;
}
