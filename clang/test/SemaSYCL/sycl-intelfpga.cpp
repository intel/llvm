// RUN: %clang_cc1 -fsycl-is-device -fintelfpga -verify=device-intelfpga -fsyntax-only %s -triple spir64_fpga
// RUN: %clang_cc1 -fsycl-is-device -verify=device -fsyntax-only -triple spir64 %s

// Tests that we do not issue errors for _Bitints of size greater than 128
// when -fintelfpga is enabled.  The backend is expected to be able to handle
// this, upto a maximum size of 4096.  When -fintelfpga is not passed,
// we continue to diagnose size greater than 128.

// device-intelfpga-error@+2 3{{signed _BitInt of bit sizes greater than 4096 not supported}}
// device-error@+1 3{{signed _BitInt of bit sizes greater than 128 not supported}}
signed _BitInt(4097) foo(signed _BitInt(4097) a, signed _BitInt(4097) b) {
  return a / b;
}
// device-error@+2 3{{signed _BitInt of bit sizes greater than 128 not supported}}
// device-intelfpga-no-diagnostic@+1
signed _BitInt(215) foo(signed _BitInt(215) a, signed _BitInt(215) b) {
  return a + b;
}
