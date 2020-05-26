// RUN: %clang_cc1 -fsycl -fsycl-is-device -fsycl-explicit-simd -fsyntax-only -verify -pedantic %s

#define ESIMD_PRIVATE __attribute__((opencl_private))
#define ESIMD_REGISTER_NUM(n) __attribute__((register_num(n)))

// no error expected
ESIMD_PRIVATE ESIMD_REGISTER_NUM(17) int privGlob;

void foo() {
  // expected-warning@+1{{'register_num' attribute only applies to global variables}}
  ESIMD_REGISTER_NUM(17)
  int privLoc;
}
