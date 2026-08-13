// Does a run with a SYCL device, where neon polyvector type errors are ignored.
// afterwards compile for the host where neon polyvector type errors aren't ignored

// polyvector type errors get ignored with SYCL enabled
// RUN: %clang_cc1 %s -fsycl-is-device -triple spir64 -aux-triple arm64-unknown-linux-gnu -target-feature +neon -fsyntax-only -verify=quiet

// diagnostic for bad_poly32_t
// RUN: %clang_cc1 %s -triple arm64-unknown-linux-gnu -target-feature +neon -fsyntax-only -verify
typedef unsigned char poly8_t;
typedef unsigned short poly16_t;
typedef unsigned long poly64_t;

typedef unsigned int bad_poly32_t;

typedef __attribute__((neon_polyvector_type(16))) poly8_t poly8x16_t;
typedef __attribute__((neon_polyvector_type(8))) poly16_t poly16x8_t;
typedef __attribute__((neon_polyvector_type(2))) poly64_t poly64x2_t;

// this error will get ignored when running SYCL
// quiet-no-diagnostics
typedef __attribute__((neon_polyvector_type(2))) bad_poly32_t bad_poly32x2_t;
// expected-error@-1{{invalid vector element type}}
