// Both of these should fail on bad_poly32_t, yielding 32-bit types
// RUN: %clang_cc1 %s -fsycl-is-device -triple spir64 -aux-triple aarch64-pc-windows-msvc -target-feature +neon -DLONG -fsyntax-only -verify
// RUN: %clang_cc1 %s -fsycl-is-device -triple spir64 -aux-triple arm64-unknown-linux-gnu -target-feature +neon -ULONG -fsyntax-only -verify
// This will succeed on linux as long is 64-bit
// RUN: %clang_cc1 %s -fsycl-is-device -triple spir64 -aux-triple arm64-unknown-linux-gnu -target-feature +neon -DLONG -fsyntax-only -verify=quiet
typedef unsigned char poly8_t;
typedef unsigned short poly16_t;
typedef __UINT64_TYPE__ poly64_t;

#if defined(LONG)
// 32-bit on windows (LLP64)
// 64-bit on linux (LP64)
typedef unsigned long bad_poly32_t;
#else
typedef unsigned int bad_poly32_t;
#endif

typedef __attribute__((neon_polyvector_type(16))) poly8_t poly8x16_t;
typedef __attribute__((neon_polyvector_type(8))) poly16_t poly16x8_t;
typedef __attribute__((neon_polyvector_type(2))) poly64_t poly64x2_t;

// quiet-no-diagnostics
typedef __attribute__((neon_polyvector_type(2))) bad_poly32_t bad_poly32x2_t;
// expected-error@-1{{invalid vector element type}}
