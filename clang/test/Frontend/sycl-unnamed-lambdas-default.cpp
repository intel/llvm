// RUN: %clang_cc1 -fsycl-is-device %s -verify -DUNNAMED_LAMBDAS
// RUN: %clang_cc1 -fsycl-is-host %s -verify -DUNNAMED_LAMBDAS
//
// RUN: %clang_cc1 -fsycl-is-host %s -verify -DUNNAMED_LAMBDAS
//
// RUN: %clang_cc1 -fsycl-is-host -fno-sycl-unnamed-lambda %s -verify
// RUN: %clang_cc1 -fsycl-is-host -fno-sycl-unnamed-lambda %s -verify
// RUN: %clang_cc1 -fsycl-is-host -fsycl-unnamed-lambda %s -verify -DUNNAMED_LAMBDAS

// expected-no-diagnostics
#if defined(UNNAMED_LAMBDAS) && !defined(__SYCL_UNNAMED_LAMBDA__)
#error "Unnamed lambdas should be enabled"
#endif

#if !defined(UNNAMED_LAMBDAS) && defined(__SYCL_UNNAMED_LAMBDA__)
#error "Unnamed lambdas should NOT be enabled"
#endif
