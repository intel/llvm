// RUN: %clang_cc1 -fsycl-is-device %s -verify -DUNNAMED_LAMBDAS
// RUN: %clang_cc1 -fsycl-is-host %s -verify -DUNNAMED_LAMBDAS
//
// RUN: %clang_cc1 -fsycl-is-host -sycl-std=2017 %s -verify
// RUN: %clang_cc1 -fsycl-is-host -sycl-std=2020 %s -verify -DUNNAMED_LAMBDAS
//
// RUN: %clang_cc1 -fsycl-is-host -fno-sycl-unnamed-lambda %s -verify
// RUN: %clang_cc1 -fsycl-is-host -sycl-std=2017 -fno-sycl-unnamed-lambda %s -verify
// RUN: %clang_cc1 -fsycl-is-host -sycl-std=2017 -fsycl-unnamed-lambda %s -verify -DUNNAMED_LAMBDAS

// expected-no-diagnostics
#if defined(UNNAMED_LAMBDAS) && !defined(__SYCL_UNNAMED_LAMBDA__)
#error "Unnamed lambdas should be enabled for SYCL2020/default"
#endif

#if !defined(UNNAMED_LAMBDAS) && defined(__SYCL_UNNAMED_LAMBDA__)
#error "Unnamed lambdas should NOT be enabled for SYCL2017, or when explicitly disabled"
#endif
