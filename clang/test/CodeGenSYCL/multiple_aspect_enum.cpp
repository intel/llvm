// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -fsyntax-only -verify -emit-llvm-only %s

// Tests for error diagnostics when multiple definitions of
// [[__sycl_detail__::sycl_type(aspect)]] enums are present.

// expected-note@+1{{previous definition is here}}
enum class [[__sycl_detail__::sycl_type(aspect)]] aspect {
  host = 0,
  cpu = 1,
  gpu = 2,
  accelerator = 3,
  custom = 4,
  fp16 = 5,
  fp64 = 6,
  future_aspect = 12
};

// expected-error@+1{{redefinition of aspect enum}}
enum class [[__sycl_detail__::sycl_type(aspect)]] aspect_redef {
  imposter_value = 3
};
