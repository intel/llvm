// RUN: %clang_cc1 -internal-isystem %S/Inputs -fsycl-is-device -triple spir64-unknown-unknown -verify -emit-llvm-only %s

// Tests for error diagnostics when multiple definitions of
// [[__sycl_detail__::sycl_type(aspect)]] enums are present.
#include "sycl.hpp"

// expected-note@#AspectEnum{{previous definition is here}}

// expected-error@+1{{redefinition of aspect enum}}
enum class [[__sycl_detail__::sycl_type(aspect)]] aspect_redef {
  imposter_value = 3
};
