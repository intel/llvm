// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2020 -verify -fsyntax-only %s
// UNSUPPORTED: system-windows
// expected-no-diagnostics
// This test checks that an undefined "__failed_assertion" without SYCL_EXTERNAL which is declared in
// a system header won't lead to SYCL sema check failure.

#include "sycl.hpp"
#include <dummy_failed_assert>

SYCL_EXTERNAL
void call_failed_assertion() {
  __failed_assertion();
}

