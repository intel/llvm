// RUN: %clang_cc1 -fsycl-is-device -std=c++20 -fsyntax-only %s -verify
// RUN: %clang_cc1 -fsycl-is-host -std=c++20 -fsyntax-only %s -verify

// This test checks that when a binding declaration is captured that
// we don't dereference the null VarDecl incorrectly.

// CHECK: expected-no-diagnostics

int main() {
  int a[2] = {1, 2};
  auto [x, y] = a;
  auto Lambda = [=]() {x = 10;};
  Lambda();
  return 0;
}
