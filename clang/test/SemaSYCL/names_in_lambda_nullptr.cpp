// RUN: %clang_cc1 -fsycl-is-device -std=c++20 -fsyntax-only %s -verify
// RUN: %clang_cc1 -fsycl-is-host -std=c++20 -fsyntax-only %s -verify

// This test checks that when a binding declaration is captured that
// we don't dereference the null VarDecl incorrectly.

// CHECK: expected-no-diagnostics

template <typename T>
struct P {
  P(T t) : t(t) {}
  T t;
};

int main() {
  auto usm = [=](float *ptr) {
    return P{ptr};
  };

  auto [in] = usm(new float());
  auto L = [=]() { *in = 42.f; };
  L();
  return 0;
}
