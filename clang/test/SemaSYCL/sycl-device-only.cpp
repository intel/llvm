// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsycl-is-host -fsyntax-only -verify %s

// Semantic tests for sycl_device_only attribute

// Valid uses
int foook(int x) {
  return x + 10;
}

__attribute__((sycl_device_only)) int foook(int x) {
  return x + 20;
}

// Conflicting attributes
// expected-note@+1 {{conflicting attribute is here}}
__attribute__((sycl_device_only, sycl_device)) // expected-error {{'sycl_device' and 'sycl_device_only' attributes are not compatible}}
int fooconflict(int x) {
  return x + 20;
}

// Bad overload
__attribute__((sycl_device))
int foobad(int x) { // expected-note {{previous definition is here}}
  return x + 10;
}

__attribute__((sycl_device_only))
int foobad(int x) { // expected-error {{redefinition of 'foobad'}}
  return x + 20;
}
