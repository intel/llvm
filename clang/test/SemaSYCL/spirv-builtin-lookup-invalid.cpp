// RUN: %clang_cc1 -fdeclare-spirv-builtins -fsyntax-only -verify %s

// Verify that invalid call to __spirv_ocl_acos (no viable overloads) get diagnosed

struct InvalidType {};
void acos(InvalidType Invalid) {
  __spirv_ocl_acos(Invalid); // expected-error {{no matching function for call to '__spirv_ocl_acos'}}
  // expected-note@-1 + {{candidate function not viable: no known conversion from}}
  // too many params
  __spirv_ocl_acos(42.f, 42.f); // expected-error {{no matching function for call to '__spirv_ocl_acos'}}
  // expected-note@-1 + {{candidate function not viable: requires 1 argument, but 2 were provided}}
}
