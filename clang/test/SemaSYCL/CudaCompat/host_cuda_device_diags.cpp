// RUN: %clang_cc1 %s -fsycl-is-device -triple nvptx64-nvidia-cuda -fsycl-cuda-compatibility -emit-llvm -o %t -verify=device
// RUN: %clang_cc1 %s -fsycl-is-host -aux-triple nvptx64-nvidia-cuda -fsycl-cuda-compatibility -emit-llvm -o %t -verify

// Check that errors happening on the device aren't raised during host compilation.

// device-no-diagnostics

__attribute__((device)) void callee() {
  float x;
  // no error
  asm("%0" : "=f"(x));
}

void caller() {
  float x;
  // expected-error@+1 {{invalid output constraint '=f' in asm}}
  asm("%0" : "=f"(x));
}
