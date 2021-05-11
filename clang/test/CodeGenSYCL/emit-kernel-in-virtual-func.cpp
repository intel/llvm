// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -emit-llvm %s -o - | FileCheck %s

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(const Func &kernelFunc) {
  kernelFunc();
}

class BASE {
public:
  virtual void initialize() {}
  virtual ~BASE();
};

template <class T>
class DERIVED : public BASE {
public:
  void initialize() {
    kernel_single_task<class FF>([]() { });
  }
};

int main() {
  BASE *Base = new DERIVED<int>;
  Base->initialize();
  delete Base;
}

// Ensure that the SPIR-Kernel function is actually emitted.
// CHECK: define {{.*}}spir_kernel void @_ZTSZN7DERIVEDIiE10initializeEvE2FF
