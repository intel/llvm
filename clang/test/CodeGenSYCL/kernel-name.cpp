// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-linux-sycldevice -std=c++11 -S -emit-llvm -x c++ %s -o - | FileCheck %s

inline namespace cl {
  namespace sycl {
    class kernel {};
  }
}

using namespace cl::sycl;

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(const Func &kernelFunc) {
  kernelFunc();
}

// CHECK: define {{.*}}spir_kernel {{.*}}2cl4sycl6kernel
int main() {
  kernel_single_task<class kernel>([]() {});
  return 0;
}
