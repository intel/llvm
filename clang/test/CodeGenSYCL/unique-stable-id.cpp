// RUN: %clang_cc1 -triple x86_64-linux-pc  -fsycl-is-host -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

//#include "Inputs/sycl.hpp"

//const char * __builtin_sycl_unique_stable_id2(const char[]);

int global;
template <typename Ty>
auto func() -> decltype(__builtin_sycl_unique_stable_id(Ty::str));
//auto func() -> decltype(__builtin_sycl_unique_stable_id2(Ty::str));

struct Derp {
  static constexpr const char str[] = "derp derp derp";
};

template <typename KernelName, typename KernelType>
void not_kernel_single_task(KernelType kernelFunc) {
  //kernelFunc();
}


int main() {
  not_kernel_single_task<class kernel2>(func<Derp>);
  // TODO: This demangles funny?
  // CHECK: call void @_Z22not_kernel_single_taskIZ4mainE7kernel2PFPKcvEEvT0_(i8* ()* @_Z4funcI4DerpEDTu31__builtin_sycl_unique_stable_idsrT_3strEEv)

  // TODO: ERICH: Whatever else we can come up with.
}
