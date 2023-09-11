// This test ensures the native-cpu device generates the expected kernel names,
// and that the MS mangler doesn't assert on the code below.

// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc   -aux-triple x86_64-pc-windows-msvc   -fsycl-is-device -fsycl-is-native-cpu -emit-llvm -o - -x c++ %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -aux-triple x86_64-unknown-linux-gnu -fsycl-is-device -fsycl-is-native-cpu -emit-llvm -o - -x c++ %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -aux-triple x86_64-pc-windows-msvc   -fsycl-is-device -fsycl-is-native-cpu -emit-llvm -o - -x c++ %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc   -aux-triple x86_64-unknown-linux-gnu -fsycl-is-device -fsycl-is-native-cpu -emit-llvm -o - -x c++ %s | FileCheck %s
// Todo: check other cpus

struct __nativecpu_state {};

template <typename KernelName, typename KernelType, typename... Props>
[[clang::sycl_kernel]] void
kernel_parallel_for_work_group(const KernelType &KernelFunc) {
  __nativecpu_state var;
  KernelFunc(&var);
}

struct name1;
template <typename KernelName = name1, typename KernelType>
void parallel_for_work_group1(const KernelType &KernelFunc) {
  kernel_parallel_for_work_group<KernelName, KernelType>(KernelFunc);
}

int main() {
  parallel_for_work_group1([=](__nativecpu_state *) {});
}

// CHECK: void @_ZTS5name1(
