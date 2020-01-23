// RUN: %clang_cc1 %s -emit-llvm -triple spir64-unknown-unknown-sycldevice -fsycl-is-device -disable-llvm-passes -o - | FileCheck %s
// CHECK: %opencl.pipe_storage_t = type opaque
// CHECK: %opencl.pipe_wo_t = type opaque
// CHECK: %opencl.pipe_ro_t = type opaque
// CHECK: @{{.*}} = internal addrspace(1) global %opencl.pipe_storage_t addrspace(1)* null

using PipeStorageTy = __attribute__((pipe("storage"))) const int;
PipeStorageTy PipeStorageCreator();
PipeStorageTy PipeStorage = PipeStorageCreator();

using WPipeTy = __attribute__((pipe("write_only"))) const int;
WPipeTy WPipeCreator(PipeStorageTy PS);

using RPipeTy = __attribute__((pipe("read_only"))) const int;
RPipeTy RPipeCreator(PipeStorageTy PS);

template <typename PipeTy>
void foo(PipeTy Pipe) {}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc();
}

int main() {
  kernel_single_task<class kernel_function>([]() {
    // CHECK: alloca %opencl.pipe_wo_t
    WPipeTy wpipe = WPipeCreator(PipeStorage);
    // CHECK: alloca %opencl.pipe_ro_t
    RPipeTy rpipe = RPipeCreator(PipeStorage);
    foo<WPipeTy>(wpipe);
    foo<RPipeTy>(rpipe);
  });
  return 0;
}

