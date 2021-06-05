// RUN: %clang_cc1 -fsycl-is-device %s -emit-llvm -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -o - | FileCheck %s
// CHECK: %opencl.pipe_wo_t
// CHECK: %opencl.pipe_ro_t

using WPipeTy = __attribute__((pipe("write_only"))) const int;
SYCL_EXTERNAL WPipeTy WPipeCreator();

using RPipeTy = __attribute__((pipe("read_only"))) const int;
SYCL_EXTERNAL RPipeTy RPipeCreator();

template <typename PipeTy>
void foo(PipeTy Pipe) {}

struct PipeStorageTy {
  int Size;
};

// CHECK:  @{{.*}}Storage = {{.*}} !io_pipe_id ![[ID0:[0-9]+]]
constexpr PipeStorageTy
    Storage __attribute__((io_pipe_id(1))) = {1};

// CHECK:  @{{.*}}TempStorage{{.*}} = {{.*}} !io_pipe_id ![[ID1:[0-9]+]]
template <int N>
constexpr PipeStorageTy
    TempStorage __attribute__((io_pipe_id(N))) = {2};

SYCL_EXTERNAL void boo(PipeStorageTy PipeStorage);

template <int ID>
struct ethernet_pipe {
  static constexpr int id = ID;
};

// CHECK:  @{{.*}}PipeStorage{{.*}} = {{.*}} !io_pipe_id ![[ID2:[0-9]+]]
template <typename name>
class pipe {
public:
  static void read() {
    boo(PipeStorage);
  }

private:
  static constexpr PipeStorageTy
      PipeStorage __attribute__((io_pipe_id(name::id))) = {3};
};

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(const Func &kernelFunc) {
  kernelFunc();
}

int main() {
  kernel_single_task<class kernel_function>([]() {
    // CHECK: alloca %opencl.pipe_wo_t
    WPipeTy wpipe = WPipeCreator();
    // CHECK: alloca %opencl.pipe_ro_t
    RPipeTy rpipe = RPipeCreator();
    foo<WPipeTy>(wpipe);
    foo<RPipeTy>(rpipe);
    boo(Storage);
    boo(TempStorage<2>);
    pipe<ethernet_pipe<42>>::read();
  });
  return 0;
}
// CHECK: ![[ID0]] = !{i32 1}
// CHECK: ![[ID1]] = !{i32 2}
// CHECK: ![[ID2]] = !{i32 42}
