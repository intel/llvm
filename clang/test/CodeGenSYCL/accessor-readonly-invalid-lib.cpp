// RUN: %clang_cc1 -triple spir64-unknown-unknown-sycldevice -fsycl-is-device -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s
//
// Test which verifies that readonly attribute is generated for unexpected access mode value.

// Dummy library with unexpected access::mode enum value.
namespace cl {
namespace sycl {

namespace access {

enum class target {
  global_buffer = 2014,
  constant_buffer,
  local,
  image,
  host_buffer,
  host_image,
  image_array
};

enum class mode {
  read = 2024,
  write,
  read_write,
  discard_write,
  discard_read_write,
  atomic
};

enum class placeholder {
  false_t,
  true_t
};

} // namespace access

template <int dim>
struct id {
  template <typename... T>
  id(T... args) {} // fake constructor
private:
  int Data;
};

template <int dim>
struct range {
  template <typename... T>
  range(T... args) {} // fake constructor
private:
  int Data;
};

template <int dim>
struct _ImplT {
  range<dim> AccessRange;
  range<dim> MemRange;
  id<dim> Offset;
};

template <typename dataT, int dimensions, access::mode accessmode,
          access::target accessTarget = access::target::global_buffer,
          access::placeholder isPlaceholder = access::placeholder::false_t>
class accessor {

public:
  void use(void) const {}
  _ImplT<dimensions> impl;

private:
  void __init(__attribute__((opencl_global)) dataT *Ptr, range<dimensions> AccessRange,
              range<dimensions> MemRange, id<dimensions> Offset) {}
};

} // namespace sycl
} // namespace cl

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(const Func &kernelFunc) {
  kernelFunc();
}

int main() {
  cl::sycl::accessor<int, 1, cl::sycl::access::mode::read> Acc;
  // CHECK: spir_kernel{{.*}}fake_kernel
  // CHECK-SAME: readonly
  kernel_single_task<class fake_kernel>([=]() {
    Acc.use();
  });
  return 0;
}
