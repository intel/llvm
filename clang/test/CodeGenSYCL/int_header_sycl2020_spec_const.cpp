// RUN: %clang_cc1 -fsycl -fsycl-is-device -triple nvptx64-unknown-unknown -fsycl-int-header=%t.h %s -o %t.out %s -o %t.out
// RUN: FileCheck -input-file=%t.h %s --check-prefix=NONATIVESUPPORT --check-prefix=ALL
// RUN: %clang_cc1 -fsycl -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -fsycl-int-header=%t.h %s -o %t.out %s -o %t.out
// RUN: FileCheck -input-file=%t.h %s --check-prefix=NATIVESUPPORT --check-prefix=ALL

// This test checks that the compiler generates required information
// in integration header for kernel_handler type (SYCL 2020 specialization 
// constants).

//FIXME: Move to headers
namespace cl {
namespace sycl {
class kernel_handler {
  void __init_specialization_constants_buffer(char *specialization_constants_buffer) {}
};
} // namespace sycl
} // namespace cl

template <typename name, typename Func>
__attribute__((sycl_kernel)) void a_kernel(Func kernelFunc, cl::sycl::kernel_handler kh) {
  kernelFunc(kh);
}

int main() {
  int a;
  cl::sycl::kernel_handler kh;

  a_kernel<class test_kernel_handler>(
      [=](auto) {
        int local = a;
      },
      kh);
}
// ALL: const kernel_param_desc_t kernel_signatures[] = {
// NONATIVESUPPORT: { kernel_param_kind_t::kind_specialization_constants_buffer, 8, 0 } 
// NATIVESUPPORT-NOT: { kernel_param_kind_t::kind_specialization_constants_buffer, 8, 0 }
