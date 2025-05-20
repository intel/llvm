#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/kernel_bundle.hpp>

#include <dlfcn.h>
#include <iostream>

using namespace sycl;

constexpr access::mode sycl_read = access::mode::read;
constexpr access::mode sycl_write = access::mode::write;

using BFP = sycl::ext::oneapi::bfloat16;

#ifdef BUILD_LIB
class FOO_KERN;
void foo() {
  queue deviceQueue;
  BFP bf16_v;
  float fp32_v = 16.5f;
  {
    buffer<float, 1> fp32_buffer{&fp32_v, 1};
    buffer<BFP, 1> bf16_buffer{&bf16_v, 1};
    deviceQueue
        .submit([&](handler &cgh) {
          auto fp32_acc = fp32_buffer.get_access<sycl_read>(cgh);
          auto bf16_acc = bf16_buffer.get_access<sycl_write>(cgh);
          cgh.single_task<FOO_KERN>([=]() { bf16_acc[0] = BFP{fp32_acc[0]}; });
        })
        .wait();
  }
  std::cout << "In foo: " << bf16_v << std::endl;
}
#else

class MAINRUN;
void main_run(queue &deviceQueue) {
  BFP bf16_v;
  float fp32_v = 16.5f;
  {
    buffer<float, 1> fp32_buffer{&fp32_v, 1};
    buffer<BFP, 1> bf16_buffer{&bf16_v, 1};
    deviceQueue
        .submit([&](handler &cgh) {
          auto fp32_acc = fp32_buffer.get_access<sycl_read>(cgh);
          auto bf16_acc = bf16_buffer.get_access<sycl_write>(cgh);
          cgh.single_task<class MAINRUN>(
              [=]() { bf16_acc[0] = BFP{fp32_acc[0] + 0.5f}; });
        })
        .wait();
  }
  std::cout << "In run: " << bf16_v << std::endl;
}

#define STRINGIFY_HELPER(A) #A
#define STRINGIFY(A) STRINGIFY_HELPER(A)
#define SO_FNAME "lib" STRINGIFY(FNAME) ".so"

int main() {
  BFP bf16_array[3];
  float fp32_array[3] = {7.0f, 8.5f, 0.5f};
  queue deviceQueue;
  std::vector<sycl::kernel_id> all_kernel_ids;
  bool dynlib_kernel_available = false;
  bool dynlib_kernel_unavailable = true;
  main_run(deviceQueue);

  void *handle = dlopen(SO_FNAME, RTLD_LAZY);
  void (*func)();
  *(void **)(&func) = dlsym(handle, "_Z3foov");
  func();
  all_kernel_ids = sycl::get_kernel_ids();
  for (auto k : all_kernel_ids) {
    if (k.get_name() && std::strstr(k.get_name(), "FOO_KERN"))
      dynlib_kernel_available = true;
  }

  // Before dlclose, the FOO_KERN from sycl dynamic library must exist.
  assert(dynlib_kernel_available);

  dlclose(handle);

  all_kernel_ids = sycl::get_kernel_ids();
  for (auto k : all_kernel_ids) {
    if (k.get_name() && std::strstr(k.get_name(), "FOO_KERN"))
      dynlib_kernel_unavailable = false;
  }

  assert(dynlib_kernel_unavailable);

  {
    buffer<float, 1> fp32_buffer{fp32_array, 3};
    buffer<BFP, 1> bf16_buffer{bf16_array, 3};
    deviceQueue
        .submit([&](handler &cgh) {
          auto fp32_acc = fp32_buffer.get_access<sycl_read>(cgh);
          auto bf16_acc = bf16_buffer.get_access<sycl_write>(cgh);
          cgh.single_task([=]() {
            bf16_acc[0] = BFP{fp32_acc[0]};
            bf16_acc[1] = BFP{fp32_acc[1]};
            bf16_acc[2] = BFP{fp32_acc[2]};
          });
        })
        .wait();
  }
  std::cout << "In main: " << bf16_array[0] << " " << bf16_array[1] << " "
            << bf16_array[2] << std::endl;

  return 0;
}
#endif
