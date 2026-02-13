#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>

using namespace sycl;

constexpr access::mode sycl_read = access::mode::read;
constexpr access::mode sycl_write = access::mode::write;

using BFP = sycl::ext::oneapi::bfloat16;

#ifdef BUILD_LIB
void foo(queue &deviceQueue) {
  BFP bf16_v;
  float fp32_v = 16.5f;
  {
    buffer<float, 1> fp32_buffer{&fp32_v, 1};
    buffer<BFP, 1> bf16_buffer{&bf16_v, 1};
    deviceQueue
        .submit([&](handler &cgh) {
          auto fp32_acc = fp32_buffer.template get_access<sycl_read>(cgh);
          auto bf16_acc = bf16_buffer.template get_access<sycl_write>(cgh);
          cgh.single_task([=]() { bf16_acc[0] = BFP{fp32_acc[0]}; });
        })
        .wait();
  }
  std::cout << "In foo: " << bf16_v << std::endl;
}
#endif

#ifdef BUILD_EXE
void foo(queue &deviceQueue);
#endif

int main() {
  BFP bf16_array[3];
  float fp32_array[3] = {7.0f, 8.5f, 0.5f};

  sycl::queue deviceQueue;
  {
    buffer<float, 1> fp32_buffer{fp32_array, 3};
    buffer<BFP, 1> bf16_buffer{bf16_array, 3};
    deviceQueue
        .submit([&](handler &cgh) {
          auto fp32_acc = fp32_buffer.template get_access<sycl_read>(cgh);
          auto bf16_acc = bf16_buffer.template get_access<sycl_write>(cgh);
          cgh.single_task([=]() {
            bf16_acc[0] = BFP{fp32_acc[0]};
            bf16_acc[1] = BFP{fp32_acc[1]};
            bf16_acc[2] = BFP{fp32_acc[2]};
          });
        })
        .wait();
  }
  std::cout << bf16_array[0] << " " << bf16_array[1] << " " << bf16_array[2]
            << std::endl;
#ifdef BUILD_EXE
  foo(deviceQueue);
#endif
  return 0;
}
