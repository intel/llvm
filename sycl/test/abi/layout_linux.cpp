// RUN: clang++ -fsycl %s -o %t && %t
#include <CL/sycl.hpp>
#include <cassert>

using namespace cl::sycl;

int main() {
  assert(sizeof(accessor<int, 1, access::mode::read, access::target::global_buffer, access::placeholder::true_t>) == 32);
  assert(sizeof(buffer<int>) == 40);
  assert(sizeof(context) == 16);
  assert(sizeof(cpu_selector) == 8);
  assert(sizeof(device) == 16);
  assert(sizeof(device_event) == 8);
  assert(sizeof(device_selector) == 8);
  assert(sizeof(event) == 16);
  assert(sizeof(gpu_selector) == 8);
  assert(sizeof(handler) == 440);
  assert(sizeof(kernel) == 16);
  assert(sizeof(platform) == 16);
  assert(sizeof(private_memory<int>) == 8);
  assert(sizeof(program) == 16);
  assert(sizeof(range<1>) == 8);
  assert(sizeof(queue) == 16);

  return 0;
}
