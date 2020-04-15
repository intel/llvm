// RUN: %clangxx -fsycl %s -o %t && %t

#include <CL/sycl.hpp>
#include <cassert>

using namespace cl::sycl;

#define CHECK_LAYOUT(class_name, size)                                    \
  if (sizeof(class_name) != size) {                                       \
    std::cout << "Size of class " << #class_name << " has changed. Was: " \
              << #size << ". Now: " << sizeof(class_name) << std::endl;   \
    HasChanged = true;                                                    \
  }

int main() {

  bool HasChanged = false;

  using accessor_t = accessor<int, 1, access::mode::read,
                              access::target::global_buffer, access::placeholder::true_t>;
  CHECK_LAYOUT(accessor_t, 32)
  CHECK_LAYOUT(buffer<int>, 40)
  CHECK_LAYOUT(context, 16)
  CHECK_LAYOUT(cpu_selector, 8)
  CHECK_LAYOUT(device, 16)
  CHECK_LAYOUT(device_event, 8)
  CHECK_LAYOUT(device_selector, 8)
  CHECK_LAYOUT(event, 16)
  CHECK_LAYOUT(gpu_selector, 8)
  CHECK_LAYOUT(handler, 472)
  CHECK_LAYOUT(image<1>, 16)
  CHECK_LAYOUT(kernel, 16)
  CHECK_LAYOUT(platform, 16)
  CHECK_LAYOUT(private_memory<int>, 8)
  CHECK_LAYOUT(program, 16)
  CHECK_LAYOUT(range<1>, 8)
  CHECK_LAYOUT(sampler, 16)
  CHECK_LAYOUT(stream, 208)
  CHECK_LAYOUT(queue, 16)

  assert(!HasChanged && "Some symbols changed their sizes!");

  return 0;
}
