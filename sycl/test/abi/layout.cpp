// RUN: clang++ -fsycl %s -o %t && %t
#include <CL/sycl.hpp>
#include <cassert>

using namespace cl::sycl;

#define CHECK_LAYOUT(class_name, size)               \
  if (sizeof(class_name) != size) {                  \
    std::cout << "Size of class " << #class_name <<  \
      " has changed. Was: " << #size << ". Now: " << \
      sizeof(class_name) << std::endl;               \
      HasFailed = true;                              \
  }

int main() {

  bool HasFailed = false;

  #include "layout_linux.def"

  return 0;
}
