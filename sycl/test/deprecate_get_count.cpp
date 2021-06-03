// RUN: %clangxx -fsycl -sycl-std=2020 -Xclang -verify -Xclang -verify-ignore-unexpected=note %s -o %t.out
#include <CL/sycl.hpp>

using namespace cl::sycl;

int main() {
  // Create a buffer of 4 ints to be used inside the kernel code.
  buffer<int, 1> Buffer(4);
  // expected-warning@+1{{'get_count' is deprecated: get_count() is deprecated, please use size() instead}}
  size_t BufferGetCount = Buffer.get_count();
  size_t BufferSize = Buffer.size();
}
