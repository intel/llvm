// RUN: %clangxx -O0 -fsycl %s -o %t.out -lsycl
//
// Deprecated SYCL_PROGRAM_BUILD_OPTIONS should work as an alias to
// SYCL_PROGRAM_COMPILE_OPTIONS:
// RUN: %CPU_RUN_PLACEHOLDER SYCL_PROGRAM_BUILD_OPTIONS="-g" %t.out
// RUN: %GPU_RUN_PLACEHOLDER SYCL_PROGRAM_BUILD_OPTIONS="-g" %t.out
// RUN: %CPU_RUN_PLACEHOLDER SYCL_PROGRAM_COMPILE_OPTIONS="-g" %t.out
// RUN: %GPU_RUN_PLACEHOLDER SYCL_PROGRAM_COMPILE_OPTIONS="-g" %t.out
// RUN: %CPU_RUN_PLACEHOLDER SYCL_PROGRAM_LINK_OPTIONS="-enable-link-options -cl-denorms-are-zero" %t.out
// RUN: %GPU_RUN_PLACEHOLDER SYCL_PROGRAM_LINK_OPTIONS="-enable-link-options -cl-denorms-are-zero" %t.out
//
// Now test for invalid options to make sure they are really passed to
// a device compiler. Intel GPU runtime doesn't give an error for
// invalid options, so we don't test it here.
//
// RUN: %CPU_RUN_PLACEHOLDER SYCL_PROGRAM_COMPILE_OPTIONS="-enable-link-options -cl-denorms-are-zero" SHOULD_CRASH=1 %t.out
// RUN: %CPU_RUN_PLACEHOLDER SYCL_PROGRAM_LINK_OPTIONS="-g" SHOULD_CRASH=1 %t.out


#include <CL/sycl.hpp>
#include <cassert>
#include <memory>

using namespace cl::sycl;

int main() {
  int data = 5;
  buffer<int, 1> buf(&data, range<1>(1));
  queue myQueue;
  if (getenv("SHOULD_CRASH")) {
    try {
      myQueue.submit([&](handler &cgh) {
        auto B = buf.get_access<access::mode::read_write>(cgh);
        cgh.single_task<class kernel_name1>([=]() { B[0] = 0; });
      });
    } catch (cl::sycl::runtime_error &e) {
      // Exit immediately, otherwise the buffer destructor may actually try to
      // enqueue the command once again, and throw another exception.
      exit(0);
    } catch (cl::sycl::compile_program_error &e) {
      exit(0);
    }
    assert(0 && "Expected exception was *not* thrown");
  } else {
    myQueue.submit([&](handler &cgh) {
      auto B = buf.get_access<access::mode::read_write>(cgh);
      cgh.single_task<class kernel_name2>([=]() { B[0] = 0; });
    });
  }
}
