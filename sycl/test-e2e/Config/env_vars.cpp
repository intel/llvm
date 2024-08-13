// REQUIRES: opencl
// Env vars are used to pass OpenCL-specific flags to PI compiling/linking.
//
// RUN: %{build} -O0 -o %t.out
//
// RUN: env SYCL_PROGRAM_COMPILE_OPTIONS="-g" %{run} %t.out
// RUN: env SYCL_PROGRAM_APPEND_COMPILE_OPTIONS="-g" %{run} %t.out
//
// Now test for invalid options to make sure they are really passed to
// a device compiler. Intel GPU runtime doesn't give an error for
// invalid options, so we don't test it here.
//
// RUN: %if cpu %{ env SYCL_PROGRAM_COMPILE_OPTIONS="-enable-link-options -cl-denorms-are-zero" SHOULD_CRASH=1 %{run} %t.out %}
// RUN: %if cpu %{ env SYCL_PROGRAM_APPEND_COMPILE_OPTIONS="-enable-link-options -cl-denorms-are-zero" SHOULD_CRASH=1 %{run} %t.out %}

#include <cassert>
#include <sycl/detail/core.hpp>

using namespace sycl;

int main() {
  int data = 5;
  buffer<int, 1> buf(&data, range<1>(1));
  queue myQueue;
  bool shouldCrash = getenv("SHOULD_CRASH");
  try {
    myQueue.submit([&](handler &cgh) {
      auto B = buf.get_access<access::mode::read_write>(cgh);
      cgh.single_task<class kernel_name1>([=]() { B[0] = 0; });
    });
    assert(!shouldCrash);
  } catch (sycl::exception &e) {
    assert(shouldCrash);
    assert(e.code() == errc::build);
  }

  return 0;
}
