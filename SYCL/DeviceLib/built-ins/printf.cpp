// UNSUPPORTED: cuda || hip
// CUDA and HIP don't support printf.
//
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out %HOST_CHECK_PLACEHOLDER
// RUN: %CPU_RUN_PLACEHOLDER %t.out %CPU_CHECK_PLACEHOLDER
// RUN: %GPU_RUN_PLACEHOLDER %t.out %GPU_CHECK_PLACEHOLDER
// RUN: %ACC_RUN_PLACEHOLDER %t.out %ACC_CHECK_PLACEHOLDER

#include <sycl/sycl.hpp>

#include <cstdint>
#include <iostream>

using namespace sycl;

// According to OpenCL C spec, the format string must be in constant address
// space
#ifdef __SYCL_DEVICE_ONLY__
#define CONSTANT __attribute__((opencl_constant))
#else
#define CONSTANT
#endif

// This is one of the possible ways to define a format string in a correct
// address space
static const CONSTANT char format_hello_world[] = "Hello, World!\n";

// Static isn't really needed if you define it in global scope
const CONSTANT char format_int[] = "%d\n";

static const CONSTANT char format_vec[] = "%d,%d,%d,%d\n";

const CONSTANT char format_hello_world_2[] = "%lu: Hello, World!\n";

int main() {
  default_selector Selector;
  {
    queue Queue(Selector);

    Queue.submit([&](handler &CGH) {
      CGH.single_task<class integral>([=]() {
        // String
        ext::oneapi::experimental::printf(format_hello_world);
        // Due to a bug in Intel CPU Runtime for OpenCL on Windows, information
        // printed using such format strings (without %-specifiers) might
        // appear in different order if output is redirected to a file or
        // another app
        // FIXME: strictly check output order once the bug is fixed
        // CHECK: {{(Hello, World!)?}}

        // Integral types
        ext::oneapi::experimental::printf(format_int, (int32_t)123);
        ext::oneapi::experimental::printf(format_int, (int32_t)-123);
        // CHECK: 123
        // CHECK-NEXT: -123

        // Floating point types
        {
          // You can declare format string in non-global scope, but in this case
          // static keyword is required
          static const CONSTANT char format[] = "%f\n";
          ext::oneapi::experimental::printf(format, 33.4f);
          ext::oneapi::experimental::printf(format, -33.4f);
        }
        // CHECK-NEXT: 33.4
        // CHECK-NEXT: -33.4

        // Vectors
        cl::sycl::vec<int, 4> v4{5, 6, 7, 8};
#ifdef __SYCL_DEVICE_ONLY__
        // On device side, vectors can be printed via native OpenCL types:
        using ocl_int4 = cl::sycl::vec<int, 4>::vector_t;
        {
          static const CONSTANT char format[] = "%v4d\n";
          ext::oneapi::experimental::printf(format, (ocl_int4)v4);
        }

        // However, you are still able to print them by-element:
        {
          ext::oneapi::experimental::printf(format_vec, (int32_t)v4.w(),
                                            (int32_t)v4.z(), (int32_t)v4.y(),
                                            (int32_t)v4.x());
        }
#else
        // On host side you always have to print them by-element:
        ext::oneapi::experimental::printf(format_vec, (int32_t)v4.x(),
                                          (int32_t)v4.y(), (int32_t)v4.z(),
                                          (int32_t)v4.w());
        ext::oneapi::experimental::printf(format_vec, (int32_t)v4.w(),
                                          (int32_t)v4.z(), (int32_t)v4.y(),
                                          (int32_t)v4.x());
#endif // __SYCL_DEVICE_ONLY__
       // CHECK-NEXT: 5,6,7,8
       // CHECK-NEXT: 8,7,6,5

        // Pointers
        int a = 5;
        int *Ptr = &a;
        // According to OpenCL spec, argument should be a void pointer
        {
          static const CONSTANT char format[] = "%p\n";
          ext::oneapi::experimental::printf(format, (void *)Ptr);
        }
        // CHECK-NEXT: {{(0x)?[0-9a-fA-F]+$}}
      });
    });
    Queue.wait();
  }

  {
    queue Queue(Selector);
    // printf in parallel_for
    Queue.submit([&](handler &CGH) {
      CGH.parallel_for<class stream_string>(range<1>(10), [=](id<1> i) {
        // cast to uint64_t to be sure that we pass 64-bit unsigned value
        ext::oneapi::experimental::printf(format_hello_world_2,
                                          (uint64_t)i.get(0));
      });
    });
    Queue.wait();
    // CHECK-NEXT: {{[0-9]+}}: Hello, World!
    // CHECK-NEXT: {{[0-9]+}}: Hello, World!
    // CHECK-NEXT: {{[0-9]+}}: Hello, World!
    // CHECK-NEXT: {{[0-9]+}}: Hello, World!
    // CHECK-NEXT: {{[0-9]+}}: Hello, World!
    // CHECK-NEXT: {{[0-9]+}}: Hello, World!
    // CHECK-NEXT: {{[0-9]+}}: Hello, World!
    // CHECK-NEXT: {{[0-9]+}}: Hello, World!
    // CHECK-NEXT: {{[0-9]+}}: Hello, World!
    // CHECK-NEXT: {{[0-9]+}}: Hello, World!
  }

  // FIXME: strictly check output order once the bug mentioned above is fixed
  // CHECK: {{(Hello, World!)?}}

  return 0;
}
