// No JITing for host devices.
// REQUIRES: opencl || level_zero || cuda
// RUN: rm -rf %t/cache_dir
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out -DTARGET_IMAGE=INC100
// Build program and add item to cache
// RUN: env SYCL_CACHE_DIR=%t/cache_dir SYCL_PI_TRACE=-1 %t.out | FileCheck %s --check-prefixes=CHECK-BUILD
// Ignore caching because image size is less than threshold
// RUN: env SYCL_CACHE_DIR=%t/cache_dir SYCL_PI_TRACE=-1 SYCL_CACHE_MIN_DEVICE_IMAGE_SIZE=100000 %t.out | FileCheck %s --check-prefixes=CHECK-BUILD
// Ignore caching because image size is more than threshold
// RUN: env SYCL_CACHE_DIR=%t/cache_dir SYCL_PI_TRACE=-1 SYCL_CACHE_MAX_DEVICE_IMAGE_SIZE=1000 %t.out | FileCheck %s --check-prefixes=CHECK-BUILD
// Use cache
// RUN: env SYCL_CACHE_DIR=%t/cache_dir SYCL_PI_TRACE=-1 %t.out | FileCheck %s --check-prefixes=CHECK-CACHE
// Ignore cache because of environment variable
// RUN: env SYCL_CACHE_DIR=%t/cache_dir SYCL_PI_TRACE=-1 SYCL_CACHE_DISABLE_PERSISTENT=1 %t.out | FileCheck %s --check-prefixes=CHECK-BUILD
//
// The test checks environment variables which may disable caching.
// Also it can be used for benchmarking cache:
// Rough data collected on 32 core machine.
// Number of lines    1      10    100    1000   10000
// Image Size(kB)     2       2	    20	   165    1700
// Device code build time in seconds
// CPU OCL JIT       0.12    0.12  0.16     1.1     16
// CPU OCL Cache     0.01    0.01  0.01	   0.02   0.08

// CHECK-BUILD: piProgramBuild
// CHECK-BUILD-NOT: piProgramCreateWithBinary

// CHECK-CACHE-NOT: piProgramBuild
// CHECK-CACHE: piProgramCreateWithBinary

#define INC1(x) ((x) = (x) + 1);

#define INC10(x)                                                               \
  INC1(x)                                                                      \
  INC1(x)                                                                      \
  INC1(x)                                                                      \
  INC1(x)                                                                      \
  INC1(x)                                                                      \
  INC1(x)                                                                      \
  INC1(x)                                                                      \
  INC1(x)                                                                      \
  INC1(x)                                                                      \
  INC1(x)

#define INC100(x)                                                              \
  INC10(x)                                                                     \
  INC10(x)                                                                     \
  INC10(x)                                                                     \
  INC10(x)                                                                     \
  INC10(x)                                                                     \
  INC10(x)                                                                     \
  INC10(x)                                                                     \
  INC10(x)                                                                     \
  INC10(x)                                                                     \
  INC10(x)

#define INC1000(x)                                                             \
  INC100(x)                                                                    \
  INC100(x)                                                                    \
  INC100(x)                                                                    \
  INC100(x)                                                                    \
  INC100(x)                                                                    \
  INC100(x)                                                                    \
  INC100(x)                                                                    \
  INC100(x)                                                                    \
  INC100(x)                                                                    \
  INC100(x)

#define INC10000(x)                                                            \
  INC1000(x)                                                                   \
  INC1000(x)                                                                   \
  INC1000(x)                                                                   \
  INC1000(x)                                                                   \
  INC1000(x)                                                                   \
  INC1000(x)                                                                   \
  INC1000(x)                                                                   \
  INC1000(x)                                                                   \
  INC1000(x)                                                                   \
  INC1000(x)

#define INC100000(x)                                                           \
  INC10000(x)                                                                  \
  INC10000(x)                                                                  \
  INC10000(x)                                                                  \
  INC10000(x)                                                                  \
  INC10000(x)                                                                  \
  INC10000(x)                                                                  \
  INC10000(x)                                                                  \
  INC10000(x)                                                                  \
  INC10000(x)                                                                  \
  INC10000(x)

#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
class Inc;
template <class Kernel> void check_build_time(cl::sycl::queue &q) {
  cl::sycl::program program(q.get_context());
  auto start = std::chrono::steady_clock::now();
  program.build_with_kernel_type<Kernel>();
  auto end = std::chrono::steady_clock::now();

  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "elapsed build time: " << elapsed_seconds.count() << "s\n";
}
int main(int argc, char **argv) {
  auto start = std::chrono::steady_clock::now();
  // Test program and kernel APIs when building a kernel.
  {
    cl::sycl::queue q;
    check_build_time<Inc>(q);

    int data = 0;
    {
      cl::sycl::buffer<int, 1> buf(&data, cl::sycl::range<1>(1));
      cl::sycl::range<1> NumOfWorkItems{buf.get_count()};

      q.submit([&](cl::sycl::handler &cgh) {
        auto acc = buf.get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.parallel_for<class Inc>(
            NumOfWorkItems, [=](cl::sycl::id<1> WIid) { TARGET_IMAGE(acc[0]) });
      });
    }
    // check_build_time<Inc>(q);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "elapsed kernel time: " << elapsed_seconds.count() << "s\n";
  }
}
