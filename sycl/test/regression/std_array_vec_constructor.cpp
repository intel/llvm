// Test to isolate sycl::vec bug due to use of std::array in
// the constructor.
// REQUIRES: windows

// RUN: not %clangxx -O0 -fsycl -D_DEBUG -shared %s &> %t.compile.log
// RUN: FileCheck %s -input-file=%t.compile.log

#include <sycl/sycl.hpp>

using namespace sycl::ext::oneapi::experimental;

// CHECK: UnsupportedVarArgFunction: Variadic functions other than 'printf' are not supported in SPIR-V.
// CHECK: clang{{.*}} error: llvm-spirv command failed with exit code 23{{.*}}
auto Reproducer(sycl::queue q, sampled_image_handle imgHanlde) {
  return q.submit([&](sycl::handler &cg) {
    cg.parallel_for(
        sycl::nd_range<3>({1, 1, 1}, {1, 1, 1}), [=](sycl::nd_item<3> item) {
          [[maybe_unused]] auto val =
              sample_image<unsigned char>(imgHanlde, sycl::float2(1, 2));
        });
  });
}
