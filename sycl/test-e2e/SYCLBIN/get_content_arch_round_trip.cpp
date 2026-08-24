// REQUIRES: aspect-usm_shared_allocations

// -- End-to-end test that the per-image "arch" metadata on a SYCLBIN survives
// -- a serialize -> reload -> serialize cycle.
// --
// -- Flow:
// --   1. Touch a kernel so the bundle has a registered native device image.
// --   2. ext_oneapi_get_content() to produce SYCLBIN bytes #1. Write to disk
// --      and dump - should report a non-empty arch (e.g. "intel_gpu_*").
// --   3. Reload the bytes as an executable bundle.
// --   4. ext_oneapi_get_content() again -> SYCLBIN bytes #2. Dump again -
// --      arch must still be the same non-empty string. If it is empty, the
// --      reload+re-serialize path lost the per-image architecture info.

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out %t.first.syclbin %t.second.syclbin
// RUN: %{run-aux} syclbin-dump %t.first.syclbin  | FileCheck %s --check-prefix CHECK-FIRST
// RUN: %{run-aux} syclbin-dump %t.second.syclbin | FileCheck %s --check-prefix CHECK-SECOND

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/syclbin_kernel_bundle.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/usm.hpp>

#include <fstream>
#include <iostream>
#include <vector>

namespace syclexp = sycl::ext::oneapi::experimental;

class ArchProbeKernel;

static void writeBytes(const char *Path, const std::vector<char> &Bytes) {
  std::ofstream OS{Path, std::ios::binary};
  OS.write(Bytes.data(), Bytes.size());
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0]
              << " <first.syclbin-out> <second.syclbin-out>\n";
    return 1;
  }
  const char *FirstPath = argv[1];
  const char *SecondPath = argv[2];

  sycl::queue Q;
  const sycl::context Ctx = Q.get_context();
  const sycl::device Dev = Q.get_device();

  // Touch the kernel so the bundle has a registered native image.
  {
    int *Sink = sycl::malloc_shared<int>(1, Q);
    Q.submit([&](sycl::handler &CGH) {
       CGH.single_task<ArchProbeKernel>([=]() { *Sink = 7; });
     }).wait_and_throw();
    sycl::free(Sink, Q);
  }

  // First serialize: from the registered executable bundle.
  auto KB1 =
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(Ctx, {Dev});
  std::vector<char> Bytes1 = KB1.ext_oneapi_get_content();
  if (Bytes1.empty()) {
    std::cout << "ext_oneapi_get_content #1 returned empty bytes\n";
    return 1;
  }
  writeBytes(FirstPath, Bytes1);

  // Reload as executable, then serialize again.
  auto KB2 = syclexp::get_kernel_bundle<sycl::bundle_state::executable>(
      Ctx, {Dev}, sycl::span<char>{Bytes1});
  std::vector<char> Bytes2 = KB2.ext_oneapi_get_content();
  if (Bytes2.empty()) {
    std::cout << "ext_oneapi_get_content #2 returned empty bytes\n";
    return 1;
  }
  writeBytes(SecondPath, Bytes2);

  std::cout << "OK first=" << Bytes1.size() << " second=" << Bytes2.size()
            << "\n";
  return 0;
}

// First serialize comes from a bundle whose images are still associated with
// device_impl objects, so the serializer fills "arch" via getArchName(...).
// We expect the "arch:" line to contain at least one alphabetic character.
// CHECK-FIRST:        SYCLBIN/native device code image metadata:
// CHECK-FIRST:        arch:{{.*[a-zA-Z].*}}

// Second serialize comes from a bundle reloaded from SYCLBIN bytes. The
// images on the reloaded side carry "arch" only in the per-image SYCLBIN
// property set; the source image has no "compile_target" property. The
// serializer's Path B falls back to getArchName(Devices[0]), so on re-emit
// "arch" must still be a non-empty architecture string. If this CHECK fails
// (empty arch), the round-trip lost the per-image architecture info.
// CHECK-SECOND:        SYCLBIN/native device code image metadata:
// CHECK-SECOND:        arch:{{.*[a-zA-Z].*}}
