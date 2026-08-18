// REQUIRES: aspect-usm_shared_allocations

// -- End-to-end test that a regular lambda kernel survives a SYCLBIN round-
// -- trip via ext_oneapi_get_content().
// --
// -- Flow:
// --   1. Anchor a lambda kernel K1 by submitting it once. This makes the
// --      bundle's kernel-id population happen the same way as any normal
// --      SYCL program.
// --   2. Get an executable kernel_bundle scoped to K1.
// --   3. ext_oneapi_get_content() to serialize.
// --   4. Reload the bytes as an executable kernel_bundle.
// --   5. Check that the reloaded bundle still reports the kernel via the
// --      C++ kernel-id API (has_kernel<K1>()).
// --
// -- The SYCLBIN serializer synthesizes a [SYCL/kernel names] property set
// -- from the runtime-tracked device_image_impl::getKernelNames(); the
// -- reader at device_image_impl.hpp populates MKernelNames from that
// -- property set, restoring kernel-id registration on reload.

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/syclbin_kernel_bundle.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/usm.hpp>

#include <iostream>
#include <vector>

namespace syclexp = sycl::ext::oneapi::experimental;

class K1;

int main() {
  sycl::queue Q;
  const sycl::context Ctx = Q.get_context();
  const sycl::device Dev = Q.get_device();

  int *S = sycl::malloc_shared<int>(1, Q);
  *S = 0;
  Q.submit([&](sycl::handler &CGH) {
     CGH.single_task<K1>([=]() { *S = 42; });
   }).wait_and_throw();
  if (*S != 42) {
    std::cout << "Pre-condition failed: baseline K1 did not run.\n";
    sycl::free(S, Q);
    return 1;
  }

  auto KB = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
      Ctx, {Dev}, std::vector<sycl::kernel_id>{sycl::get_kernel_id<K1>()});

  if (!KB.has_kernel<K1>() || KB.get_kernel_ids().size() != 1) {
    std::cout << "Pre-condition failed: original bundle does not report K1.\n";
    sycl::free(S, Q);
    return 1;
  }

  std::vector<char> Bytes = KB.ext_oneapi_get_content();
  if (Bytes.empty()) {
    std::cout << "ext_oneapi_get_content returned empty bytes.\n";
    sycl::free(S, Q);
    return 1;
  }

  auto KBR = syclexp::get_kernel_bundle<sycl::bundle_state::executable>(
      Ctx, {Dev}, sycl::span<char>{Bytes});

  const bool HasK1 = KBR.has_kernel<K1>();
  const size_t NumIds = KBR.get_kernel_ids().size();

  std::cout << "KB    has_kernel<K1>=" << KB.has_kernel<K1>()
            << " kernel_ids=" << KB.get_kernel_ids().size() << "\n";
  std::cout << "KBR   has_kernel<K1>=" << HasK1 << " kernel_ids=" << NumIds
            << "\n";

  sycl::free(S, Q);

  if (!HasK1 || NumIds == 0) {
    std::cout << "FAIL: reloaded bundle dropped all kernel-id registration.\n";
    return 1;
  }
  std::cout << "OK: kernel-id round-trip preserved.\n";
  return 0;
}
