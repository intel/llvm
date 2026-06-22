
// REQUIRES: aspect-usm_shared_allocations

// -- End-to-end round-trip test for ext_oneapi_get_content() preserving a
// -- user-set specialization constant value.
// --
// -- Flow:
// --   1. Build an input-state kernel_bundle. The kernel below references
// --      the spec const, so it is statically registered with the image.
// --   2. Override the spec const via set_specialization_constant<>.
// --   3. sycl::build it to executable; the override is applied at JIT time.
// --   4. ext_oneapi_get_content() to serialize.
// --   5. Reload the bytes as an executable kernel_bundle.
// --   6. Read the spec const value back via the *bundle* host API on the
// --      reloaded bundle. It must equal the user-set value (12345), not
// --      the default (42).
// --

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/syclbin_kernel_bundle.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/specialization_id.hpp>
#include <sycl/usm.hpp>

#include <iostream>
#include <vector>

namespace syclexp = sycl::ext::oneapi::experimental;

constexpr static int DefaultValue = 42;
constexpr static int UserValue = 12345;

constexpr sycl::specialization_id<int> SC{DefaultValue};

// A trivial kernel that references the spec const at compile time so the
// frontend ties the spec const to a registered device image. We do not need
// to actually run this kernel against the reloaded bundle to verify the
// round-trip.
class ReadSpecConstKernel;

int main() {
  sycl::queue Q;
  const sycl::context Ctx = Q.get_context();
  const sycl::device Dev = Q.get_device();

  // Touch the kernel once at compile-time so the spec const is associated
  // with a registered image. We submit it on the side to also exercise the
  // baseline path; the round-trip work below uses a fresh input-state bundle.
  {
    int *Sink = sycl::malloc_shared<int>(1, Q);
    Q.submit([&](sycl::handler &CGH) {
       CGH.single_task<ReadSpecConstKernel>([=](sycl::kernel_handler KH) {
         *Sink = KH.get_specialization_constant<SC>();
       });
     }).wait_and_throw();
    sycl::free(Sink, Q);
  }

  // 1. Input-state bundle.
  auto KBInput = sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev});

  if (!KBInput.contains_specialization_constants()) {
    std::cout << "FAIL: input bundle has no specialization constants.\n";
    return 1;
  }

  // 2. Override the spec const.
  if (KBInput.get_specialization_constant<SC>() != DefaultValue) {
    std::cout << "Pre-condition failed: input bundle did not start with the "
                 "default spec const value.\n";
    return 1;
  }
  KBInput.set_specialization_constant<SC>(UserValue);
  if (KBInput.get_specialization_constant<SC>() != UserValue) {
    std::cout << "Pre-condition failed: host read of input bundle spec const "
                 "did not return user value after set.\n";
    return 1;
  }

  // 3. Build to executable. The override is applied to the UR program here.
  auto KBExe = sycl::build(KBInput);

  if (KBExe.get_specialization_constant<SC>() != UserValue) {
    std::cout << "Pre-condition failed: host read of executable bundle "
                 "spec const did not return user value after build.\n";
    return 1;
  }

  // 4. Serialize.
  std::vector<char> Bytes = KBExe.ext_oneapi_get_content();
  if (Bytes.empty()) {
    std::cout << "ext_oneapi_get_content returned empty bytes.\n";
    return 1;
  }

  // 5. Reload as executable.
  auto KBReloaded = syclexp::get_kernel_bundle<sycl::bundle_state::executable>(
      Ctx, {Dev}, sycl::span<char>{Bytes});

  // 6. Compare host-side spec const value on the reloaded bundle.
  if (!KBReloaded.has_specialization_constant<SC>()) {
    std::cout
        << "FAIL: reloaded bundle does not advertise the spec const at all. "
        << "ext_oneapi_get_content lost the [SYCL/specialization "
        << "constants] property set on serialization.\n";
    return 1;
  }

  const int Got = KBReloaded.get_specialization_constant<SC>();
  if (Got == UserValue) {
    std::cout << "OK: round-tripped spec const value " << Got << "\n";
    return 0;
  }
  if (Got == DefaultValue) {
    std::cout << "FAIL: round-tripped spec const reverted to default "
              << "(got " << Got << ", expected " << UserValue << "). "
              << "The user-set override did not survive the SYCLBIN "
              << "round-trip.\n";
    return 1;
  }
  std::cout << "FAIL: unexpected spec const value " << Got << " (expected "
            << UserValue << ")\n";
  return 1;
}
