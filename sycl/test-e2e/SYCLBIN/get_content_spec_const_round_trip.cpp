
// REQUIRES: aspect-usm_shared_allocations

// -- End-to-end round-trip test for ext_oneapi_get_content() preserving
// -- user-set specialization constant values.
// --
// -- Flow:
// --   1. Build an input-state kernel_bundle. The kernel below references
// --      two spec constants, so they are statically registered with the
// --      image.
// --   2. Override the spec constants via set_specialization_constant<>:
// --      SCA is set to a user value; SCB is set then reset back to its
// --      default. This exercises the default-vs-set distinction across more
// --      than one constant.
// --   3. sycl::build it to executable; the overrides are applied at JIT time.
// --   4. ext_oneapi_get_content() to serialize.
// --   5. Reload the bytes as an executable kernel_bundle.
// --   6. Read the spec const values back via the *bundle* host API on the
// --      reloaded bundle. SCA must equal the user value (12345); SCB must
// --      read back as its default (7), not the stale intermediate value, and
// --      the two constants' blob slices must not be swapped.
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

constexpr static int DefaultA = 42;
constexpr static int UserA = 12345;
constexpr static int DefaultB = 7;
constexpr static int StaleB = 99;

constexpr sycl::specialization_id<int> SCA{DefaultA};
constexpr sycl::specialization_id<int> SCB{DefaultB};

// A trivial kernel that references the spec constants at compile time so the
// frontend ties them to a registered device image. We do not need to actually
// run this kernel against the reloaded bundle to verify the round-trip.
class ReadSpecConstKernel;

int main() {
  sycl::queue Q;
  const sycl::context Ctx = Q.get_context();
  const sycl::device Dev = Q.get_device();

  // Touch the kernel once at compile-time so the spec consts are associated
  // with a registered image.
  {
    int *Sink = sycl::malloc_shared<int>(2, Q);
    Q.submit([&](sycl::handler &CGH) {
       CGH.single_task<ReadSpecConstKernel>([=](sycl::kernel_handler KH) {
         Sink[0] = KH.get_specialization_constant<SCA>();
         Sink[1] = KH.get_specialization_constant<SCB>();
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

  // 2. Override the spec constants. SCB is set to a stale value then reset to
  // its default; after that it must report as unset (default). SCA is set to
  // a distinct user value.
  if (KBInput.get_specialization_constant<SCA>() != DefaultA ||
      KBInput.get_specialization_constant<SCB>() != DefaultB) {
    std::cout << "Pre-condition failed: input bundle did not start with the "
                 "default spec const values.\n";
    return 1;
  }
  KBInput.set_specialization_constant<SCB>(StaleB);
  KBInput.set_specialization_constant<SCB>(DefaultB);
  KBInput.set_specialization_constant<SCA>(UserA);

  if (KBInput.get_specialization_constant<SCA>() != UserA) {
    std::cout << "Pre-condition failed: host read of input bundle SCA did not "
                 "return user value after set.\n";
    return 1;
  }
  if (KBInput.get_specialization_constant<SCB>() != DefaultB) {
    std::cout << "Pre-condition failed: SCB on input did not short-circuit to "
                 "default; got "
              << KBInput.get_specialization_constant<SCB>() << "\n";
    return 1;
  }

  // 3. Build to executable. The overrides are applied to the UR program here.
  auto KBExe = sycl::build(KBInput);

  if (KBExe.get_specialization_constant<SCA>() != UserA) {
    std::cout << "Pre-condition failed: host read of executable bundle SCA did "
                 "not return user value after build.\n";
    return 1;
  }
  if (KBExe.get_specialization_constant<SCB>() != DefaultB) {
    std::cout << "Pre-condition failed: SCB on exe did not short-circuit to "
                 "default; got "
              << KBExe.get_specialization_constant<SCB>() << "\n";
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

  // 6. Compare host-side spec const values on the reloaded bundle.
  if (!KBReloaded.has_specialization_constant<SCA>() ||
      !KBReloaded.has_specialization_constant<SCB>()) {
    std::cout
        << "FAIL: reloaded bundle does not advertise the spec consts at all. "
        << "ext_oneapi_get_content lost the [SYCL/specialization "
        << "constants] property set on serialization.\n";
    return 1;
  }

  const int GotA = KBReloaded.get_specialization_constant<SCA>();
  const int GotB = KBReloaded.get_specialization_constant<SCB>();

  std::cout << "After round-trip:\n"
            << "  SCA = " << GotA << "  (expected " << UserA << ")\n"
            << "  SCB = " << GotB << "  (expected " << DefaultB << ")\n";

  int Failed = 0;
  if (GotA != UserA) {
    std::cout << "FAIL: SCA round-tripped to " << GotA << "; expected " << UserA
              << ". The user-set override did not survive the "
                 "SYCLBIN round-trip.\n";
    ++Failed;
  }
  if (GotB != DefaultB) {
    std::cout << "FAIL: SCB read back as " << GotB << "; expected " << DefaultB
              << ". A reset-to-default const must not round-trip as set, and "
                 "the two constants' blob slices must not be swapped.\n";
    ++Failed;
  }
  if (!Failed)
    std::cout << "OK: round-tripped spec const values\n";
  return Failed;
}
