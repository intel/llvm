// UNSUPPORTED: cuda, hip
// UNSUPPORTED-INTENDED: CUDA and HIP targets produce only native device
// binaries and can therefore not produce input-state SYCLBIN files.

// -- Regression test for CMPLRLLVM-77316: on the -fsyclbin=input path,
// -- set_specialization_constant<SC_A>(v) must take effect even when v equals
// -- another referenced constant's default. Setting SC_A=1024 (== SC_B default)
// -- once dropped SC_A to its own default 256; it must resolve to 1024. Checked
// -- host-side after build(), which exercises the same blob-offset resolution.

// RUN: %clangxx --offload-new-driver -fsyclbin=input %{sycl_target_opts} %S/Inputs/spec_const_collision_kernel.cpp -o %t.syclbin
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out %t.syclbin

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/syclbin_kernel_bundle.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/specialization_id.hpp>

#include <cassert>
#include <iostream>

namespace syclexp = sycl::ext::oneapi::experimental;

inline constexpr sycl::specialization_id<int> SC_A{256};
inline constexpr sycl::specialization_id<int> SC_B{1024};

static constexpr int DefaultA = 256;
static constexpr int DefaultB = 1024;

// Sets SC_A=ValueA, SC_B=DefaultB on the input bundle, builds, returns the
// executable bundle's resolved (SC_A, SC_B).
static std::pair<int, int> resolveWithSCA(sycl::queue &Q, const char *Path,
                                          int ValueA) {
  const sycl::context Ctx = Q.get_context();
  auto KBInput = syclexp::get_kernel_bundle<sycl::bundle_state::input>(
      Ctx, std::string{Path});
  KBInput.set_specialization_constant<SC_A>(ValueA);
  KBInput.set_specialization_constant<SC_B>(DefaultB);
  auto KBExe = sycl::build(KBInput);
  return {KBExe.get_specialization_constant<SC_A>(),
          KBExe.get_specialization_constant<SC_B>()};
}

int main(int argc, char **argv) {
  assert(argc == 2);
  sycl::queue Q;
  int Failed = 0;

  // Regression case: SC_A set to SC_B's default must stay 1024, not drop to
  // 256.
  {
    auto [A, B] = resolveWithSCA(Q, argv[1], DefaultB);
    std::cout << "SC_A=1024 (== SC_B default): A=" << A << " B=" << B << "\n";
    if (A != DefaultB || B != DefaultB) {
      std::cout << "FAIL: expected A=1024 B=1024 (CMPLRLLVM-77316).\n";
      ++Failed;
    }
  }

  // Control: value colliding with nothing.
  {
    auto [A, B] = resolveWithSCA(Q, argv[1], 777);
    std::cout << "SC_A=777: A=" << A << " B=" << B << "\n";
    if (A != 777 || B != DefaultB) {
      std::cout << "FAIL: expected A=777 B=1024\n";
      ++Failed;
    }
  }

  // Control: SC_A set to its own default.
  {
    auto [A, B] = resolveWithSCA(Q, argv[1], DefaultA);
    std::cout << "SC_A=256 (own default): A=" << A << " B=" << B << "\n";
    if (A != DefaultA || B != DefaultB) {
      std::cout << "FAIL: expected A=256 B=1024\n";
      ++Failed;
    }
  }

  if (!Failed)
    std::cout << "OK\n";
  return Failed;
}
