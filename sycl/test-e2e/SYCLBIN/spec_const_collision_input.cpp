// REQUIRES: aspect-usm_shared_allocations

// UNSUPPORTED: cuda, hip
// UNSUPPORTED-INTENDED: CUDA and HIP targets produce only native device
// binaries and can therefore not produce input-state SYCLBIN files.

// -- Regression test for CMPLRLLVM-77316: on the -fsyclbin=input path,
// -- set_specialization_constant<SC_A>(v) must take effect even when v equals
// -- another referenced constant's default. Setting SC_A=1024 (== SC_B default)
// -- once dropped SC_A to its own default 256; it must yield 1024.

// RUN: %clangxx --offload-new-driver -fsyclbin=input %{sycl_target_opts} %S/Inputs/spec_const_collision_kernel.cpp -o %t.syclbin
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out %t.syclbin

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/syclbin_kernel_bundle.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/specialization_id.hpp>
#include <sycl/usm.hpp>

#include <cassert>
#include <iostream>

namespace syclexp = sycl::ext::oneapi::experimental;

inline constexpr sycl::specialization_id<int> SC_A{256};
inline constexpr sycl::specialization_id<int> SC_B{1024};

static constexpr int DefaultA = 256;
static constexpr int DefaultB = 1024;

// Sets SC_A=ValueA, SC_B=DefaultB on the input bundle, builds, launches,
// returns (out[0], out[1]).
static std::pair<int, int> runWithSCA(sycl::queue &Q, const char *Path,
                                      int ValueA) {
  const sycl::context Ctx = Q.get_context();

  auto KBInput = syclexp::get_kernel_bundle<sycl::bundle_state::input>(
      Ctx, std::string{Path});
  KBInput.set_specialization_constant<SC_A>(ValueA);
  KBInput.set_specialization_constant<SC_B>(DefaultB);

  auto KBExe = sycl::build(KBInput);
  sycl::kernel Kern = KBExe.ext_oneapi_get_kernel("spec_const_collision");

  int *Out = sycl::malloc_shared<int>(2, Q);
  Out[0] = Out[1] = -1;
  Q.submit([&](sycl::handler &CGH) {
     CGH.use_kernel_bundle(KBExe);
     CGH.set_args(Out);
     CGH.parallel_for(sycl::nd_range<1>{{1}, {1}}, Kern);
   }).wait_and_throw();

  std::pair<int, int> Result{Out[0], Out[1]};
  sycl::free(Out, Q);
  return Result;
}

int main(int argc, char **argv) {
  assert(argc == 2);
  sycl::queue Q;

  int Failed = 0;

  // The regression case: SC_A set to SC_B's default. Must see A == 1024.
  {
    auto [A, B] = runWithSCA(Q, argv[1], DefaultB);
    std::cout << "SC_A=1024 (== SC_B default): A=" << A << " B=" << B << "\n";
    if (A != DefaultB) {
      std::cout << "FAIL: SC_A was dropped to " << A << "; expected "
                << DefaultB << " (CMPLRLLVM-77316).\n";
      ++Failed;
    }
    if (B != DefaultB) {
      std::cout << "FAIL: SC_B = " << B << "; expected " << DefaultB << "\n";
      ++Failed;
    }
  }

  // Control: a value that collides with nothing.
  {
    auto [A, B] = runWithSCA(Q, argv[1], 777);
    std::cout << "SC_A=777: A=" << A << " B=" << B << "\n";
    if (A != 777 || B != DefaultB) {
      std::cout << "FAIL: expected A=777 B=1024\n";
      ++Failed;
    }
  }

  // Control: SC_A set to its own default.
  {
    auto [A, B] = runWithSCA(Q, argv[1], DefaultA);
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
