// REQUIRES: target-spir

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out %S/../Graph/Inputs/Kernels/update_with_indices_accessor.spv

// Kernels loaded from SPIR-V have no compile-time signature, so set_arg cannot
// convert a value to the parameter type. Check that a by-value argument whose
// size does not match the kernel parameter is diagnosed before launch.

#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>

#include <cassert>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace syclex = sycl::ext::oneapi::experimental;

int main(int, char **argv) {
  sycl::queue Q;
  std::ifstream SpvStream(argv[1], std::ios::binary);
  std::vector<char> Bytes{std::istreambuf_iterator<char>(SpvStream), {}};
  std::vector<std::byte> Spv(Bytes.size());
  std::memcpy(Spv.data(), Bytes.data(), Bytes.size());
  auto Source = syclex::create_kernel_bundle_from_source(
      Q.get_context(), syclex::source_language::spirv, Spv);
  auto Bundle = syclex::build(Source);
  // Writes I to Acc[I] for I < 1024; parameter 1 is a by-value sycl::id<1>.
  sycl::kernel K = Bundle.ext_oneapi_get_kernel(
      "_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_EUlvE_");

  constexpr size_t N = 1024;
  std::vector<int> Data(N, -1);
  sycl::buffer<int> Buf{Data.data(), sycl::range<1>{N}};
  auto Launch = [&](auto Offset) {
    Q.submit([&](sycl::handler &CGH) {
       CGH.set_arg(0, Buf.get_access(CGH));
       CGH.set_arg(1, Offset);
       CGH.single_task(K);
     }).wait();
  };

  try {
    Launch(0); // int, 4 bytes
    assert(false && "expected a size mismatch exception");
  } catch (const sycl::exception &E) {
    assert(E.code() == sycl::errc::invalid);
    assert(std::string{E.what()}.find("the kernel expects 8 bytes but 4 bytes "
                                      "were provided") != std::string::npos);
  }

  Launch(sycl::id<1>{0});
  sycl::host_accessor Out{Buf};
  for (size_t I = 0; I < N; ++I)
    assert(Out[I] == static_cast<int>(I));
  return 0;
}
