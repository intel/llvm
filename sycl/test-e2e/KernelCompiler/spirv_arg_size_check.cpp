// RUN: %{build} -o %t.out
// RUN: %{run} %t.out %S/Kernels/kernels.spv

// Kernels loaded from SPIR-V have no compile-time parameter information, so
// set_arg cannot convert values to the parameter type. Check that a value whose
// size differs from the SPIR-V parameter size is rejected at submission instead
// of the kernel reading garbage, and that correctly sized values still work.

#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/usm.hpp>

#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

namespace syclex = sycl::ext::oneapi::experimental;

static sycl::kernel_bundle<sycl::bundle_state::executable>
loadKernelsFromFile(sycl::queue &Q, const std::string &FileName) {
  std::ifstream Stream(FileName, std::ios::binary);
  Stream.seekg(0, std::ios::end);
  size_t Size = Stream.tellg();
  Stream.seekg(0);
  std::vector<std::byte> Spv(Size);
  Stream.read(reinterpret_cast<char *>(Spv.data()), Size);

  auto Src = syclex::create_kernel_bundle_from_source(
      Q.get_context(), syclex::source_language::spirv, Spv);
  return syclex::build(Src);
}

template <typename F>
static void expectArgSizeError(F &&Submit, const std::string &Expected) {
  try {
    Submit();
  } catch (const sycl::exception &E) {
    std::cout << E.what() << std::endl;
    assert(E.code() == sycl::errc::kernel_argument);
    assert(std::string(E.what()).find(Expected) != std::string::npos);
    return;
  }
  assert(false && "expected an errc::kernel_argument exception");
}

// OpTypeInt32 takes (int32_t a, int32_t *b, global int32_t *out,
// local int32_t *tmp) and computes a * a + b[0] * b[0].
static void testScalarParam(sycl::queue &Q, const sycl::kernel &K) {
  int32_t *B = sycl::malloc_shared<int32_t>(1, Q);
  *B = 5;
  int32_t Output = 0;
  {
    sycl::buffer OutBuf(&Output, sycl::range<1>(1));
    auto Submit = [&](auto A) {
      Q.submit([&](sycl::handler &CGH) {
        CGH.set_arg(0, A);
        CGH.set_arg(1, B);
        CGH.set_arg(2, OutBuf.get_access<sycl::access_mode::write>(CGH));
        CGH.set_arg(3, sycl::local_accessor<int32_t, 1>(1, CGH));
        CGH.parallel_for(sycl::range<1>{1}, K);
      });
    };

    expectArgSizeError([&] { Submit(int64_t{2}); },
                       "argument 0 of kernel 'OpTypeInt32' is 8 bytes, but "
                       "the kernel expects 4 bytes");
    expectArgSizeError([&] { Submit(int16_t{2}); },
                       "argument 0 of kernel 'OpTypeInt32' is 2 bytes");
    Submit(int32_t{2});
  }
  assert(Output == 2 * 2 + 5 * 5);
  sycl::free(B, Q);
}

// OpTypeStruct takes a struct by value and a pointer to the same struct type.
static void testStructParam(sycl::queue &Q, const sycl::kernel &K) {
  // This definition must match the one used in the kernel.
  struct S {
    int32_t i;
    float f;
    int32_t *p;
    struct Inner {
      int32_t i;
      float f;
      int32_t *p;
    } inner;
  };
  // The leading members of S without the nested struct.
  struct Truncated {
    int32_t i;
    float f;
    int32_t *p;
  };

  int32_t *InP = sycl::malloc_shared<int32_t>(2, Q);
  int32_t *OutP = sycl::malloc_shared<int32_t>(2, Q);
  S *Output = sycl::malloc_shared<S>(1, Q);
  InP[0] = 3;
  InP[1] = 6;
  *Output = S{0, 0, &OutP[0], S::Inner{0, 0, &OutP[1]}};

  expectArgSizeError(
      [&] {
        Q.submit([&](sycl::handler &CGH) {
          CGH.set_arg(0, Truncated{1, 2.0f, &InP[0]});
          CGH.set_arg(1, Output);
          CGH.parallel_for(sycl::range<1>{1}, K);
        });
      },
      "argument 0 of kernel 'OpTypeStruct' is 16 bytes, but the kernel "
      "expects 32 bytes");

  Q.submit([&](sycl::handler &CGH) {
     CGH.set_arg(0, S{1, 2.0f, &InP[0], S::Inner{4, 5.0f, &InP[1]}});
     CGH.set_arg(1, Output);
     CGH.parallel_for(sycl::range<1>{1}, K);
   }).wait();
  assert(Output->i == 2 && Output->inner.i == 8);
  assert(OutP[0] == 6 && OutP[1] == 12);

  sycl::free(InP, Q);
  sycl::free(OutP, Q);
  sycl::free(Output, Q);
}

int main(int argc, char **argv) {
  assert(argc == 2 && "Usage: ./%t.out <kernel-spv-file>");
  sycl::queue Q;
  if (!Q.get_device().ext_oneapi_can_build(syclex::source_language::spirv)) {
    std::cout << "Device cannot build SPIR-V, skipping" << std::endl;
    return 0;
  }

  auto Bundle = loadKernelsFromFile(Q, argv[1]);
  testScalarParam(Q, Bundle.ext_oneapi_get_kernel("OpTypeInt32"));
  testStructParam(Q, Bundle.ext_oneapi_get_kernel("OpTypeStruct"));
  return 0;
}
