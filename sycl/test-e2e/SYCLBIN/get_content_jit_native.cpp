// REQUIRES: aspect-usm_device_allocations

// -- End-to-end test that ext_oneapi_get_content() on an executable-state
// -- kernel_bundle whose source image is SPIR-V emits native device code
// -- images (extracted from the JIT-built UR program) rather than re-emitting
// -- the original SPIR-V. Re-loading the resulting bytes and running the
// -- kernel must work without throwing UR_RESULT_ERROR_INVALID_BINARY and
// -- without re-JIT'ing.

// RUN: %clangxx --offload-new-driver -fsyclbin=input %{sycl_target_opts} %S/Inputs/basic_kernel.cpp -o %t.syclbin
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out %t.syclbin

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/syclbin_kernel_bundle.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/usm.hpp>

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

namespace syclexp = sycl::ext::oneapi::experimental;

static constexpr size_t NUM = 1024;
static constexpr size_t WGSIZE = 16;
static constexpr float EPS = 0.001f;

// Strip OffloadBinary v2 wrapper and return raw SYCLBIN bytes.
std::pair<const char *, size_t>
stripOffloadWrapper(const std::vector<char> &Bytes) {
  struct Hdr {
    uint8_t Magic[4];
    uint32_t Version;
    uint64_t Size;
    uint64_t EntriesOffset;
    uint64_t EntriesCount;
  };
  struct Entry {
    uint16_t ImageKind;
    uint16_t OffloadKind;
    uint32_t Flags;
    uint64_t StringOffset;
    uint64_t NumStrings;
    uint64_t ImageOffset;
    uint64_t ImageSize;
  };
  assert(Bytes.size() >= sizeof(Hdr));
  const auto *H = reinterpret_cast<const Hdr *>(Bytes.data());
  const auto *E =
      reinterpret_cast<const Entry *>(Bytes.data() + H->EntriesOffset);
  return {Bytes.data() + E->ImageOffset, static_cast<size_t>(E->ImageSize)};
}

// SYCLBIN file header layout matches sycl/source/detail/syclbin.hpp.
struct SYCLBINFileHeader {
  uint32_t Magic;
  uint32_t Version;
  uint32_t AbstractModuleCount;
  uint32_t IRModuleCount;
  uint32_t NativeDeviceCodeImageCount;
  uint64_t MetadataByteTableSize;
  uint64_t BinaryByteTableSize;
  uint64_t GlobalMetadataOffset;
  uint64_t GlobalMetadataSize;
};

int main(int argc, char *argv[]) {
  assert(argc == 2);

  sycl::queue Q;
  const sycl::context Ctx = Q.get_context();
  const std::vector<sycl::device> Devs{Q.get_device()};

  // Load the input-state SYCLBIN, then build to executable. After build, the
  // bundle's underlying device images still point at SPIR-V, but the JIT
  // result lives in the UR program. ext_oneapi_get_content must extract those
  // native bytes.
  auto KBInput = syclexp::get_kernel_bundle<sycl::bundle_state::input>(
      Ctx, std::string{argv[1]});
  auto KBExe = sycl::build(KBInput);

  std::vector<char> Bytes = KBExe.ext_oneapi_get_content();
  if (Bytes.empty()) {
    std::cout << "ext_oneapi_get_content returned empty vector\n";
    return 1;
  }

  // Inspect the SYCLBIN file header to confirm we have at least one native
  // image and zero IR modules. This is the strong assertion for the spec
  // requirement that an executable-state SYCLBIN contain native binaries.
  auto [Raw, Size] = stripOffloadWrapper(Bytes);
  assert(Size >= sizeof(SYCLBINFileHeader));
  SYCLBINFileHeader FH{};
  std::memcpy(&FH, Raw, sizeof(FH));

  if (FH.NativeDeviceCodeImageCount == 0) {
    std::cout << "Executable-state SYCLBIN unexpectedly has 0 native images\n";
    return 1;
  }
  if (FH.IRModuleCount != 0) {
    std::cout << "Executable-state SYCLBIN unexpectedly carries "
              << FH.IRModuleCount << " IR module(s); expected 0\n";
    return 1;
  }

  // Re-load the serialized bytes and run the kernel from the re-loaded bundle.
  auto KBReloaded = syclexp::get_kernel_bundle<sycl::bundle_state::executable>(
      Ctx, Devs, sycl::span<char>{Bytes});

  if (!KBReloaded.ext_oneapi_has_kernel("iota")) {
    std::cout << "Re-loaded kernel_bundle does not contain expected kernel "
                 "\"iota\"\n";
    return 1;
  }

  sycl::kernel IotaKern = KBReloaded.ext_oneapi_get_kernel("iota");

  float *Ptr = sycl::malloc_shared<float>(NUM, Q);
  Q.submit([&](sycl::handler &CGH) {
     CGH.set_args(3.14f, Ptr);
     CGH.parallel_for(sycl::nd_range{{NUM}, {WGSIZE}}, IotaKern);
   }).wait_and_throw();

  int Failed = 0;
  for (size_t I = 0; I < NUM; ++I) {
    const float Truth = 3.14f + static_cast<float>(I);
    if (std::abs(Ptr[I] - Truth) > EPS) {
      std::cout << "Result[" << I << "] = " << Ptr[I] << ", expected " << Truth
                << "\n";
      ++Failed;
    }
  }
  sycl::free(Ptr, Q);
  return Failed;
}
