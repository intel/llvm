//==--- sycl.cpp --- kernel_compiler extension tests -----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: (opencl || level_zero)
// REQUIRES: aspect-usm_device_allocations

// UNSUPPORTED: accelerator
// UNSUPPORTED-INTENDED: while accelerator is AoT only, this cannot run there.

// UNSUPPORTED: windows && arch-intel_gpu_bmg_g21
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/17255

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %{l0_leak_check} %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/usm.hpp>

auto constexpr AddEmH = R"===(
  int AddEm(int a, int b){
    return a + b + 5;
  }
)===";

auto constexpr PlusEmH = R"===(
  int PlusEm(int a, int b){
    return a + b + 6;
  }
)===";

// TODO: remove SYCL_EXTERNAL once it is no longer needed.
auto constexpr SYCLSource = R"===(
#include <sycl/sycl.hpp>
#include "intermediate/AddEm.h"
#include "intermediate/PlusEm.h"

// use extern "C" to avoid name mangling
extern "C" SYCL_EXTERNAL SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((sycl::ext::oneapi::experimental::nd_range_kernel<1>))
void ff_cp(int *ptr, int *unused) {

  // intentionally using deprecated routine, as opposed to this_work_item::get_nd_item<1>()
  sycl::nd_item<1> Item = sycl::ext::oneapi::experimental::this_nd_item<1>();

  sycl::id<1> GId = Item.get_global_id();
  ptr[GId.get(0)] = AddEm(GId.get(0), 37);
}

// this name will be mangled
template <typename T>
SYCL_EXTERNAL SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((sycl::ext::oneapi::experimental::nd_range_kernel<1>))
void ff_templated(T *ptr, T *unused) {

  sycl::nd_item<1> Item = sycl::ext::oneapi::this_work_item::get_nd_item<1>();

  sycl::id<1> GId = Item.get_global_id();
  ptr[GId.get(0)] = PlusEm(GId.get(0), 38);
}
)===";

auto constexpr SYCLSource2 = R"""(
#include <sycl/sycl.hpp>

extern "C" SYCL_EXTERNAL 
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((sycl::ext::oneapi::experimental::nd_range_kernel<1>))
void vec_add(float* in1, float* in2, float* out){
  size_t id = sycl::ext::oneapi::this_work_item::get_nd_item<1>().get_global_linear_id();
  out[id] = in1[id] + in2[id];
}
)""";

auto constexpr ESIMDSource = R"===(
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

using namespace sycl::ext::intel::esimd;

constexpr int VL = 16;

extern "C" SYCL_EXTERNAL SYCL_ESIMD_KERNEL SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((sycl::ext::oneapi::experimental::nd_range_kernel<1>))
void vector_add_esimd(float *A, float *B, float *C) {
    sycl::nd_item<1> item = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    unsigned int i = item.get_global_id(0);
    unsigned int offset = i * VL ;

    simd<float, VL> va(A + offset);
    simd<float, VL> vb(B + offset);
    simd<float, VL> vc = va + vb;
    vc.copy_to(C + offset);
    }
)===";

auto constexpr DeviceCodeSplitSource = R"===(
#include <sycl/sycl.hpp>

template<typename T, unsigned WG = 16> SYCL_EXTERNAL 
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(sycl::ext::oneapi::experimental::nd_range_kernel<1>)
[[sycl::reqd_work_group_size(WG)]]
void vec_add(T* in1, T* in2, T* out){
  size_t id = sycl::ext::oneapi::this_work_item::get_nd_item<1>().get_global_linear_id();
  out[id] = in1[id] + in2[id];
}
)===";

auto constexpr DeviceLibrariesSource = R"===(
#include <sycl/sycl.hpp>
#include <cmath>
#include <complex>
#include <sycl/ext/intel/math.hpp>

extern "C" SYCL_EXTERNAL 
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(sycl::ext::oneapi::experimental::single_task_kernel)
void device_libs_kernel(float *ptr) {
  // Extension list: llvm/lib/SYCLLowerIR/SYCLDeviceLibReqMask.cpp

  // cl_intel_devicelib_assert is not available for opencl:gpu; skip testing it.
  // Only test the fp32 variants of complex, math and imf to keep this test
  // device-agnostic.
  
  // cl_intel_devicelib_math
  ptr[0] = erff(ptr[0]);

  // cl_intel_devicelib_complex
  ptr[1] = std::abs(std::complex<float>{1.0f, ptr[1]});

  // cl_intel_devicelib_cstring
  ptr[2] = memcmp(ptr + 2, ptr + 2, sizeof(float));

  // cl_intel_devicelib_imf
  ptr[3] = sycl::ext::intel::math::sqrt(ptr[3] * 2);

  // cl_intel_devicelib_imf_bf16
  ptr[4] = sycl::ext::intel::math::float2bfloat16(ptr[4] * 0.5f);

  // cl_intel_devicelib_bfloat16
  ptr[5] = sycl::ext::oneapi::bfloat16{ptr[5] / 0.25f};
}
)===";

auto constexpr BadSource = R"===(
#include <sycl/sycl.hpp>

extern "C" SYCL_EXTERNAL SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((sycl::ext::oneapi::experimental::nd_range_kernel<1>))
void ff_cp(int *ptr) {

  sycl::nd_item<1> Item = sycl::ext::oneapi::this_work_item::get_nd_item<1>();

  sycl::id<1> GId = Item.get_global_id() + no semi colon !!
  ptr[GId.get(0)] = GId.get(0) + 41;
}
)===";

auto constexpr WarningSource = R"===(
#include <sycl/sycl.hpp>

extern "C" SYCL_EXTERNAL SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((sycl::ext::oneapi::experimental::nd_range_kernel<1>))
void ff_cp(int *ptr) {

  // intentionally using deprecated routine, as opposed to this_work_item::get_nd_item<1>()
  // to provoke a warning.
  sycl::nd_item<1> Item = sycl::ext::oneapi::experimental::this_nd_item<1>();

  sycl::id<1> GId = Item.get_global_id();
  ptr[GId.get(0)] = GId.get(0) + 41;
}
)===";

namespace syclex = sycl::ext::oneapi::experimental;

void run_1(sycl::queue &Queue, sycl::kernel &Kernel, int seed) {
  constexpr int Range = 10;
  int *usmPtr = sycl::malloc_shared<int>(Range, Queue);
  int start = 3;

  sycl::nd_range<1> R1{{Range}, {1}};

  bool Passa = true;

  memset(usmPtr, 0, Range * sizeof(int));
  Queue.submit([&](sycl::handler &Handler) {
    Handler.set_arg(0, usmPtr);
    Handler.set_arg(1, usmPtr);
    Handler.parallel_for(R1, Kernel);
  });
  Queue.wait();

  for (int i = 0; i < Range; i++) {
    std::cout << usmPtr[i] << "=" << (i + seed) << " ";
    assert(usmPtr[i] == i + seed);
  }
  std::cout << std::endl;

  sycl::free(usmPtr, Queue);
}

void run_2(sycl::queue &Queue, sycl::kernel &Kernel, bool ESIMD, float seed) {
  constexpr int VL = 16; // this constant also in ESIMDSource string.
  constexpr int size = VL * 16;

  float *A = sycl::malloc_shared<float>(size, Queue);
  float *B = sycl::malloc_shared<float>(size, Queue);
  float *C = sycl::malloc_shared<float>(size, Queue);
  for (size_t i = 0; i < size; i++) {
    A[i] = seed;
    B[i] = seed * 2.0f;
    C[i] = 0.0f;
  }
  sycl::range<1> GlobalRange(size / (ESIMD ? VL : 1));
  sycl::range<1> LocalRange(ESIMD ? 1 : VL);
  sycl::nd_range<1> NDRange{GlobalRange, LocalRange};

  Queue
      .submit([&](sycl::handler &Handler) {
        Handler.set_arg(0, A);
        Handler.set_arg(1, B);
        Handler.set_arg(2, C);
        Handler.parallel_for(NDRange, Kernel);
      })
      .wait();

  // Check.
  for (size_t i = 0; i < size; i++) {
    assert(C[i] == seed * 3.0f);
  }

  sycl::free(A, Queue);
  sycl::free(B, Queue);
  sycl::free(C, Queue);
}

int test_build_and_run(sycl::queue q) {
  namespace syclex = sycl::ext::oneapi::experimental;
  using source_kb = sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source>;
  using exe_kb = sycl::kernel_bundle<sycl::bundle_state::executable>;

  sycl::context ctx = q.get_context();

  // Create from source.
  syclex::include_files incFiles{"intermediate/AddEm.h", AddEmH};
  incFiles.add("intermediate/PlusEm.h", PlusEmH);
  source_kb kbSrc = syclex::create_kernel_bundle_from_source(
      ctx, syclex::source_language::sycl, SYCLSource,
      syclex::properties{incFiles});

  // Double check kernel_bundle.get_source() / get_backend().
  sycl::context ctxRes = kbSrc.get_context();
  assert(ctxRes == ctx);
  sycl::backend beRes = kbSrc.get_backend();
  assert(beRes == ctx.get_backend());

  // Compilation of empty prop list, no devices.
  exe_kb kbExe1 = syclex::build(kbSrc);

  // Compilation with props and devices
  std::string log;
  std::vector<std::string> flags{"-fno-fast-math",
                                 "-fsycl-instrument-device-code"};
  std::vector<sycl::device> devs = kbSrc.get_devices();
  exe_kb kbExe2 = syclex::build(
      kbSrc, devs,
      syclex::properties{syclex::build_options{flags}, syclex::save_log{&log},
                         syclex::registered_names{"ff_templated<int>"}});

  // extern "C" was used, so the name "ff_cp" is implicitly known.
  sycl::kernel k = kbExe2.ext_oneapi_get_kernel("ff_cp");

  // The templated function name was registered.
  sycl::kernel k2 = kbExe2.ext_oneapi_get_kernel("ff_templated<int>");

  // Get compiler-generated names.
  std::string cgn = kbExe2.ext_oneapi_get_raw_kernel_name("ff_cp");
  std::string cgn2 = kbExe2.ext_oneapi_get_raw_kernel_name("ff_templated<int>");
  assert(cgn == "__sycl_kernel_ff_cp");
  assert(cgn2 == "_Z26__sycl_kernel_ff_templatedIiEvPT_S1_");

  // We can also use the compiler-generated names directly.
  assert(kbExe2.ext_oneapi_has_kernel(cgn));
  assert(kbExe2.ext_oneapi_has_kernel(cgn2));

  // Test the kernels.
  run_1(q, k, 37 + 5);  // ff_cp seeds 37. AddEm will add 5 more.
  run_1(q, k2, 38 + 6); // ff_templated seeds 38. PlusEm adds 6 more.

  // Create and compile new bundle with different header.
  std::string AddEmHModified = AddEmH;
  AddEmHModified[AddEmHModified.find('5')] = '7';
  syclex::include_files incFiles2{"intermediate/AddEm.h", AddEmHModified};
  incFiles2.add("intermediate/PlusEm.h", PlusEmH);
  source_kb kbSrc2 = syclex::create_kernel_bundle_from_source(
      ctx, syclex::source_language::sycl, SYCLSource,
      syclex::properties{incFiles2});

  exe_kb kbExe3 = syclex::build(kbSrc2);
  sycl::kernel k3 = kbExe3.ext_oneapi_get_kernel("ff_cp");
  run_1(q, k3, 37 + 7);

  // Can we still run the original compilation?
  sycl::kernel k4 = kbExe1.ext_oneapi_get_kernel("ff_cp");
  run_1(q, k4, 37 + 5);

  return 0;
}

int test_device_code_split(sycl::queue q) {
  namespace syclex = sycl::ext::oneapi::experimental;
  using source_kb = sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source>;
  using exe_kb = sycl::kernel_bundle<sycl::bundle_state::executable>;

  sycl::context ctx = q.get_context();

  source_kb kbSrc = syclex::create_kernel_bundle_from_source(
      ctx, syclex::source_language::sycl, DeviceCodeSplitSource);

  // Test explicit device code split
  std::vector<std::string> names{"vec_add<float>", "vec_add<int>",
                                 "vec_add<short>"};
  auto build = [&](const std::string &mode) -> size_t {
    exe_kb kbExe = syclex::build(
        kbSrc, syclex::properties{
                   syclex::registered_names{names},
                   syclex::build_options{"-fsycl-device-code-split=" + mode}});
    return std::distance(kbExe.begin(), kbExe.end());
  };

  size_t perKernelNImg = build("per_kernel");
  size_t perSourceNImg = build("per_source");
  size_t offNImg = build("off");
  size_t autoNImg = build("auto");

  assert(perKernelNImg == 3);
  assert(perSourceNImg == 1);
  assert(offNImg == 1);
  assert(autoNImg >= offNImg && autoNImg <= perKernelNImg);

  // Test implicit device code split
  names = {"vec_add<float, 8>", "vec_add<float, 16>"};
  exe_kb kbDiffWorkGroupSizes =
      syclex::build(kbSrc, syclex::properties{syclex::registered_names{names}});
  assert(std::distance(kbDiffWorkGroupSizes.begin(),
                       kbDiffWorkGroupSizes.end()) == 2);

  return 0;
}

int test_device_libraries(sycl::queue q) {
  namespace syclex = sycl::ext::oneapi::experimental;
  using source_kb = sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source>;
  using exe_kb = sycl::kernel_bundle<sycl::bundle_state::executable>;

  sycl::context ctx = q.get_context();

  source_kb kbSrc = syclex::create_kernel_bundle_from_source(
      ctx, syclex::source_language::sycl, DeviceLibrariesSource);
  exe_kb kbExe = syclex::build(kbSrc);

  sycl::kernel k = kbExe.ext_oneapi_get_kernel("device_libs_kernel");
  constexpr size_t nElem = 6;
  float *ptr = sycl::malloc_shared<float>(nElem, q);
  for (int i = 0; i < nElem; ++i)
    ptr[i] = 1.0f;

  q.submit([&](sycl::handler &cgh) {
    cgh.set_arg(0, ptr);
    cgh.single_task(k);
  });
  q.wait_and_throw();

  // Check that the kernel was executed. Given the {1.0, ..., 1.0} input,
  // the expected result is approximately {0.84, 1.41, 0.0, 1.41, 0.5, 4.0}.
  for (unsigned i = 0; i < nElem; ++i) {
    std::cout << ptr[i] << ' ';
    assert(ptr[i] != 1.0f);
  }
  std::cout << std::endl;

  sycl::free(ptr, q);

  return 0;
}

int test_esimd(sycl::queue q) {
  namespace syclex = sycl::ext::oneapi::experimental;
  using source_kb = sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source>;
  using exe_kb = sycl::kernel_bundle<sycl::bundle_state::executable>;

  sycl::context ctx = q.get_context();

  if (!q.get_device().has(sycl::aspect::ext_intel_esimd)) {
    std::cout << "Device '"
              << q.get_device().get_info<sycl::info::device::name>()
              << "' does not support ESIMD, skipping test." << std::endl;
    return 0;
  }

  std::string log;

  source_kb kbSrc = syclex::create_kernel_bundle_from_source(
      ctx, syclex::source_language::sycl, ESIMDSource);
  exe_kb kbExe =
      syclex::build(kbSrc, syclex::properties{syclex::save_log{&log}});

  // extern "C" was used, so the name "vector_add_esimd" is not mangled and can
  // be used directly.
  sycl::kernel k = kbExe.ext_oneapi_get_kernel("vector_add_esimd");

  // Now test it.
  run_2(q, k, true, 3.14f);

  // Mix ESIMD and normal kernel.
  std::string mixedSource = std::string{ESIMDSource} + SYCLSource2;
  source_kb kbSrcMixed = syclex::create_kernel_bundle_from_source(
      ctx, syclex::source_language::sycl, mixedSource);
  exe_kb kbExeMixed = syclex::build(kbSrcMixed);

  // Both kernels should be available.
  sycl::kernel kESIMD = kbExeMixed.ext_oneapi_get_kernel("vector_add_esimd");
  sycl::kernel kSYCL = kbExeMixed.ext_oneapi_get_kernel("vec_add");

  // Device code split is mandatory.
  assert(std::distance(kbExeMixed.begin(), kbExeMixed.end()) == 2);

  // Test execution.
  run_2(q, kESIMD, true, 2.38f);
  run_2(q, kSYCL, false, 1.41f);

  // Deactivate implicit module splitting to exercise the downstream
  // ESIMD-specific splitting.
  source_kb kbSrcMixed2 = syclex::create_kernel_bundle_from_source(
      ctx, syclex::source_language::sycl, mixedSource);
  exe_kb kbExeMixed2 =
      syclex::build(kbSrcMixed2, syclex::properties{syclex::build_options{
                                     "-fsycl-device-code-split=off"}});

  assert(kbExeMixed2.ext_oneapi_has_kernel("vector_add_esimd"));
  assert(kbExeMixed2.ext_oneapi_has_kernel("vec_add"));
  assert(std::distance(kbExeMixed2.begin(), kbExeMixed2.end()) == 2);

  return 0;
}

int test_unsupported_options(sycl::queue q) {
  namespace syclex = sycl::ext::oneapi::experimental;
  using source_kb = sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source>;

  sycl::context ctx = q.get_context();

  source_kb kbSrc = syclex::create_kernel_bundle_from_source(
      ctx, syclex::source_language::sycl, "");
  std::vector<sycl::device> devs = kbSrc.get_devices();

  try {
    // Don't attempt to test exhaustively here...
    syclex::build(kbSrc, devs,
                  syclex::properties{
                      syclex::build_options{"-fsycl-targets=intel_gpu_pvc"}});
    assert(false && "unsupported option not detected");
  } catch (sycl::exception &e) {
    assert(e.code() == sycl::errc::invalid);
    assert(std::string(e.what()).find("Parsing of user arguments failed") !=
           std::string::npos);
  }

  return 0;
}

int test_error(sycl::queue q) {
  namespace syclex = sycl::ext::oneapi::experimental;
  using source_kb = sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source>;
  using exe_kb = sycl::kernel_bundle<sycl::bundle_state::executable>;

  sycl::context ctx = q.get_context();

  source_kb kbSrc = syclex::create_kernel_bundle_from_source(
      ctx, syclex::source_language::sycl, BadSource);
  try {
    exe_kb kbExe = syclex::build(kbSrc);
    assert(false && "we should not be here");
  } catch (sycl::exception &e) {
    // yas!
    assert(e.code() == sycl::errc::build);
    assert(std::string(e.what()).find(
               "error: expected ';' at end of declaration") !=
           std::string::npos);
  }
  return 0;
}

int test_warning(sycl::queue q) {
  namespace syclex = sycl::ext::oneapi::experimental;
  using source_kb = sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source>;
  using exe_kb = sycl::kernel_bundle<sycl::bundle_state::executable>;

  sycl::context ctx = q.get_context();

  std::string build_log;

  source_kb kbSrc = syclex::create_kernel_bundle_from_source(
      ctx, syclex::source_language::sycl, WarningSource);
  exe_kb kbExe =
      syclex::build(kbSrc, syclex::properties{syclex::save_log{&build_log}});
  bool found_warning =
      (build_log.find("warning: 'this_nd_item<1>' is deprecated") !=
       std::string::npos);
  assert(found_warning);
  return 0;
}

int test_no_visible_ids(sycl::queue q) {
  using source_kb = sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source>;
  using exe_kb = sycl::kernel_bundle<sycl::bundle_state::executable>;
  sycl::context ctx = q.get_context();
  source_kb kbSrc = syclex::create_kernel_bundle_from_source(
      ctx, syclex::source_language::sycl, SYCLSource2);
  exe_kb kbExe = syclex::build(kbSrc);
  assert(kbExe.get_kernel_ids().size() == 0 && "Visible RTC kernel ids");
  assert(sycl::get_kernel_ids().size() == 0 && "Visible RTC kernel ids");
  return 0;
}

int main() {
#ifdef SYCL_EXT_ONEAPI_KERNEL_COMPILER
  sycl::queue q;
  sycl::context ctx = q.get_context();

  bool ok =
      q.get_device().ext_oneapi_can_compile(syclex::source_language::sycl);
  if (!ok) {
    return -1;
  }
  // Run test_device_libraries twice to verify bfloat16 device library.
  return test_build_and_run(q) || test_device_code_split(q) ||
         test_device_libraries(q) || test_esimd(q) ||
         test_device_libraries(q) || test_unsupported_options(q) ||
         test_error(q) || test_no_visible_ids(q) || test_warning(q);
#else
  static_assert(false, "Kernel Compiler feature test macro undefined");
#endif
  return 0;
}
