### Call ESIMD from SYCL using invoke_simd and Shared Local Memory (SLM)

In this example, we will manipulate Shared Local Memory (SLM) with SYCL and `invoke_simd`.

Compile and run:
```bash
> clang++ -fsycl -fno-sycl-device-code-split-esimd -Xclang -fsycl-allow-func-ptr invoke_simd_slm.cpp

> IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 ONEAPI_DEVICE_SELECTOR=level_zero:gpu ./a.out
Running on Intel(R) UHD Graphics 630
Passed
```
Source code:
```c++
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/oneapi/experimental/invoke_simd.hpp>
#include <sycl/sycl.hpp>
#include <iostream>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental;
namespace esimd = sycl::ext::intel::esimd;

constexpr int VL = 16;
constexpr uint32_t LOCAL_RANGE = VL * 2;           // 2 sub-groups per 1 group.
constexpr uint32_t GLOBAL_RANGE = LOCAL_RANGE * 2; // 2 groups.

[[intel::device_indirectly_callable]] SYCL_EXTERNAL void __regcall invoke_slm_load_store(
    local_accessor<int, 1> *local_acc, uint32_t slm_byte_offset, int *in, int *out,
    simd<uint32_t, VL> global_byte_offsets) SYCL_ESIMD_FUNCTION {
  esimd::simd<uint32_t, VL> esimd_global_byte_offsets = global_byte_offsets;
  // Read SLM in ESIMD context.
  auto local1 = esimd::block_load<int, VL>(*local_acc, slm_byte_offset);
  auto local2 = esimd::block_load<int, VL>(*local_acc, slm_byte_offset + 
                                               LOCAL_RANGE * sizeof(int));
  auto global = esimd::gather(in, esimd_global_byte_offsets);
  auto res = global + local1 + local2;
  esimd::scatter(out, esimd_global_byte_offsets, res);
}

int main(void) {
  auto q = queue{gpu_selector_v};
  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<sycl::info::device::name>()
            << std::endl;

  sycl::nd_range<1> nd_range{range<1>{GLOBAL_RANGE}, range<1>{LOCAL_RANGE}};

  auto *in = malloc_shared<int>(GLOBAL_RANGE, q);
  auto *out = malloc_shared<int>(GLOBAL_RANGE, q);

  for (auto i = 0; i < GLOBAL_RANGE; i++) {
    in[i] = i;
    out[i] = 0;
  }
  try {
    q.submit([&](handler &cgh) {
       auto local_acc = local_accessor<int, 1>(LOCAL_RANGE * 2, cgh);
       cgh.parallel_for(nd_range, [=](nd_item<1> item) {
         uint32_t global_id = item.get_global_id(0);
         uint32_t local_id = item.get_local_id(0);
         // Write/initialize SLM in SYCL context.
         auto local_acc_copy = local_acc;
         local_acc_copy[local_id] = global_id * 2;
         local_acc_copy[local_id + LOCAL_RANGE] = global_id * 3;
         item.barrier();

         uint32_t la_byte_offset = (local_id / VL) * VL * sizeof(int);
         uint32_t global_byte_offset = global_id * sizeof(int);
         sycl::sub_group sg = item.get_sub_group();
         // Pass the local-accessor to initialized SLM memory to ESIMD context.
         // Pointer to a local copy of the local accessor is passed instead of a local-accessor value now
         // to work-around a known/temporary issue in GPU driver.
         auto local_acc_arg = uniform{&local_acc_copy};
         invoke_simd(sg, invoke_slm_load_store, local_acc_arg,
                     uniform{la_byte_offset}, uniform{in}, uniform{out},
                     global_byte_offset);
       });
     }).wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    free(in, q);
    free(out, q);
    return e.code().value();
  }

  bool pass = true;
  for (auto i = 0; i < GLOBAL_RANGE; i++) {
    int expected = in[i] + i * 5;
    if (out[i] != expected) {
      std::cout << "Error: out[" << i << "]:" << out[i]
                << " != [expected]:" << expected << std::endl;
      pass = false;
    }
  }

  free(in, q);
  free(out, q);

  std::cout << (pass ? "Passed" : "FAILED") << std::endl;
  return pass ? 0 : 1;
}
```
