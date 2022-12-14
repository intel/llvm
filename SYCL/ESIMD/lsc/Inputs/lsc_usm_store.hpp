//==------- lsc_flat_store.hpp - DPC++ ESIMD on-device test ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include <iostream>

#include "common.hpp"

using namespace sycl;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;

#ifdef USE_64_BIT_OFFSET
typedef uint64_t Toffset;
#else
typedef uint32_t Toffset;
#endif

template <int case_num, typename T, uint32_t Groups, uint32_t Threads,
          uint16_t VL, uint16_t VS, bool transpose,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none>
bool test(uint32_t pmask = 0xffffffff) {
  static_assert((VL == 1) || !transpose, "Transpose must have exec size 1");
  if constexpr (DS == lsc_data_size::u8u32 || DS == lsc_data_size::u16u32) {
    static_assert(!transpose, "Conversion types may not use vector");
    static_assert(VS == 1, "Only D32 and D64 support vector load");
  }

  static_assert(DS != lsc_data_size::u16u32h, "D16U32h not supported in HW");

  if constexpr (!transpose && VS > 1) {
    static_assert(VL == 16 || VL == 32,
                  "IGC prohibits execution size less than SIMD size when "
                  "vector size is greater than 1");
  }

  uint16_t Size = Groups * Threads * VL * VS;
  using Tuint = sycl::_V1::ext::intel::esimd::detail::uint_type_t<sizeof(T)>;

  Tuint vmask = (Tuint)-1;
  if constexpr (DS == lsc_data_size::u8u32)
    vmask = (T)0xff;
  if constexpr (DS == lsc_data_size::u16u32)
    vmask = (T)0xffff;
  if constexpr (DS == lsc_data_size::u16u32h)
    vmask = (T)0xffff0000;

  T old_val = get_rand<T>();
  T new_val = get_rand<T>();

  auto q = queue{gpu_selector_v};
  auto dev = q.get_device();
  std::cout << "Running case #" << case_num << " on "
            << dev.get_info<sycl::info::device::name>() << "\n";
  auto ctx = q.get_context();

  // workgroups
  sycl::range<1> GlobalRange{Groups};
  // threads in each group
  sycl::range<1> LocalRange{Threads};
  sycl::nd_range<1> Range{GlobalRange * LocalRange, LocalRange};

  T *out = static_cast<T *>(sycl::malloc_shared(Size * sizeof(T), dev, ctx));
  for (int i = 0; i < Size; i++)
    out[i] = old_val;

  try {
    auto e = q.submit([&](handler &cgh) {
      cgh.parallel_for<KernelID<case_num>>(
          Range, [=](sycl::nd_item<1> ndi) SYCL_ESIMD_KERNEL {
            uint16_t globalID = ndi.get_global_id(0);
            uint32_t elem_off = globalID * VL * VS;
            uint32_t byte_off = elem_off * sizeof(T);

            if constexpr (transpose) {
              simd<T, VS> vals(new_val + elem_off, 1);
              lsc_block_store<T, VS, DS, L1H, L3H>(out + elem_off, vals);
            } else {
              simd<Toffset, VL> offset(byte_off, VS * sizeof(T));
              simd_mask<VL> pred;
              for (int i = 0; i < VL; i++)
                pred.template select<1, 1>(i) = (pmask >> i) & 1;

              T val = new_val + elem_off;
              simd<T, VS * VL> vals;
              for (int i = 0; i < VL; i++)
                for (int j = 0; j < VS; j++)
                  vals.template select<1, 1>(i + j * VL) = val++;

              lsc_scatter<T, VS, DS, L1H, L3H, VL>(out, offset, vals, pred);
            }
          });
    });
    e.wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    sycl::free(out, ctx);
    return false;
  }

  bool passed = true;

  if constexpr (transpose) {
    for (int i = 0; i < Size; i++) {
      T expected_value = new_val + i;
      Tuint e = sycl::bit_cast<Tuint>(expected_value);
      Tuint out_val = sycl::bit_cast<Tuint>(out[i]);
      if (out_val != e) {
        passed = false;
        std::cout << "out[" << i << "] = 0x" << std::hex << out_val
                  << " vs etalon = 0x" << e << std::dec << std::endl;
      }
    }
  } else {
    for (int i = 0; i < Size; i++) {
      T expected_value = new_val + i;
      Tuint in_val = sycl::bit_cast<Tuint>(expected_value);
      Tuint out_val = sycl::bit_cast<Tuint>(out[i]);

      Tuint e =
          (pmask >> ((i / VS) % VL)) & 1
              ? (in_val & vmask) | (sycl::bit_cast<Tuint>(old_val) & ~vmask)
              : sycl::bit_cast<Tuint>(old_val);
      if (out_val != e) {
        passed = false;
        std::cout << "out[" << i << "] = 0x" << std::hex << out_val
                  << " vs etalon = 0x" << e << std::dec << std::endl;
      }
    }
  }

  if (!passed)
    std::cout << "Case #" << case_num << " FAILED" << std::endl;

  sycl::free(out, ctx);

  return passed;
}
