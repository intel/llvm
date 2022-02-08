//==------- slm_gather_scatter_heavy.cpp  - DPC++ ESIMD on-device test -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// UNSUPPORTED: cuda || hip
// TODO: esimd_emulator fails due to outdated memory intrinsic
// XFAIL: esimd_emulator
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
//
// The test checks functionality of the slm gather/scatter ESIMD intrinsics.
// It varies element type, vector length and stride of gather/scatter operation.
// For simplicity of calculations, workgroup size (number of work items same
// as number of subgroups/threads for ESIMD) is always equal to the stride.
//
// To avoid the effect of "even number of bugs" leading to test pass even in
// case of bugs, gather and scatter test cases are separated.
// Also, on Gen checking that predicate works for 'gather' is not possible
// reliably. In
//   x = slm_gather<int, 16>(offsets, pred);
// lanes masked out by pred will still be overwritten by some undefined values
// returned in these lanes by 'slm_gather'.
//

#include "../esimd_test_utils.hpp"

#include <CL/sycl.hpp>
#include <iomanip>
#include <iostream>
#include <sycl/ext/intel/experimental/esimd.hpp>

using namespace cl::sycl;

template <class T>
using Acc = accessor<T, 1, access_mode::read_write, access::target::device>;

using namespace sycl::ext::intel::experimental;
using namespace sycl::ext::intel::experimental::esimd;
constexpr int DEFAULT_VAL = -1;

// Test case IDs - whether to use scalar or vector memory access, how to
// construct a mask (aka predicate) within the test case.
enum TestCase : unsigned char {
  TEST_SCALAR,            // masking does not make sense
  TEST_VECTOR_NO_MASK,    // memory accesses are unmasked
  TEST_VECTOR_CONST_MASK, // mask is a compile-time constant
  TEST_VECTOR_VAR_MASK    // mask is a variable
};

template <TestCase TC> static constexpr bool is_masked_tc() {
  return TC == TEST_VECTOR_CONST_MASK || TC == TEST_VECTOR_VAR_MASK;
}

template <TestCase TC> static const char *tc_to_string() {
  if constexpr (TC == TEST_SCALAR)
    return "TEST_SCALAR";
  if constexpr (TC == TEST_VECTOR_NO_MASK)
    return "TEST_VECTOR_NO_MASK";
  if constexpr (TC == TEST_VECTOR_CONST_MASK)
    return "TEST_VECTOR_CONST_MASK";
  if constexpr (TC == TEST_VECTOR_VAR_MASK)
    return "TEST_VECTOR_VAR_MASK";
  return "<UNKNOWN>";
}

constexpr int MASKED_LANE = 4;
static int MaskedLane = MASKED_LANE;

template <class T, unsigned VL, unsigned STRIDE> struct KernelBase {
  Acc<T> acc_in;
  Acc<T> acc_out;
  int masked_lane;
  KernelBase(Acc<T> acc_in, Acc<T> acc_out, int masked_lane)
      : acc_in(acc_in), acc_out(acc_out), masked_lane(masked_lane) {}

  static inline constexpr auto WG_SIZE = STRIDE;
  static inline constexpr auto WI_CHUNK_SIZE = VL;
  static inline constexpr auto WG_CHUNK_SIZE = WI_CHUNK_SIZE * WG_SIZE;
  static inline constexpr auto SLM_CHUNK_SIZE =
      WG_CHUNK_SIZE * sizeof(T); // in bytes

  unsigned inline get_wi_local_offset(nd_item<1> i) const {
    unsigned wi_local_id = static_cast<unsigned>(i.get_local_id(0));
    unsigned wi_local_offset = wi_local_id * WI_CHUNK_SIZE;
    return wi_local_offset;
  }

  unsigned inline get_wi_offset(nd_item<1> i) const {
    unsigned group_id = static_cast<unsigned>(i.get_group().get_id(0));
    unsigned wg_offset = group_id * WG_CHUNK_SIZE;
    unsigned wi_offset = wg_offset + get_wi_local_offset(i);
    return wi_offset;
  }
};

// Examples below in the gather and scatter kernel code are for the case
// VL = 4, WG size = access stride = 3, WG_CUHNK_SIZE = 4 * 3 = 12
// Each WG reads a continuous chunk of memory at offset wg_id * WG_CUNK_SIZE,
// re-shuffles elements within the chunk using SLM scatter or gather and then
// stores reshuffled data back to memory at the same offset.

template <class T, unsigned VL, unsigned STRIDE, TestCase TC>
struct GatherKernel : KernelBase<T, VL, STRIDE> {
  using B = KernelBase<T, VL, STRIDE>;
  using B::B;

  static const char *get_name() { return "slm_gather"; }

  void operator()(nd_item<1> i) const SYCL_ESIMD_KERNEL {
    slm_init(B::SLM_CHUNK_SIZE);

    // first, read data w/o shuffling into SLM
    simd<T, VL> val;
    val.copy_from(B::acc_in, B::get_wi_offset(i) * sizeof(T),
                  element_aligned_tag{});
    slm_block_store((unsigned)(B::get_wi_local_offset(i) * sizeof(T)), val);

    // wait for peers
    esimd::barrier();

    // now gather from SLM with stride and write shuffled vectors back to memory
    unsigned wi_local_id = static_cast<unsigned>(i.get_local_id(0));

    if constexpr (TC == TEST_SCALAR) {
      for (auto j = 0; j < VL; ++j) {
        val[j] = slm_scalar_load<T>((wi_local_id + j * STRIDE) * sizeof(T));
      }
    } else {
      simd<uint32_t, VL> offsets(wi_local_id * sizeof(T), STRIDE * sizeof(T));

      if constexpr (TC == TEST_VECTOR_NO_MASK) {
        val = slm_gather<T, VL>(offsets);
      } else if constexpr (TC == TEST_VECTOR_CONST_MASK) {
        simd_mask<VL> pred(1);
        pred[MASKED_LANE] = 0;
        val = slm_gather<T, VL>(offsets, pred);
      } else if constexpr (TC == TEST_VECTOR_VAR_MASK) {
        simd_mask<VL> pred(1);
        pred[B::masked_lane] = 0;
        val = slm_gather<T, VL>(offsets, pred);
      }
    }

    // clang-format off
    // Source memory:
    //         00 01 02 03 04 05 06 07 08 09 10 11 | 12 13 14 15 16 17 18 19 20 21 22 23 | ...
    // val (group id/local id):
    // (0/0)    *        *        *        *
    // (0/1)       *        *        *        *
    // (0/2)          *        *        *        *
    // (1/0)                                          *        *        *        *
    // (1/1)                                             *        *        *        *
    // (1/2)                                                *        *        *        *
    // Destination memory:
    //         00 03 06 09 01 04 07 10 02 05 08 11 | 12 15 18 21 13 16 19 22 14 17 20 23 | ...
    //                    |           |            |            |           |            |
    //         00 00 00 00 01 01 01 01 02 02 02 02   10 10 10 10 11 11 11 11 12 12 12 12 ...
    // The line above shows which WG/WI writes which portion of the destination
    // clang-format on

    // no need to wait after read
    val.copy_to(B::acc_out, B::get_wi_offset(i) * sizeof(T));
  }
};

template <class T, unsigned VL, unsigned STRIDE, TestCase TC>
struct ScatterKernel : KernelBase<T, VL, STRIDE> {
  using B = KernelBase<T, VL, STRIDE>;
  using B::B;
  static const char *get_name() { return "slm_scatter"; }

  ESIMD_INLINE void operator()(nd_item<1> i) const SYCL_ESIMD_KERNEL {
    slm_init(B::SLM_CHUNK_SIZE);

    // first, read data from memory into registers w/o shuffling
    simd<T, VL> val;
    val.copy_from(B::acc_in, B::get_wi_offset(i) * sizeof(T));

    // now write to SLM shuffling on the fly
    unsigned wi_local_id = static_cast<unsigned>(i.get_local_id(0));

    if constexpr (TC == TEST_SCALAR) {
      for (auto j = 0; j < VL; ++j) {
        slm_scalar_store((wi_local_id + j * STRIDE) * sizeof(T), (T)val[j]);
      }
    } else {
      simd<uint32_t, VL> offsets(wi_local_id * sizeof(T), STRIDE * sizeof(T));
      simd_mask<VL> pred = 1;

      if constexpr (TC == TEST_VECTOR_CONST_MASK) {
        pred[MASKED_LANE] = 0;
      } else if constexpr (TC == TEST_VECTOR_VAR_MASK) {
        pred[B::masked_lane] = 0;
      }
      if constexpr (is_masked_tc<TC>()) {
        // first, write something meaningful (which can be further verified)
        // into the elements which will be masked out in the later (main)
        // scatter
        simd<T, VL> val1 = (T)DEFAULT_VAL;
        slm_scatter(offsets, val1, !pred);
      }
      slm_scatter(offsets, val, pred);
    }

    // clang-format off
    // Source memory:
    //         00 01 02 03 04 05 06 07 08 09 10 11 | 12 13 14 15 16 17 18 19 20 21 22 23 | ...
    // val (group id/local_id):
    // (0/0)    *  *  *  *
    // (0/1)                *  *  *  *
    // (0/2)                            *  *  *  *
    // (1/0)                                          *  *  *  *
    // (1/1)                                                      *  *  *  *
    // (1/2)                                                                  *  *  *  *
    // Destination memory:
    //         00 04 08 01 05 09 02 06 10 03 07 11 | 12 16 20 13 17 21 14 18 22 15 19 23 | ...
    // WG/WI   00 01 02 00 01 02 00 01 02 00 01 02   10 11 12 10 11 12 10 11 12 10 11 12
    // The line above shows which WG/WI writes which portion of the destination
    // clang-format on

    // wait for peers
    esimd::barrier();

    // now copy shuffled data from SLM back to memory
    val = slm_block_load<T, VL>(
        (unsigned)(B::get_wi_local_offset(i) * sizeof(T)));
    val.copy_to(B::acc_out, B::get_wi_offset(i) * sizeof(T));
  }
};

// Partial specialization of the gather kernel to test vector length = 1.
template <class T, unsigned STRIDE>
struct GatherKernel<T, 1, STRIDE, TEST_VECTOR_NO_MASK>
    : KernelBase<T, 1, STRIDE> {
  using B = KernelBase<T, 1, STRIDE>;
  using B::B;

  static const char *get_name() { return "slm_gather_vl1"; }

  void operator()(nd_item<1> i) const SYCL_ESIMD_KERNEL {
    slm_init(B::SLM_CHUNK_SIZE);

    // first, read data into SLM
    T val = scalar_load<T>(B::acc_in, B::get_wi_offset(i) * sizeof(T));
    slm_scalar_store((unsigned)(B::get_wi_local_offset(i) * sizeof(T)), val);

    // wait for peers
    esimd::barrier();

    // now load from SLM and write back to memory
    unsigned wi_local_id = static_cast<unsigned>(i.get_local_id(0));
    simd<uint32_t, 1> offsets(wi_local_id * sizeof(T));
    simd<T, 1> vec1 = slm_gather<T, 1>(offsets); /*** THE TESTED API ***/
    scalar_store(B::acc_out, B::get_wi_offset(i) * sizeof(T), (T)vec1[0]);
  }
};

// Partial specialization of the scatter kernel to test vector length = 1.
template <class T, unsigned STRIDE>
struct ScatterKernel<T, 1, STRIDE, TEST_VECTOR_NO_MASK>
    : KernelBase<T, 1, STRIDE> {
  using B = KernelBase<T, 1, STRIDE>;
  using B::B;
  static const char *get_name() { return "slm_scatter_vl1"; }

  ESIMD_INLINE void operator()(nd_item<1> i) const SYCL_ESIMD_KERNEL {
    slm_init(B::SLM_CHUNK_SIZE);

    // first, read data from memory into registers
    simd<T, 1> val;
    val[0] = scalar_load<T>(B::acc_in, B::get_wi_offset(i) * sizeof(T));

    // now write to SLM
    unsigned wi_local_id = static_cast<unsigned>(i.get_local_id(0));
    simd<uint32_t, 1> offsets(wi_local_id * sizeof(T));
    slm_scatter(offsets, val); /*** THE TESTED API ***/

    // wait for peers
    esimd::barrier();

    // now copy data from SLM back to memory
    T v = slm_scalar_load<T>(B::get_wi_local_offset(i) * sizeof(T));
    scalar_store(B::acc_out, B::get_wi_offset(i) * sizeof(T), v);
  }
};

enum MemIODir { MEM_SCATTER, MEM_GATHER };

template <
    class T,
    class T1 = std::conditional_t<
        std::is_same_v<T, char>, int,
        std::conditional_t<std::is_same_v<T, unsigned char>, unsigned int, T>>>
T1 conv(T val) {
  return (T1)val;
}

// Verification algorithm depends on whether gather or scatter result is tested.
// With gather, consequitive source array elements get into resulting array with
// with VL stride. With scatter, they go with WG_SIZE stride.
template <class T, unsigned VL, unsigned STRIDE, MemIODir Dir, TestCase TC>
static bool verify(T *A, size_t size) {
  int err_cnt = 0;
  constexpr unsigned WG_SIZE = STRIDE;
  constexpr unsigned WG_CHUNK_SIZE = VL * STRIDE;
  assert(size % WG_CHUNK_SIZE == 0);
  T gold = (T)0;
  constexpr bool MASKED = is_masked_tc<TC>();
  constexpr bool is_gather = Dir == MEM_GATHER;

  // iterate over work group chunks
  for (unsigned wg_id = 0, wg_offset = 0; wg_offset < size;
       wg_id++, wg_offset += WG_CHUNK_SIZE) {
    // iterate over lanes within the work item - should contain consecutive
    // numbers
    constexpr unsigned UB1 = is_gather ? VL : WG_SIZE;

    for (unsigned x = 0; x < UB1; x++) {
      // iterate over work items (ESIMD threads) within each work group
      constexpr unsigned UB2 = is_gather ? WG_SIZE : VL;

      for (unsigned y = 0; y < UB2; y++) {
        bool is_lane_masked =
            MASKED ? (is_gather ? x == MASKED_LANE : y == MASKED_LANE) : false;
        unsigned off = wg_offset + y * UB1 + x;
        T real_gold = is_lane_masked ? (T)DEFAULT_VAL : gold;
        T val = A[off];

        // If lane was masked during gather operation, it may contain any value,
        // so skip verification.
        if (!(is_gather && is_lane_masked) && (val != real_gold)) {
          if (++err_cnt < 41) {
            std::cout << "  error at " << off << " (wg_id=" << wg_id
                      << ", x=" << x << ", y=" << y << "): " << conv(val)
                      << " != " << conv(real_gold) << " (gold)\n";
          }
        }
        // traversal is done so that each iteration (size/VL total) of this
        // inner loop peeks into a consecutive datum value:
        gold++;
      }
    }
  }
  if (err_cnt > 0) {
    std::cout << "  pass rate: "
              << ((float)(size - err_cnt) / (float)size) * 100.0f << "% ("
              << (size - err_cnt) << "/" << size << ")\n";
  }
  return err_cnt == 0;
}

template <class T, unsigned VL, unsigned STRIDE, MemIODir Dir, TestCase TC>
bool test_impl(queue q) {
  size_t size = VL == 1 ? 8 * STRIDE : VL * STRIDE;

  using KernelType =
      std::conditional_t<Dir == MEM_GATHER, GatherKernel<T, VL, STRIDE, TC>,
                         ScatterKernel<T, VL, STRIDE, TC>>;
  const char *title = KernelType::get_name();

  std::cout << title << " test, T=" << typeid(T).name() << " VL=" << VL
            << " STRIDE=" << STRIDE << ", TC=" << tc_to_string<TC>();
  if (is_masked_tc<TC>()) {
    std::cout << ", masked_lane=" << MaskedLane;
  }
  std::cout << "...\n";
  T *A = new T[size];
  T *B = new T[size];

  for (unsigned i = 0; i < size; ++i) {
    A[i] = (T)i;
    B[i] = (T)DEFAULT_VAL;
  }

  try {
    buffer<T, 1> buf_in(A, range<1>(size));
    buffer<T, 1> buf_out(B, range<1>(size));
    range<1> glob_range{size / VL};

    auto e = q.submit([&](handler &cgh) {
      auto acc_in = buf_in.template get_access<access::mode::read_write>(cgh);
      auto acc_out = buf_out.template get_access<access::mode::read_write>(cgh);
      constexpr auto WG_SIZE = STRIDE; // for simplicity of the test
      KernelType kernel(acc_in, acc_out, MaskedLane);
      cgh.parallel_for(nd_range<1>{glob_range, range<1>(WG_SIZE)}, kernel);
    });
    e.wait();
  } catch (cl::sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    delete[] A;
    delete[] B;
    return false;
  }
#ifdef DUMP_DATA
  std::cout << "Input data:\n";
  for (unsigned i = 0; i < size; ++i) {
    std::cout << std::setw(4) << (int)A[i] << " ";
    if ((i + 1) % 40 == 0)
      std::cout << "\n";
  }
  std::cout << "\n";
  std::cout << "Output data:\n";
  for (unsigned i = 0; i < size; ++i) {
    std::cout << std::setw(4) << (int)B[i] << " ";
    if ((i + 1) % 40 == 0)
      std::cout << "\n";
  }
  std::cout << "\n";
#endif // DUMP_DATA

  bool passed = verify<T, VL, STRIDE, Dir, TC>(B, size);
  delete[] A;
  delete[] B;

  std::cout << (!passed ? "  FAILED\n" : "  Passed\n");
  return passed;
}

template <class T, unsigned VL, unsigned STRIDE> bool test(queue q) {
  bool passed = true;
  std::cout << "\n";
  passed &= test_impl<T, VL, STRIDE, MEM_GATHER, TEST_SCALAR>(q);
  passed &= test_impl<T, VL, STRIDE, MEM_GATHER, TEST_VECTOR_NO_MASK>(q);
  passed &= test_impl<T, VL, STRIDE, MEM_GATHER, TEST_VECTOR_CONST_MASK>(q);
  // TODO FIXME enable TEST_VECTOR_VAR_MASK test cases once the VCBE bug with
  // handling non-compile-time constant masks in scatter is fixed.
  // passed &= test_impl<T, VL, STRIDE, MEM_GATHER, TEST_VECTOR_VAR_MASK>(q);
  passed &= test_impl<T, VL, STRIDE, MEM_SCATTER, TEST_SCALAR>(q);
  passed &= test_impl<T, VL, STRIDE, MEM_SCATTER, TEST_VECTOR_NO_MASK>(q);
  passed &= test_impl<T, VL, STRIDE, MEM_SCATTER, TEST_VECTOR_CONST_MASK>(q);
  // passed &= test_impl<T, VL, STRIDE, MEM_SCATTER, TEST_VECTOR_VAR_MASK>(q);
  return passed;
}

template <class T, unsigned STRIDE> bool test_vl1(queue q) {
  bool passed = true;
  std::cout << "\n";
  passed &= test_impl<T, 1, STRIDE, MEM_GATHER, TEST_VECTOR_NO_MASK>(q);
  passed &= test_impl<T, 1, STRIDE, MEM_SCATTER, TEST_VECTOR_NO_MASK>(q);
  return passed;
}

int main(int argc, char **argv) {
  queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

  bool passed = true;
  passed &= test_vl1<char, 3>(q);
  passed &= test<char, 16, 3>(q);
  passed &= test<char, 32, 3>(q);
  passed &= test<short, 8, 8>(q);
  passed &= test<short, 16, 1>(q);
  passed &= test<short, 32, 1>(q);
  passed &= test<int, 8, 2>(q);
  passed &= test<int, 8, 3>(q);
  passed &= test<int, 16, 2>(q);
  passed &= test<int, 16, 1>(q);
  passed &= test<int, 32, 1>(q);
  passed &= test<float, 8, 2>(q);
  passed &= test<float, 16, 5>(q);
  passed &= test<float, 32, 3>(q);
  passed &= test_vl1<float, 7>(q);
  passed &= test_vl1<half, 7>(q);
  passed &= test<half, 16, 2>(q);

  std::cout << (!passed ? "TEST FAILED\n" : "TEST Passed\n");
  return passed ? 0 : 1;
}
