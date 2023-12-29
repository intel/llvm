//==-------lsc_slm_atomic_smoke.cpp  - DPC++ ESIMD on-device test --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This test checks LSC SLM atomic operations.
//===----------------------------------------------------------------------===//
// REQUIRES: gpu-intel-pvc || gpu-intel-dg2
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "../esimd_test_utils.hpp"

#include <iostream>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;

constexpr int Signed = 1;
constexpr int Unsigned = 2;

constexpr int64_t threads_per_group = 8;
constexpr int64_t n_groups = 1;
constexpr int64_t start_ind = 3;
constexpr int64_t masked_lane = 1;
constexpr int64_t repeat = 1;
constexpr int64_t stride = 4;

// ----------------- Helper functions

using LSCAtomicOp = sycl::ext::intel::esimd::atomic_op;
using AtomicOp = LSCAtomicOp;
constexpr char MODE[] = "LSC";

template <class, int, template <class, int> class> class TestID;

const char *to_string(LSCAtomicOp op) {
  switch (op) {
  case LSCAtomicOp::add:
    return "lsc::add";
  case LSCAtomicOp::sub:
    return "lsc::sub";
  case LSCAtomicOp::inc:
    return "lsc::inc";
  case LSCAtomicOp::dec:
    return "lsc::dec";
  case LSCAtomicOp::umin:
    return "lsc::umin";
  case LSCAtomicOp::umax:
    return "lsc::umax";
  case LSCAtomicOp::cmpxchg:
    return "lsc::cmpxchg";
  case LSCAtomicOp::bit_and:
    return "lsc::bit_and";
  case LSCAtomicOp::bit_or:
    return "lsc::bit_or";
  case LSCAtomicOp::bit_xor:
    return "lsc::bit_xor";
  case LSCAtomicOp::smin:
    return "lsc::smin";
  case LSCAtomicOp::smax:
    return "lsc::smax";
  case LSCAtomicOp::fmax:
    return "lsc::fmax";
  case LSCAtomicOp::fmin:
    return "lsc::fmin";
  case LSCAtomicOp::fcmpxchg:
    return "lsc::fcmpxchg";
  case LSCAtomicOp::load:
    return "lsc::load";
  case LSCAtomicOp::store:
    return "lsc::store";
  }
  return "lsc::<unknown>";
}

template <int N> inline bool any(simd_mask<N> m, simd_mask<N> ignore_mask) {
  simd_mask<N> m1 = 0;
  m.merge(m1, ignore_mask);
  return m.any();
}

// ----------------- The main test function

template <class T, int N, template <class, int> class ImplF>
bool test(queue q) {
  constexpr auto op = ImplF<T, N>::atomic_op;
  using CurAtomicOpT = decltype(op);
  constexpr int n_args = ImplF<T, N>::n_args;

  std::cout << "Testing mode=" << MODE << " op=" << to_string(op)
            << " T=" << esimd_test::type_name<T>() << " N=" << N << "\n\t"
            << "{ thr_per_group=" << threads_per_group
            << " n_groups=" << n_groups << " start_ind=" << start_ind
            << " masked_lane=" << masked_lane << " repeat=" << repeat
            << " stride=" << stride << " }...";

  constexpr size_t size = start_ind + (N - 1) * stride + 1;
  T *arr = malloc_shared<T>(size, q);
  constexpr int n_threads = threads_per_group * n_groups;

  for (int i = 0; i < size; ++i) {
    arr[i] = ImplF<T, N>::init(i);
  }

  range<1> glob_rng(n_threads);
  range<1> loc_rng(threads_per_group);
  nd_range<1> rng(glob_rng, loc_rng);

  try {
    auto e = q.submit([&](handler &cgh) {
      cgh.parallel_for<TestID<T, N, ImplF>>(
          rng, [=](id<1> ii) SYCL_ESIMD_KERNEL {
            int i = ii;
            slm_init<32768>();
            simd<uint32_t, N> offsets(start_ind * sizeof(T),
                                      stride * sizeof(T));
            simd<T, size> data;
            data.copy_from(arr);

            simd<uint32_t, size> slm_offsets(0, sizeof(T));
            lsc_slm_scatter(slm_offsets, data);

            simd_mask<N> m = 1;
            if (masked_lane < N)
              m[masked_lane] = 0;
            // Intra-work group barrier.
            barrier();

            // the atomic operation itself applied in a loop:
            for (int cnt = 0; cnt < repeat; ++cnt) {
              if constexpr (n_args == 0) {
                lsc_slm_atomic_update<op, T>(offsets, m);
              } else if constexpr (n_args == 1) {
                simd<T, N> v0 = ImplF<T, N>::arg0(i);
                lsc_slm_atomic_update<op, T>(offsets, v0, m);
              } else if constexpr (n_args == 2) {
                simd<T, N> new_val = ImplF<T, N>::arg0(i); // new value
                simd<T, N> exp_val = ImplF<T, N>::arg1(i); // expected value
                // do compare-and-swap in a loop until we get expected value;
                // arg0 and arg1 must provide values which guarantee the loop
                // is not endless:
                for (auto old_val = lsc_slm_atomic_update<op, T>(
                         offsets, exp_val, new_val, m);
                     any(old_val < exp_val, !m);
                     old_val = lsc_slm_atomic_update<op, T>(offsets, exp_val,
                                                            new_val, m))
                  ;
              }
            }
            auto data0 = lsc_slm_gather<T>(slm_offsets);
            data0.copy_to(arr);
          });
    });
    e.wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    free(arr, q);
    return false;
  }
  int err_cnt = 0;

  for (int i = 0; i < size; ++i) {
    T gold = ImplF<T, N>::gold(i);
    T test = arr[i];

    if ((gold != test) && (++err_cnt < 10)) {
      if (err_cnt == 1) {
        std::cout << "\n";
      }
      std::cout << "  failed at index " << i << ": " << test << " != " << gold
                << "(gold)\n";
    }
  }
  if (err_cnt > 0) {
    std::cout << "  FAILED\n  pass rate: "
              << ((float)(size - err_cnt) / (float)size) * 100.0f << "% ("
              << (size - err_cnt) << "/" << size << ")\n";
  } else {
    std::cout << " passed\n";
  }
  free(arr, q);
  return err_cnt == 0;
}

// ----------------- Functions providing input and golden values for atomic
// ----------------- operations.

static int dense_ind(int ind, int VL) { return (ind - start_ind) / stride; }

static bool is_updated(int ind, int VL) {
  if ((ind < start_ind) || (((ind - start_ind) % stride) != 0)) {
    return false;
  }
  int ii = dense_ind(ind, VL);
  bool res = (ii % VL) != masked_lane;
  return res;
}

// ----------------- Actual "traits" for each operation.

template <class T, int N, class C, C Op> struct ImplIncBase {
  static constexpr C atomic_op = Op;
  static constexpr int n_args = 0;

  static T init(int i) { return (T)0; }

  static T gold(int i) {
    T gold =
        is_updated(i, N) ? (T)(repeat * threads_per_group * n_groups) : init(i);
    return gold;
  }
};

template <class T, int N, class C, C Op> struct ImplDecBase {
  static constexpr C atomic_op = Op;
  static constexpr int n_args = 0;
  static constexpr int base = 5;

  static T init(int i) {
    return (T)(repeat * threads_per_group * n_groups + base);
  }

  static T gold(int i) {
    T gold = is_updated(i, N) ? (T)base : init(i);
    return gold;
  }
};

// The purpose of this is validate that floating point data is correctly
// processed.
constexpr float FPDELTA = 0.5f;

template <class T, int N, class C, C Op> struct ImplLoadBase {
  static constexpr C atomic_op = Op;
  static constexpr int n_args = 0;

  static T init(int i) { return (T)(i + FPDELTA); }

  static T gold(int i) {
    T gold = init(i);
    return gold;
  }
};

template <class T, int N, class C, C Op> struct ImplStoreBase {
  static constexpr C atomic_op = Op;
  static constexpr int n_args = 1;
  static constexpr T base = (T)(2 + FPDELTA);

  static T init(int i) { return 0; }

  static T gold(int i) {
    T gold = is_updated(i, N) ? base : init(i);
    return gold;
  }

  static T arg0(int i) { return base; }
};

template <class T, int N, class C, C Op> struct ImplAdd {
  static constexpr C atomic_op = Op;
  static constexpr int n_args = 1;

  static T init(int i) { return 0; }

  static T gold(int i) {
    T gold = is_updated(i, N)
                 ? (T)(repeat * threads_per_group * n_groups * (T)(1 + FPDELTA))
                 : init(i);
    return gold;
  }

  static T arg0(int i) { return (T)(1 + FPDELTA); }
};

template <class T, int N, class C, C Op> struct ImplSub {
  static constexpr C atomic_op = Op;
  static constexpr int n_args = 1;
  static constexpr T base = (T)(5 + FPDELTA);

  static T init(int i) {
    return (T)(repeat * threads_per_group * n_groups * (T)(1 + FPDELTA) + base);
  }

  static T gold(int i) {
    T gold = is_updated(i, N) ? base : init(i);
    return gold;
  }

  static T arg0(int i) { return (T)(1 + FPDELTA); }
};

template <class T, int N, class C, C Op> struct ImplMin {
  static constexpr C atomic_op = Op;
  static constexpr int n_args = 1;

  static T init(int i) { return std::numeric_limits<T>::max(); }

  static T gold(int i) {
    T ExpectedFoundMin;
    if constexpr (std::is_signed_v<T>)
      ExpectedFoundMin = FPDELTA - (threads_per_group * n_groups - 1);
    else
      ExpectedFoundMin = FPDELTA;
    T gold = is_updated(i, N) ? ExpectedFoundMin : init(i);
    return gold;
  }

  static T arg0(int i) {
    int64_t sign = std::is_signed_v<T> ? -1 : 1;
    return sign * i + FPDELTA;
  }
};

template <class T, int N, class C, C Op> struct ImplMax {
  static constexpr C atomic_op = Op;
  static constexpr int n_args = 1;

  static T init(int i) { return std::numeric_limits<T>::lowest(); }

  static T gold(int i) {
    T ExpectedFoundMax = FPDELTA;
    if constexpr (!std::is_signed_v<T>)
      ExpectedFoundMax += threads_per_group * n_groups - 1;

    T gold = is_updated(i, N) ? ExpectedFoundMax : init(i);
    return gold;
  }

  static T arg0(int i) {
    int64_t sign = std::is_signed_v<T> ? -1 : 1;
    return sign * i + FPDELTA;
  }
};

template <class T, int N>
struct ImplStore : ImplStoreBase<T, N, LSCAtomicOp, LSCAtomicOp::store> {};
template <class T, int N>
struct ImplLoad : ImplLoadBase<T, N, LSCAtomicOp, LSCAtomicOp::load> {};
template <class T, int N>
struct ImplInc : ImplIncBase<T, N, LSCAtomicOp, LSCAtomicOp::inc> {};
template <class T, int N>
struct ImplDec : ImplDecBase<T, N, LSCAtomicOp, LSCAtomicOp::dec> {};
template <class T, int N>
struct ImplIntAdd : ImplAdd<T, N, LSCAtomicOp, LSCAtomicOp::add> {};
template <class T, int N>
struct ImplIntSub : ImplSub<T, N, LSCAtomicOp, LSCAtomicOp::sub> {};
template <class T, int N>
struct ImplSMin : ImplMin<T, N, LSCAtomicOp, LSCAtomicOp::smin> {};
template <class T, int N>
struct ImplUMin : ImplMin<T, N, LSCAtomicOp, LSCAtomicOp::umin> {};
template <class T, int N>
struct ImplSMax : ImplMax<T, N, LSCAtomicOp, LSCAtomicOp::smax> {};
template <class T, int N>
struct ImplUMax : ImplMax<T, N, LSCAtomicOp, LSCAtomicOp::umax> {};

template <class T, int N>
struct ImplLSCFmin : ImplMin<T, N, LSCAtomicOp, LSCAtomicOp::fmin> {};
template <class T, int N>
struct ImplLSCFmax : ImplMax<T, N, LSCAtomicOp, LSCAtomicOp::fmax> {};

template <class T, int N, class C, C Op> struct ImplCmpxchgBase {
  static constexpr C atomic_op = Op;
  static constexpr int n_args = 2;
  static constexpr T base = (T)(2 + FPDELTA);

  static T init(int i) { return base - 1; }

  static T gold(int i) {
    T gold = is_updated(i, N) ? (T)(threads_per_group * n_groups - 1 + base)
                              : init(i);
    return gold;
  }

  // "Replacement value" argument in CAS
  static inline T arg0(int i) { return i + base; }

  // "Expected value" argument in CAS
  static inline T arg1(int i) { return i + base - 1; }
};

template <class T, int N>
struct ImplCmpxchg : ImplCmpxchgBase<T, N, LSCAtomicOp, LSCAtomicOp::cmpxchg> {
};

template <class T, int N>
struct ImplLSCFcmpwr
    : ImplCmpxchgBase<T, N, LSCAtomicOp, LSCAtomicOp::fcmpxchg> {};

// ----------------- Main function and test combinations.

template <int N, template <class, int> class Op,
          int SignMask = (Signed | Unsigned)>
bool test_int_types(queue q) {
  bool passed = true;
  if constexpr (SignMask & Signed) {
    passed &= test<int16_t, N, Op>(q);

    // TODO: Enable testing of 8-bit integers is supported in HW.
    // passed &= test<int8_t, N, Op>(q);

    passed &= test<int32_t, N, Op>(q);
    if constexpr (std::is_same_v<Op<int64_t, N>, ImplCmpxchg<int64_t, N>>) {
      passed &= test<int64_t, N, Op>(q);
    }
  }

  if constexpr (SignMask & Unsigned) {
    passed &= test<uint16_t, N, Op>(q);

    // TODO: Enable testing of 8-bit integers is supported in HW.
    // passed &= test<uint8_t, N, Op>(q);

    passed &= test<uint32_t, N, Op>(q);
    if constexpr (std::is_same_v<Op<uint64_t, N>, ImplCmpxchg<uint64_t, N>>) {
      passed &= test<uint64_t, N, Op>(q);
    }
  }
  return passed;
}

template <int N, template <class, int> class Op> bool test_fp_types(queue q) {
  bool passed = true;
  if constexpr (std::is_same_v<Op<sycl::half, N>, ImplLSCFmax<sycl::half, N>> ||
                std::is_same_v<Op<sycl::half, N>, ImplLSCFmin<sycl::half, N>> ||
                std::is_same_v<Op<sycl::half, N>,
                               ImplLSCFcmpwr<sycl::half, N>>) {
    auto dev = q.get_device();
    if (dev.has(sycl::aspect::fp16)) {
      passed &= test<sycl::half, N, Op>(q);
    }
  }
  passed &= test<float, N, Op>(q);
  return passed;
}

template <template <class, int> class Op, int SignMask = (Signed | Unsigned)>
bool test_int_types_and_sizes(queue q) {
  bool passed = true;
  passed &= test_int_types<1, Op, SignMask>(q);
  passed &= test_int_types<2, Op, SignMask>(q);
  passed &= test_int_types<4, Op, SignMask>(q);
  passed &= test_int_types<8, Op, SignMask>(q);
  passed &= test_int_types<16, Op, SignMask>(q);

  return passed;
}

template <template <class, int> class Op>
bool test_fp_types_and_sizes(queue q) {
  bool passed = true;
  passed &= test_fp_types<1, Op>(q);
  passed &= test_fp_types<2, Op>(q);
  passed &= test_fp_types<4, Op>(q);
  passed &= test_fp_types<8, Op>(q);
  passed &= test_fp_types<16, Op>(q);
  return passed;
}

int main(void) {
  queue q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

  bool passed = true;
#ifndef CMPXCHG_TEST
  passed &= test_int_types_and_sizes<ImplInc>(q);
  passed &= test_int_types_and_sizes<ImplDec>(q);

  passed &= test_int_types_and_sizes<ImplIntAdd>(q);
  passed &= test_int_types_and_sizes<ImplIntSub>(q);

  passed &= test_int_types_and_sizes<ImplSMax, Signed>(q);
  passed &= test_int_types_and_sizes<ImplSMin, Signed>(q);

  passed &= test_int_types_and_sizes<ImplUMax, Unsigned>(q);
  passed &= test_int_types_and_sizes<ImplUMin, Unsigned>(q);

  passed &= test_fp_types_and_sizes<ImplLSCFmax>(q);
  passed &= test_fp_types_and_sizes<ImplLSCFmin>(q);

  // Check load/store operations
  passed &= test_int_types_and_sizes<ImplLoad>(q);
  passed &= test_int_types_and_sizes<ImplStore>(q);
  passed &= test_fp_types_and_sizes<ImplStore>(q);
#else
  passed &= test_int_types_and_sizes<ImplCmpxchg>(q);
  passed &= test_fp_types_and_sizes<ImplLSCFcmpwr>(q);
#endif
  std::cout << (passed ? "Passed\n" : "FAILED\n");
  return passed ? 0 : 1;
}
