//==-------atomic_update_slm.hpp - DPC++ ESIMD on-device test --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../../esimd_test_utils.hpp"

#include <CL/sycl.hpp>
#include <iostream>
#include <sycl/ext/intel/esimd.hpp>

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

// Helper functions

template <class, int, template <class, int> class> class TestID;

const char *to_string(atomic_op op) {
  switch (op) {
  case atomic_op::add:
    return "add";
  case atomic_op::sub:
    return "sub";
  case atomic_op::inc:
    return "inc";
  case atomic_op::dec:
    return "dec";
  case atomic_op::umin:
    return "umin";
  case atomic_op::umax:
    return "umax";
  case atomic_op::xchg:
    return "xchg";
  case atomic_op::cmpxchg:
    return "cmpxchg";
  case atomic_op::bit_and:
    return "bit_and";
  case atomic_op::bit_or:
    return "bit_or";
  case atomic_op::bit_xor:
    return "bit_xor";
  case atomic_op::smin:
    return "smin";
  case atomic_op::smax:
    return "smax";
  case atomic_op::fmax:
    return "fmax";
  case atomic_op::fmin:
    return "fmin";
  case atomic_op::fadd:
    return "fadd";
  case atomic_op::fsub:
    return "fsub";
  case atomic_op::fcmpxchg:
    return "fcmpxchg";
  case atomic_op::load:
    return "load";
  case atomic_op::store:
    return "store";
  case atomic_op::predec:
    return "predec";
  }
  return "<unknown>";
}

template <int N> inline bool any(simd_mask<N> m, simd_mask<N> ignore_mask) {
  simd_mask<N> m1 = 0;
  m.merge(m1, ignore_mask);
  return m.any();
}

// The main test function

template <class T, int N, template <class, int> class ImplF, bool UseMask>
bool test(queue q) {
  constexpr auto op = ImplF<T, N>::atomic_op;
  using CurAtomicOpT = decltype(op);
  constexpr int n_args = ImplF<T, N>::n_args;

  std::cout << "SLM testing" << " op=" << to_string(op)
            << " T=" << esimd_test::type_name<T>() << " N=" << N << "\n\t"
            << " UseMask=" << (UseMask ? "true" : "false")
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
      cgh.parallel_for(rng, [=](id<1> ii) SYCL_ESIMD_KERNEL {
        int i = ii;
        slm_init<32768>();
        simd<uint32_t, N> offsets(start_ind * sizeof(T), stride * sizeof(T));
        simd<T, size> data;
        data.copy_from(arr);

        simd<uint32_t, size> slm_offsets(0, sizeof(T));
        // TODO: replace this API with unified version.
        lsc_slm_scatter(slm_offsets, data);

        simd_mask<N> m = 1;
        if constexpr (UseMask) {
          if (masked_lane < N)
            m[masked_lane] = 0;
        }
        // Intra-work group barrier.
        barrier();

        // the atomic operation itself applied in a loop:
        for (int cnt = 0; cnt < repeat; ++cnt) {
          if constexpr (n_args == 0) {
            if constexpr (UseMask) {
              slm_atomic_update<op, T>(offsets, m);
            } else {
              slm_atomic_update<op, T>(offsets);
            }
          } else if constexpr (n_args == 1) {
            simd<T, N> v0 = ImplF<T, N>::arg0(i);
            if constexpr (UseMask) {
              slm_atomic_update<op, T>(offsets, v0, m);
            } else {
              slm_atomic_update<op, T>(offsets, v0);
            }
          } else if constexpr (n_args == 2) {
            simd<T, N> new_val = ImplF<T, N>::arg0(i); // new value
            simd<T, N> exp_val = ImplF<T, N>::arg1(i); // expected value
            // do compare-and-swap in a loop until we get expected value;
            // arg0 and arg1 must provide values which guarantee the loop
            // is not endless:
            if constexpr (UseMask) {
              for (simd<T, N> old_val =
                       slm_atomic_update<op, T>(offsets, new_val, exp_val, m);
                   any(old_val < exp_val, !m);
                   old_val =
                       slm_atomic_update<op, T>(offsets, new_val, exp_val, m))
                ;
            } else {
              for (simd<T, N> old_val =
                       slm_atomic_update<op, T>(offsets, new_val, exp_val);
                   any(old_val < exp_val, !m);
                   old_val =
                       slm_atomic_update<op, T>(offsets, new_val, exp_val))
                ;
            }
          }
        }
        // TODO: replace this API with unified version.
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
    T gold = ImplF<T, N>::gold(i, UseMask);
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

static bool is_updated(int ind, int VL, bool use_mask) {
  if ((ind < start_ind) || (((ind - start_ind) % stride) != 0)) {
    return false;
  }
  int ii = dense_ind(ind, VL);
  bool res = true;
  if (use_mask)
    res = (ii % VL) != masked_lane;
  return res;
}

// ----------------- Actual "traits" for each operation.

template <class T, int N, class C, C Op> struct ImplIncBase {
  static constexpr C atomic_op = Op;
  static constexpr int n_args = 0;

  static T init(int i) { return (T)0; }

  static T gold(int i, bool use_mask) {
    T gold = is_updated(i, N, use_mask)
                 ? (T)(repeat * threads_per_group * n_groups)
                 : init(i);
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

  static T gold(int i, bool use_mask) {
    T gold = is_updated(i, N, use_mask) ? (T)base : init(i);
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

  static T gold(int i, bool use_mask) {
    T gold = init(i);
    return gold;
  }
};

template <class T, int N, class C, C Op> struct ImplStoreBase {
  static constexpr C atomic_op = Op;
  static constexpr int n_args = 1;
  static constexpr T base = (T)(2 + FPDELTA);

  static T init(int i) { return 0; }

  static T gold(int i, bool use_mask) {
    T gold = is_updated(i, N, use_mask) ? base : init(i);
    return gold;
  }

  static T arg0(int i) { return base; }
};

template <class T, int N, class C, C Op> struct ImplAdd {
  static constexpr C atomic_op = Op;
  static constexpr int n_args = 1;

  static T init(int i) { return 0; }

  static T gold(int i, bool use_mask) {
    T gold = is_updated(i, N, use_mask)
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

  static T gold(int i, bool use_mask) {
    T gold = is_updated(i, N, use_mask) ? base : init(i);
    return gold;
  }

  static T arg0(int i) { return (T)(1 + FPDELTA); }
};

template <class T, int N, class C, C Op> struct ImplMin {
  static constexpr C atomic_op = Op;
  static constexpr int n_args = 1;

  static T init(int i) { return std::numeric_limits<T>::max(); }

  static T gold(int i, bool use_mask) {
    T ExpectedFoundMin;
    if constexpr (std::is_signed_v<T>)
      ExpectedFoundMin = FPDELTA - (threads_per_group * n_groups - 1);
    else
      ExpectedFoundMin = FPDELTA;
    T gold = is_updated(i, N, use_mask) ? ExpectedFoundMin : init(i);
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

  static T gold(int i, bool use_mask) {
    T ExpectedFoundMax = FPDELTA;
    if constexpr (!std::is_signed_v<T>)
      ExpectedFoundMax += threads_per_group * n_groups - 1;

    T gold = is_updated(i, N, use_mask) ? ExpectedFoundMax : init(i);
    return gold;
  }

  static T arg0(int i) {
    int64_t sign = std::is_signed_v<T> ? -1 : 1;
    return sign * i + FPDELTA;
  }
};

template <class T, int N>
struct ImplStore : ImplStoreBase<T, N, atomic_op, atomic_op::store> {};
template <class T, int N>
struct ImplLoad : ImplLoadBase<T, N, atomic_op, atomic_op::load> {};
template <class T, int N>
struct ImplInc : ImplIncBase<T, N, atomic_op, atomic_op::inc> {};
template <class T, int N>
struct ImplDec : ImplDecBase<T, N, atomic_op, atomic_op::dec> {};
template <class T, int N>
struct ImplIntAdd : ImplAdd<T, N, atomic_op, atomic_op::add> {};
template <class T, int N>
struct ImplIntSub : ImplSub<T, N, atomic_op, atomic_op::sub> {};
template <class T, int N>
struct ImplSMin : ImplMin<T, N, atomic_op, atomic_op::smin> {};
template <class T, int N>
struct ImplUMin : ImplMin<T, N, atomic_op, atomic_op::umin> {};
template <class T, int N>
struct ImplSMax : ImplMax<T, N, atomic_op, atomic_op::smax> {};
template <class T, int N>
struct ImplUMax : ImplMax<T, N, atomic_op, atomic_op::umax> {};

template <class T, int N>
struct ImplLSCFmin : ImplMin<T, N, atomic_op, atomic_op::fmin> {};
template <class T, int N>
struct ImplLSCFmax : ImplMax<T, N, atomic_op, atomic_op::fmax> {};

template <class T, int N, class C, C Op> struct ImplCmpxchgBase {
  static constexpr C atomic_op = Op;
  static constexpr int n_args = 2;
  static constexpr T base = (T)(2 + FPDELTA);

  static T init(int i) { return base - 1; }

  static T gold(int i, bool use_mask) {
    T gold = is_updated(i, N, use_mask)
                 ? (T)(threads_per_group * n_groups - 1 + base)
                 : init(i);
    return gold;
  }

  // "Replacement value" argument in CAS
  static inline T arg0(int i) { return i + base; }

  // "Expected value" argument in CAS
  static inline T arg1(int i) { return i + base - 1; }
};

template <class T, int N>
struct ImplCmpxchg : ImplCmpxchgBase<T, N, atomic_op, atomic_op::cmpxchg> {};

template <class T, int N>
struct ImplLSCFcmpwr : ImplCmpxchgBase<T, N, atomic_op, atomic_op::fcmpxchg> {};

// ----------------- Main function and test combinations.

template <int N, template <class, int> class Op, bool UseMask,
          bool UsePVCFeatures, int SignMask = (Signed | Unsigned)>
bool test_int_types(queue q) {
  bool passed = true;
  if constexpr (SignMask & Signed) {
    if constexpr (UsePVCFeatures)
      passed &= test<int16_t, N, Op, UseMask>(q);

    passed &= test<int32_t, N, Op, UseMask>(q);
    if constexpr (std::is_same_v<Op<int64_t, N>, ImplCmpxchg<int64_t, N>>) {
      passed &= test<int64_t, N, Op, UseMask>(q);
    }
  }

  if constexpr (SignMask & Unsigned) {
    if constexpr (UsePVCFeatures)
      passed &= test<uint16_t, N, Op, UseMask>(q);

    passed &= test<uint32_t, N, Op, UseMask>(q);
    if constexpr (std::is_same_v<Op<uint64_t, N>, ImplCmpxchg<uint64_t, N>>) {
      passed &= test<uint64_t, N, Op, UseMask>(q);
    }
  }
  return passed;
}

template <int N, template <class, int> class Op, bool UseMask,
          bool UsePVCFeatures>
bool test_fp_types(queue q) {
  bool passed = true;
  if constexpr (UsePVCFeatures) {
    if constexpr (std::is_same_v<Op<sycl::half, N>,
                                 ImplLSCFmax<sycl::half, N>> ||
                  std::is_same_v<Op<sycl::half, N>,
                                 ImplLSCFmin<sycl::half, N>> ||
                  std::is_same_v<Op<sycl::half, N>,
                                 ImplLSCFcmpwr<sycl::half, N>>) {
      auto dev = q.get_device();
      if (dev.has(sycl::aspect::fp16)) {
        passed &= test<sycl::half, N, Op, UseMask>(q);
      }
    }
  }
  passed &= test<float, N, Op, UseMask>(q);
  return passed;
}

template <template <class, int> class Op, bool UseMask, bool UsePVCFeatures,
          int SignMask = (Signed | Unsigned)>
bool test_int_types_and_sizes(queue q) {
  bool passed = true;
  passed &= test_int_types<1, Op, UseMask, UsePVCFeatures, SignMask>(q);
  passed &= test_int_types<2, Op, UseMask, UsePVCFeatures, SignMask>(q);
  passed &= test_int_types<4, Op, UseMask, UsePVCFeatures, SignMask>(q);
  passed &= test_int_types<8, Op, UseMask, UsePVCFeatures, SignMask>(q);
  passed &= test_int_types<16, Op, UseMask, UsePVCFeatures, SignMask>(q);

  // Supported by LSC atomic:
  if constexpr (UsePVCFeatures) {
    passed &= test_int_types<32, Op, UseMask, UsePVCFeatures, SignMask>(q);
    passed &= test_int_types<64, Op, UseMask, UsePVCFeatures, SignMask>(q);
    // non power of two values are supported only in newer driver.
    // TODO: Enable this when the new driver reaches test infrastructure
    // (v27556).
#if 0
    passed &= test_int_types<12, Op, UseMask, UsePVCFeatures, SignMask>(q);
    passed &= test_int_types<33, Op, UseMask, UsePVCFeatures, SignMask>(q);
#endif
  }

  return passed;
}

template <template <class, int> class Op, bool UseMask, bool UsePVCFeatures>
bool test_fp_types_and_sizes(queue q) {
  bool passed = true;
  passed &= test_fp_types<1, Op, UseMask, UsePVCFeatures>(q);
  passed &= test_fp_types<2, Op, UseMask, UsePVCFeatures>(q);
  passed &= test_fp_types<4, Op, UseMask, UsePVCFeatures>(q);
  passed &= test_fp_types<8, Op, UseMask, UsePVCFeatures>(q);
  passed &= test_fp_types<16, Op, UseMask, UsePVCFeatures>(q);

  // Supported by LSC atomic:
  if constexpr (UsePVCFeatures) {
    passed &= test_fp_types<32, Op, UseMask, UsePVCFeatures>(q);
    passed &= test_fp_types<64, Op, UseMask, UsePVCFeatures>(q);
#if 0
    passed &= test_fp_types<33, Op, UseMask, UsePVCFeatures>(q);
    passed &= test_fp_types<65, Op, UseMask, UsePVCFeatures>(q);
#endif
  }
  return passed;
}

template <bool UseMask, bool UsePVCFeatures> int test_with_mask(queue q) {
  bool passed = true;
#ifndef CMPXCHG_TEST
  passed &= test_int_types_and_sizes<ImplInc, UseMask, UsePVCFeatures>(q);
  passed &= test_int_types_and_sizes<ImplDec, UseMask, UsePVCFeatures>(q);

  passed &= test_int_types_and_sizes<ImplIntAdd, UseMask, UsePVCFeatures>(q);
  passed &= test_int_types_and_sizes<ImplIntSub, UseMask, UsePVCFeatures>(q);

  passed &=
      test_int_types_and_sizes<ImplSMax, UseMask, UsePVCFeatures, Signed>(q);
  passed &=
      test_int_types_and_sizes<ImplSMin, UseMask, UsePVCFeatures, Signed>(q);

  passed &=
      test_int_types_and_sizes<ImplUMax, UseMask, UsePVCFeatures, Unsigned>(q);
  passed &=
      test_int_types_and_sizes<ImplUMin, UseMask, UsePVCFeatures, Unsigned>(q);

  if constexpr (UsePVCFeatures) {
    passed &= test_fp_types_and_sizes<ImplLSCFmax, UseMask, UsePVCFeatures>(q);
    passed &= test_fp_types_and_sizes<ImplLSCFmin, UseMask, UsePVCFeatures>(q);

    // Check load/store operations.
    passed &= test_int_types_and_sizes<ImplLoad, UseMask, UsePVCFeatures>(q);
    passed &= test_int_types_and_sizes<ImplStore, UseMask, UsePVCFeatures>(q);
    passed &= test_fp_types_and_sizes<ImplStore, UseMask, UsePVCFeatures>(q);
  } else {
    // These operations are not supported by LSC SLM.
    passed &= test_fp_types_and_sizes<ImplFmax, UseMask, UsePVCFeatures>(q);
    passed &= test_fp_types_and_sizes<ImplFmin, UseMask, UsePVCFeatures>(q);
  }
#else
  passed &= test_int_types_and_sizes<ImplCmpxchg, UseMask, UsePVCFeatures>(q);
  passed &= test_fp_types_and_sizes<ImplLSCFcmpwr, UseMask, UsePVCFeatures>(q);
#endif
  return passed;
}

template <bool UsePVCFeatures> bool test_main(queue q) {
  bool passed = true;

  constexpr const bool UseMask = true;

  passed &= test_with_mask<UseMask, UsePVCFeatures>(q);
  passed &= test_with_mask<!UseMask, UsePVCFeatures>(q);

  return passed;
}
