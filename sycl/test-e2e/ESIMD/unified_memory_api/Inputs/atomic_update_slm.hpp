//==-------atomic_update_slm.hpp - DPC++ ESIMD on-device test --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "common.hpp"

#include <iostream>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::intel::esimd;

constexpr int Signed = 1;
constexpr int Unsigned = 2;

constexpr int64_t threads_per_group = 8;
constexpr int64_t n_groups = 1;
constexpr int64_t start_ind = 3;
constexpr int64_t masked_lane = 1;
constexpr int64_t repeat = 1;
constexpr int64_t stride = 4;

// Helper functions

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
bool test_slm(queue q) {
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
      cgh.parallel_for(rng, [=](sycl::nd_item<1> ndi) SYCL_ESIMD_KERNEL {
        int i = ndi.get_global_id(0);
        constexpr uint32_t SLMSize = size * sizeof(T);
        slm_init<SLMSize>();

        simd<uint32_t, N> offsets(start_ind * sizeof(T), stride * sizeof(T));
        simd<T, size> data;
        data.copy_from(arr);

        if (ndi.get_local_id(0) == 0)
          slm_block_store(0, data);

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
        barrier();
        if (ndi.get_local_id(0) == 0) {
          auto data0 = slm_block_load<T, size>(0);
          data0.copy_to(arr);
        }
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

template <class T, int N, template <class, int> class ImplF, bool UseMask>
bool test_slm_acc(queue q) {
  constexpr auto op = ImplF<T, N>::atomic_op;
  using CurAtomicOpT = decltype(op);
  constexpr int n_args = ImplF<T, N>::n_args;

  std::cout << "SLM ACC testing" << " op=" << to_string(op)
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
      local_accessor<T, 1> LocalAcc(size, cgh);
      cgh.parallel_for(rng, [=](sycl::nd_item<1> NDI) SYCL_ESIMD_KERNEL {
        int i = NDI.get_global_id(0);
        uint16_t LocalID = NDI.get_local_id(0);
        simd<uint32_t, N> offsets(start_ind * sizeof(T), stride * sizeof(T));

        if (LocalID == 0)
          for (int I = 0; I < threads_per_group * N; I++)
            LocalAcc[I] = arr[i * N + I];
        barrier();

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
              atomic_update<op, T>(LocalAcc, offsets, m);
            } else {
              atomic_update<op, T>(LocalAcc, offsets);
            }
          } else if constexpr (n_args == 1) {
            simd<T, N> v0 = ImplF<T, N>::arg0(i);
            if constexpr (UseMask) {
              atomic_update<op, T>(LocalAcc, offsets, v0, m);
            } else {
              atomic_update<op, T>(LocalAcc, offsets, v0);
            }
          } else if constexpr (n_args == 2) {
            simd<T, N> new_val = ImplF<T, N>::arg0(i); // new value
            simd<T, N> exp_val = ImplF<T, N>::arg1(i); // expected value
            // do compare-and-swap in a loop until we get expected value;
            // arg0 and arg1 must provide values which guarantee the loop
            // is not endless:
            if constexpr (UseMask) {
              for (simd<T, N> old_val = atomic_update<op, T>(
                       LocalAcc, offsets, new_val, exp_val, m);
                   any(old_val < exp_val, !m);
                   old_val = atomic_update<op, T>(LocalAcc, offsets, new_val,
                                                  exp_val, m))
                ;
            } else {
              for (simd<T, N> old_val = atomic_update<op, T>(LocalAcc, offsets,
                                                             new_val, exp_val);
                   any(old_val < exp_val, !m);
                   old_val = atomic_update<op, T>(LocalAcc, offsets, new_val,
                                                  exp_val))
                ;
            }
          }
        }
        barrier();
        if (LocalID == 0)
          for (int I = 0; I < threads_per_group * N; I++)
            arr[i * N + I] = LocalAcc[I];
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

  static T init(int i) { return 0; }

  static T gold(int i, bool use_mask) {
    T base = (T)(2 + FPDELTA);
    T gold = is_updated(i, N, use_mask) ? base : init(i);
    return gold;
  }

  static T arg0(int i) {
    T base = (T)(2 + FPDELTA);
    return base;
  }
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

  static T init(int i) {
    T base = (T)(5 + FPDELTA);
    return (T)(repeat * threads_per_group * n_groups * (T)(1 + FPDELTA) + base);
  }

  static T gold(int i, bool use_mask) {
    T base = (T)(5 + FPDELTA);
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
struct ImplFadd : ImplAdd<T, N, atomic_op, atomic_op::fadd> {};
template <class T, int N>
struct ImplFsub : ImplSub<T, N, atomic_op, atomic_op::fsub> {};
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

template <bool UseAcc, class T, int N, template <class, int> class ImplF,
          bool UseMask>
auto run_test(queue q) {
  if constexpr (UseAcc) {
    return test_slm_acc<T, N, ImplF, UseMask>(q);
  } else {
    return test_slm<T, N, ImplF, UseMask>(q);
  }
}

template <int N, template <class, int> class Op, bool UseMask,
          TestFeatures Features, bool UseAcc,
          int SignMask = (Signed | Unsigned)>
bool test_int_types(queue q) {
  bool passed = true;
  if constexpr (SignMask & Signed) {
    if constexpr (Features == TestFeatures::DG2 ||
                  Features == TestFeatures::PVC)
      passed &= run_test<UseAcc, int16_t, N, Op, UseMask>(q);

    passed &= run_test<UseAcc, int32_t, N, Op, UseMask>(q);

    // int64_t not supported on DG2
    if constexpr (Features == TestFeatures::PVC) {
      passed &= run_test<UseAcc, int64_t, N, Op, UseMask>(q);
    }
  }

  if constexpr (SignMask & Unsigned) {
    if constexpr (Features == TestFeatures::DG2 ||
                  Features == TestFeatures::PVC)
      passed &= run_test<UseAcc, uint16_t, N, Op, UseMask>(q);

    passed &= run_test<UseAcc, uint32_t, N, Op, UseMask>(q);

    // uint64_t not supported on DG2
    if constexpr (Features == TestFeatures::PVC) {
      passed &= run_test<UseAcc, uint64_t, N, Op, UseMask>(q);
    }
  }
  return passed;
}

template <int N, template <class, int> class Op, bool UseMask,
          TestFeatures Features, bool UseAcc>
bool test_fp_types(queue q) {
  bool passed = true;
  if constexpr (Features == TestFeatures::DG2 ||
                Features == TestFeatures::PVC) {
    if constexpr (std::is_same_v<Op<sycl::half, N>,
                                 ImplLSCFmax<sycl::half, N>> ||
                  std::is_same_v<Op<sycl::half, N>,
                                 ImplLSCFmin<sycl::half, N>> ||
                  std::is_same_v<Op<sycl::half, N>,
                                 ImplLSCFcmpwr<sycl::half, N>>) {
      auto dev = q.get_device();
      if (dev.has(sycl::aspect::fp16)) {
        passed &= run_test<UseAcc, sycl::half, N, Op, UseMask>(q);
      }
    }
  }

  passed &= run_test<UseAcc, float, N, Op, UseMask>(q);

  if constexpr (Features == TestFeatures::DG2 ||
                Features == TestFeatures::PVC) {
    if (q.get_device().has(sycl::aspect::atomic64) &&
        q.get_device().has(sycl::aspect::fp64)) {
      passed &= run_test<UseAcc, double, N, Op, UseMask>(q);
    }
  }
  return passed;
}

template <template <class, int> class Op, bool UseMask, TestFeatures Features,
          bool UseAcc, int SignMask = (Signed | Unsigned)>
bool test_int_types_and_sizes(queue q) {
  bool passed = true;
  passed &= test_int_types<1, Op, UseMask, Features, UseAcc, SignMask>(q);
  passed &= test_int_types<2, Op, UseMask, Features, UseAcc, SignMask>(q);
  passed &= test_int_types<4, Op, UseMask, Features, UseAcc, SignMask>(q);
  passed &= test_int_types<8, Op, UseMask, Features, UseAcc, SignMask>(q);
  if (UseMask && Features == TestFeatures::Generic &&
      esimd_test::isGPUDriverGE(q, esimd_test::GPUDriverOS::LinuxAndWindows,
                                "26918", "101.4953", false)) {
    passed &= test_int_types<16, Op, UseMask, Features, UseAcc, SignMask>(q);
    passed &= test_int_types<32, Op, UseMask, Features, UseAcc, SignMask>(q);
  }

  // Supported by LSC atomic:
  if constexpr (Features == TestFeatures::DG2 ||
                Features == TestFeatures::PVC) {
    passed &= test_int_types<64, Op, UseMask, Features, UseAcc, SignMask>(q);
    passed &= test_int_types<12, Op, UseMask, Features, UseAcc, SignMask>(q);
    passed &= test_int_types<33, Op, UseMask, Features, UseAcc, SignMask>(q);
  }

  return passed;
}

template <template <class, int> class Op, bool UseMask, TestFeatures Features,
          bool UseAcc>
bool test_fp_types_and_sizes(queue q) {
  bool passed = true;
  passed &= test_fp_types<1, Op, UseMask, Features, UseAcc>(q);
  passed &= test_fp_types<2, Op, UseMask, Features, UseAcc>(q);
  passed &= test_fp_types<4, Op, UseMask, Features, UseAcc>(q);
  passed &= test_fp_types<8, Op, UseMask, Features, UseAcc>(q);
  passed &= test_fp_types<16, Op, UseMask, Features, UseAcc>(q);
  passed &= test_fp_types<32, Op, UseMask, Features, UseAcc>(q);

  // Supported by LSC atomic:
  if constexpr (Features == TestFeatures::DG2 ||
                Features == TestFeatures::PVC) {
    passed &= test_fp_types<64, Op, UseMask, Features, UseAcc>(q);
    passed &= test_fp_types<33, Op, UseMask, Features, UseAcc>(q);
    passed &= test_fp_types<65, Op, UseMask, Features, UseAcc>(q);
  }
  return passed;
}

template <bool UseMask, TestFeatures Features, bool UseAcc>
int test_with_mask(queue q) {
  bool passed = true;
#ifndef CMPXCHG_TEST
  passed &= test_int_types_and_sizes<ImplInc, UseMask, Features, UseAcc>(q);
  passed &= test_int_types_and_sizes<ImplDec, UseMask, Features, UseAcc>(q);

  passed &= test_int_types_and_sizes<ImplIntAdd, UseMask, Features, UseAcc>(q);
  passed &= test_int_types_and_sizes<ImplIntSub, UseMask, Features, UseAcc>(q);

  passed &=
      test_int_types_and_sizes<ImplSMax, UseMask, Features, UseAcc, Signed>(q);
  passed &=
      test_int_types_and_sizes<ImplSMin, UseMask, Features, UseAcc, Signed>(q);

  passed &=
      test_int_types_and_sizes<ImplUMax, UseMask, Features, UseAcc, Unsigned>(
          q);
  passed &=
      test_int_types_and_sizes<ImplUMin, UseMask, Features, UseAcc, Unsigned>(
          q);

  if constexpr (Features == TestFeatures::DG2 ||
                Features == TestFeatures::PVC) {
    passed &=
        test_fp_types_and_sizes<ImplLSCFmax, UseMask, Features, UseAcc>(q);
    passed &=
        test_fp_types_and_sizes<ImplLSCFmin, UseMask, Features, UseAcc>(q);

    // TODO: fadd/fsub are emulated in the newer driver, but do not pass
    // validation.
#if 0
    passed &= test_fp_types_and_sizes<ImplFadd, UseMask, Features, UseAcc>(q);
    passed &= test_fp_types_and_sizes<ImplFsub, UseMask, Features, UseAcc>(q);
#endif

    // Check load/store operations.
    passed &= test_int_types_and_sizes<ImplLoad, UseMask, Features, UseAcc>(q);
    passed &= test_int_types_and_sizes<ImplStore, UseMask, Features, UseAcc>(q);
    passed &= test_fp_types_and_sizes<ImplStore, UseMask, Features, UseAcc>(q);
  }
#else
  passed &= test_int_types_and_sizes<ImplCmpxchg, UseMask, Features, UseAcc>(q);
  passed &=
      test_fp_types_and_sizes<ImplLSCFcmpwr, UseMask, Features, UseAcc>(q);
#endif
  return passed;
}

template <TestFeatures Features> bool test_main(queue q) {
  bool passed = true;

  constexpr const bool UseMask = true;
  constexpr const bool UseAcc = true;

  passed &= test_with_mask<UseMask, Features, !UseAcc>(q);
  passed &= test_with_mask<!UseMask, Features, !UseAcc>(q);

  return passed;
}

template <TestFeatures Features> bool test_main_acc(queue q) {
  bool passed = true;

  constexpr const bool UseMask = true;
  constexpr const bool UseAcc = true;

  passed &= test_with_mask<UseMask, Features, UseAcc>(q);
  passed &= test_with_mask<!UseMask, Features, UseAcc>(q);

  return passed;
}
