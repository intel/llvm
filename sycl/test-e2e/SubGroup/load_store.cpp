// RUN: %{build} -Wno-error=deprecated-declarations -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %{run} %t.out
//
// TODO: Flaky reenable, see https://github.com/intel/llvm/issues/14765
// UNSUPPORTED: windows, linux
//
//==----------- load_store.cpp - SYCL sub_group load/store test ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "helper.hpp"

#include <algorithm>

template <typename T, int N> class sycl_subgr;

using namespace sycl;

template <typename T, int N> void check(queue &Queue) {
  const int G = 512, L = 256;

  auto sg_sizes = Queue.get_device().get_info<info::device::sub_group_sizes>();
  size_t max_sg_size = *std::max_element(sg_sizes.begin(), sg_sizes.end());

  try {
    nd_range<1> NdRange(G, L);
    buffer<T> syclbuf(G + max_sg_size * N);
    buffer<size_t> sgsizebuf(1);
    {
      host_accessor acc(syclbuf);
      for (int i = 0; i < G; i++) {
        acc[i] = i;
        acc[i] += 0.25; // Check that floating point types are not casted to int
      }
    }
    Queue.submit([&](handler &cgh) {
      auto acc = syclbuf.template get_access<access::mode::read_write>(cgh);
      auto sgsizeacc = sgsizebuf.get_access<access::mode::read_write>(cgh);
      local_accessor<T, 1> LocalMem({L + max_sg_size * N}, cgh);
      cgh.parallel_for<sycl_subgr<T, N>>(NdRange, [=](nd_item<1> NdItem) {
        sycl::sub_group SG = NdItem.get_sub_group();
        auto SGid = SG.get_group_id().get(0);
        auto SGsize = SG.get_max_local_range().get(0);
        /* Avoid overlapping data ranges inside and between local groups */
        if (SGid % N == 0 && (SGid + N) * SGsize <= L) {
          size_t SGOffset = SGid * SGsize;
          size_t WGSGoffset = NdItem.get_group(0) * L + SGOffset;
          auto mp = address_space_cast<access::address_space::global_space,
                                       sycl::access::decorated::yes>(
              &acc[WGSGoffset]);
          auto MPL = address_space_cast<access::address_space::local_space,
                                        sycl::access::decorated::yes>(
              &LocalMem[SGOffset]);

          // half does not have full support for volatile type qualifier
          using CVT = std::conditional_t<std::is_same_v<T, half>, const T,
                                         const volatile T>;

          multi_ptr<CVT, mp.address_space, sycl::access::decorated::yes> mp_cv(
              mp);
          multi_ptr<CVT, MPL.address_space, sycl::access::decorated::yes>
              MPL_CV(MPL);
          // Add all values in read block
          vec<T, N> v(SG.load<N, T>(mp));
          vec<T, N> v_cv(SG.load<N, CVT>(mp_cv));
          if (utils<T, N>::cmp_vec(
                  v, v_cv)) // Store result only if same for non-cv and cv
            SG.store<N, T>(MPL, v);
          vec<T, N> t(utils<T, N>::add_vec(SG.load<N, T>(MPL)));
          vec<T, N> t_cv(utils<T, N>::add_vec(SG.load<N, CVT>(MPL_CV)));
          if (utils<T, N>::cmp_vec(
                  t, t_cv)) // Store result only if same for non-cv and cv
            SG.store<N, T>(mp, t);
        }
        if (NdItem.get_global_id(0) == 0)
          sgsizeacc[0] = SGsize;
      });
    });
    host_accessor acc(syclbuf);
    host_accessor sgsizeacc(sgsizebuf);
    size_t sg_size = sgsizeacc[0];
    int WGid = -1, SGid = 0;
    for (int j = 0; j < (G - (sg_size * N)); j++) {
      if (j % L % sg_size == 0) {
        SGid++;
      }
      if (j % L == 0) {
        WGid++;
        SGid = 0;
      }
      T ref = 0;
      if (SGid % N) {
        ref = acc[j - (SGid % N) * sg_size];
      } else {
        for (int i = 0; i < N; i++) {
          ref += (T)(j + i * sg_size) + 0.25;
        }
      }
      /* There is no defined out-of-range behavior for these functions. */
      if ((SGid + N) * sg_size <= L) {
        std::string s("Vector<");
        s += std::string(typeid(ref).name()) + std::string(",") +
             std::to_string(N) + std::string(">[") + std::to_string(j) +
             std::string("]");
        exit_if_not_equal<T>(acc[j], ref, s.c_str());
      }
    }
  } catch (exception e) {
    std::cout << "SYCL exception caught: " << e.what();
    exit(1);
  }
}
template <typename T> void check(queue &Queue) {
  const int G = 128, L = 64;
  try {
    nd_range<1> NdRange(G, L);
    buffer<T> syclbuf(G);
    buffer<size_t> sgsizebuf(1);
    {
      host_accessor acc(syclbuf);
      for (int i = 0; i < G; i++) {
        acc[i] = i;
        acc[i] += 0.1; // Check that floating point types are not casted to int
      }
    }

    Queue.submit([&](handler &cgh) {
      auto acc = syclbuf.template get_access<access::mode::read_write>(cgh);
      auto sgsizeacc = sgsizebuf.get_access<access::mode::read_write>(cgh);
      local_accessor<T, 1> LocalMem({L}, cgh);
      cgh.parallel_for<sycl_subgr<T, 0>>(NdRange, [=](nd_item<1> NdItem) {
        sycl::sub_group SG = NdItem.get_sub_group();
        if (NdItem.get_global_id(0) == 0)
          sgsizeacc[0] = SG.get_max_local_range()[0];
        size_t SGOffset =
            SG.get_group_id().get(0) * SG.get_max_local_range().get(0);
        size_t WGSGoffset = NdItem.get_group(0) * L + SGOffset;
        auto mp =
            address_space_cast<access::address_space::global_space,
                               sycl::access::decorated::yes>(&acc[WGSGoffset]);
        auto MPL = address_space_cast<access::address_space::local_space,
                                      sycl::access::decorated::yes>(
            &LocalMem[SGOffset]);

        // half does not have full support for volatile type qualifier
        using CVT = std::conditional_t<std::is_same_v<T, half>, const T,
                                       const volatile T>;

        multi_ptr<CVT, mp.address_space, sycl::access::decorated::yes> mp_cv(
            mp);
        multi_ptr<CVT, MPL.address_space, sycl::access::decorated::yes> MPL_CV(
            MPL);
        T s = SG.load<T>(mp) + (T)SG.get_local_id().get(0);
        T s_cv = SG.load<CVT>(mp_cv) + (T)SG.get_local_id().get(0);
        if (s == s_cv) // Store result only if same for non-cv and cv
          SG.store<T>(MPL, s);
        T t = SG.load<T>(MPL) + (T)SG.get_local_id().get(0);
        T t_cv = SG.load<CVT>(MPL_CV) + (T)SG.get_local_id().get(0);
        if (t == t_cv) // Store result only if same for non-cv and cv
          SG.store<T>(mp, t);
      });
    });
    host_accessor acc(syclbuf);
    host_accessor sgsizeacc(sgsizebuf);
    size_t sg_size = sgsizeacc[0];
    int WGid = -1, SGid = 0;
    for (int j = 0; j < G; j++) {
      if (j % L % sg_size == 0) {
        SGid++;
      }
      if (j % L == 0) {
        WGid++;
        SGid = 0;
      }
      std::string s("Scalar<");
      s += std::string(typeid(acc[j]).name()) + std::string(">[") +
           std::to_string(j) + std::string("]");

      exit_if_not_equal<T>(acc[j], (T)(j + 2 * (j % L % sg_size)) + 0.1,
                           s.c_str());
    }

  } catch (exception e) {
    std::cout << "SYCL exception caught: " << e.what();
    exit(1);
  }
}

int main() {
  queue Queue;
  std::string PlatformName =
      Queue.get_device().get_platform().get_info<info::platform::name>();
  auto Vec = Queue.get_device().get_info<info::device::extensions>();
  if (std::find(Vec.begin(), Vec.end(), "cl_intel_subgroups") !=
          std::end(Vec) ||
      PlatformName.find("CUDA") != std::string::npos) {
    typedef bool aligned_char __attribute__((aligned(16)));
    check<aligned_char>(Queue);
    typedef int aligned_int __attribute__((aligned(16)));
    check<aligned_int>(Queue);
    check<aligned_int, 1>(Queue);
    check<aligned_int, 2>(Queue);
    check<aligned_int, 3>(Queue);
    check<aligned_int, 4>(Queue);
    check<aligned_int, 8>(Queue);
    check<aligned_int, 16>(Queue);
    typedef unsigned int aligned_uint __attribute__((aligned(16)));
    check<aligned_uint>(Queue);
    check<aligned_uint, 1>(Queue);
    check<aligned_uint, 2>(Queue);
    check<aligned_uint, 3>(Queue);
    check<aligned_uint, 4>(Queue);
    check<aligned_uint, 8>(Queue);
    check<aligned_uint, 16>(Queue);
    typedef float aligned_float __attribute__((aligned(16)));
    check<aligned_float>(Queue);
    check<aligned_float, 1>(Queue);
    check<aligned_float, 2>(Queue);
    check<aligned_float, 3>(Queue);
    check<aligned_float, 4>(Queue);
    check<aligned_float, 8>(Queue);
    check<aligned_float, 16>(Queue);
  }
  if (std::find(Vec.begin(), Vec.end(), "cl_intel_subgroups_short") !=
          std::end(Vec) ||
      PlatformName.find("CUDA") != std::string::npos) {
    typedef short aligned_short __attribute__((aligned(16)));
    check<aligned_short>(Queue);
    check<aligned_short, 1>(Queue);
    check<aligned_short, 2>(Queue);
    check<aligned_short, 3>(Queue);
    check<aligned_short, 4>(Queue);
    check<aligned_short, 8>(Queue);
    check<aligned_short, 16>(Queue);
    if (Queue.get_device().has(sycl::aspect::fp16) ||
        PlatformName.find("CUDA") != std::string::npos) {
      typedef half aligned_half __attribute__((aligned(16)));
      check<aligned_half>(Queue);
      check<aligned_half, 1>(Queue);
      check<aligned_half, 2>(Queue);
      check<aligned_half, 3>(Queue);
      check<aligned_half, 4>(Queue);
      check<aligned_half, 8>(Queue);
      check<aligned_half, 16>(Queue);

      typedef sycl::ext::oneapi::bfloat16 aligned_bfloat16
          __attribute__((aligned(16)));
      check<aligned_bfloat16>(Queue);
      check<aligned_bfloat16, 1>(Queue);
      check<aligned_bfloat16, 2>(Queue);
      check<aligned_bfloat16, 3>(Queue);
      check<aligned_bfloat16, 4>(Queue);
      check<aligned_bfloat16, 8>(Queue);
      check<aligned_bfloat16, 16>(Queue);
    }
  }
  if (std::find(Vec.begin(), Vec.end(), "cl_intel_subgroups_long") !=
          std::end(Vec) ||
      PlatformName.find("CUDA") != std::string::npos) {
    typedef long aligned_long __attribute__((aligned(16)));
    check<aligned_long>(Queue);
    check<aligned_long, 1>(Queue);
    check<aligned_long, 2>(Queue);
    check<aligned_long, 3>(Queue);
    check<aligned_long, 4>(Queue);
    check<aligned_long, 8>(Queue);
    check<aligned_long, 16>(Queue);
    typedef unsigned long aligned_ulong __attribute__((aligned(16)));
    check<aligned_ulong>(Queue);
    check<aligned_ulong, 1>(Queue);
    check<aligned_ulong, 2>(Queue);
    check<aligned_ulong, 3>(Queue);
    check<aligned_ulong, 4>(Queue);
    check<aligned_ulong, 8>(Queue);
    check<aligned_ulong, 16>(Queue);
    if (Queue.get_device().has(sycl::aspect::fp64)) {
      typedef double aligned_double __attribute__((aligned(16)));
      check<aligned_double>(Queue);
      check<aligned_double, 1>(Queue);
      check<aligned_double, 2>(Queue);
      check<aligned_double, 3>(Queue);
      check<aligned_double, 4>(Queue);
      check<aligned_double, 8>(Queue);
      check<aligned_double, 16>(Queue);
    }
  }
  std::cout << "Test passed." << std::endl;
  return 0;
}
