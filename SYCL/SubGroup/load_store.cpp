// UNSUPPORTED: cpu
// #2252 Disable until all variants of built-ins are available in OpenCL CPU
// runtime for every supported ISA
//
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//
//==----------- load_store.cpp - SYCL sub_group load/store test ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "helper.hpp"
#include <CL/sycl.hpp>
template <typename T, int N> class sycl_subgr;

using namespace cl::sycl;

template <typename T, int N> void check(queue &Queue) {
  const int G = 1024, L = 128;

  // Pad arrays based on sub-group size to ensure no out-of-bounds accesses
  // Workaround for info::device::sub_group_sizes support on some devices
  size_t max_sg_size = 128;
#if 0
  auto sg_sizes = Queue.get_device().get_info<info::device::sub_group_sizes>();
  size_t max_sg_size = *std::max_element(sg_sizes.begin(), sg_sizes.end());
#endif

  try {
    nd_range<1> NdRange(G, L);
    buffer<T> syclbuf(G + max_sg_size * N);
    buffer<size_t> sgsizebuf(1);
    {
      auto acc = syclbuf.template get_access<access::mode::read_write>();
      for (int i = 0; i < G; i++) {
        acc[i] = i;
        acc[i] += 0.1; // Check that floating point types are not casted to int
      }
    }
    Queue.submit([&](handler &cgh) {
      auto acc = syclbuf.template get_access<access::mode::read_write>(cgh);
      auto sgsizeacc = sgsizebuf.get_access<access::mode::read_write>(cgh);
      accessor<T, 1, access::mode::read_write, access::target::local> LocalMem(
          {L + max_sg_size * N}, cgh);
      cgh.parallel_for<sycl_subgr<T, N>>(NdRange, [=](nd_item<1> NdItem) {
        ONEAPI::sub_group SG = NdItem.get_sub_group();
        if (SG.get_group_id().get(0) % N == 0) {
          size_t SGOffset =
              SG.get_group_id().get(0) * SG.get_max_local_range().get(0);
          size_t WGSGoffset = NdItem.get_group(0) * L + SGOffset;
          multi_ptr<T, access::address_space::global_space> mp(
              &acc[WGSGoffset]);
          multi_ptr<T, access::address_space::local_space> MPL(
              &LocalMem[SGOffset]);
          // Add all values in read block
          vec<T, N> v(utils<T, N>::add_vec(SG.load<N, T>(mp)));
          SG.store<N, T>(MPL, v);
          vec<T, N> t(utils<T, N>::add_vec(SG.load<N, T>(MPL)));
          SG.store<N, T>(mp, t);
        }
        if (NdItem.get_global_id(0) == 0)
          sgsizeacc[0] = SG.get_max_local_range()[0];
      });
    });
    auto acc = syclbuf.template get_access<access::mode::read_write>();
    auto sgsizeacc = sgsizebuf.get_access<access::mode::read_write>();
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
          ref += (T)(j + i * sg_size) + 0.1;
        }
        ref *= N;
      }
      /* There is no defined out-of-range behavior for these functions. */
      if ((SGid + N) * sg_size < L) {
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
      auto acc = syclbuf.template get_access<access::mode::read_write>();
      for (int i = 0; i < G; i++) {
        acc[i] = i;
        acc[i] += 0.1; // Check that floating point types are not casted to int
      }
    }

    Queue.submit([&](handler &cgh) {
      auto acc = syclbuf.template get_access<access::mode::read_write>(cgh);
      auto sgsizeacc = sgsizebuf.get_access<access::mode::read_write>(cgh);
      accessor<T, 1, access::mode::read_write, access::target::local> LocalMem(
          {L}, cgh);
      cgh.parallel_for<sycl_subgr<T, 0>>(NdRange, [=](nd_item<1> NdItem) {
        ONEAPI::sub_group SG = NdItem.get_sub_group();
        if (NdItem.get_global_id(0) == 0)
          sgsizeacc[0] = SG.get_max_local_range()[0];
        size_t SGOffset =
            SG.get_group_id().get(0) * SG.get_max_local_range().get(0);
        size_t WGSGoffset = NdItem.get_group(0) * L + SGOffset;
        multi_ptr<T, access::address_space::global_space> mp(&acc[WGSGoffset]);
        multi_ptr<T, access::address_space::local_space> MPL(
            &LocalMem[SGOffset]);
        T s = SG.load<T>(mp) + (T)SG.get_local_id().get(0);
        SG.store<T>(MPL, s);
        T t = SG.load<T>(MPL) + (T)SG.get_local_id().get(0);
        SG.store<T>(mp, t);
      });
    });
    auto acc = syclbuf.template get_access<access::mode::read_write>();
    auto sgsizeacc = sgsizebuf.get_access<access::mode::read_write>();
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
  if (Queue.get_device().is_host()) {
    std::cout << "Skipping test\n";
    return 0;
  }
  std::string PlatformName =
      Queue.get_device().get_platform().get_info<info::platform::name>();
  if (Queue.get_device().has_extension("cl_intel_subgroups") ||
      PlatformName.find("CUDA") != std::string::npos) {
    typedef bool aligned_char __attribute__((aligned(16)));
    check<aligned_char>(Queue);
    typedef int aligned_int __attribute__((aligned(16)));
    check<aligned_int>(Queue);
    check<aligned_int, 1>(Queue);
    check<aligned_int, 2>(Queue);
    check<aligned_int, 4>(Queue);
    check<aligned_int, 8>(Queue);
    typedef unsigned int aligned_uint __attribute__((aligned(16)));
    check<aligned_uint>(Queue);
    check<aligned_uint, 1>(Queue);
    check<aligned_uint, 2>(Queue);
    check<aligned_uint, 4>(Queue);
    check<aligned_uint, 8>(Queue);
    typedef float aligned_float __attribute__((aligned(16)));
    check<aligned_float>(Queue);
    check<aligned_float, 1>(Queue);
    check<aligned_float, 2>(Queue);
    check<aligned_float, 4>(Queue);
    check<aligned_float, 8>(Queue);
  }
  if (Queue.get_device().has_extension("cl_intel_subgroups_short") ||
      PlatformName.find("CUDA") != std::string::npos) {
    typedef short aligned_short __attribute__((aligned(16)));
    check<aligned_short>(Queue);
    check<aligned_short, 1>(Queue);
    check<aligned_short, 2>(Queue);
    check<aligned_short, 4>(Queue);
    check<aligned_short, 8>(Queue);
    if (Queue.get_device().has_extension("cl_khr_fp16") ||
        PlatformName.find("CUDA") != std::string::npos) {
      typedef half aligned_half __attribute__((aligned(16)));
      check<aligned_half>(Queue);
      check<aligned_half, 1>(Queue);
      check<aligned_half, 2>(Queue);
      check<aligned_half, 4>(Queue);
      check<aligned_half, 8>(Queue);
    }
  }
  if (Queue.get_device().has_extension("cl_intel_subgroups_long") ||
      PlatformName.find("CUDA") != std::string::npos) {
    typedef long aligned_long __attribute__((aligned(16)));
    check<aligned_long>(Queue);
    check<aligned_long, 1>(Queue);
    check<aligned_long, 2>(Queue);
    check<aligned_long, 4>(Queue);
    check<aligned_long, 8>(Queue);
    typedef unsigned long aligned_ulong __attribute__((aligned(16)));
    check<aligned_ulong>(Queue);
    check<aligned_ulong, 1>(Queue);
    check<aligned_ulong, 2>(Queue);
    check<aligned_ulong, 4>(Queue);
    check<aligned_ulong, 8>(Queue);
    typedef double aligned_double __attribute__((aligned(16)));
    check<aligned_double>(Queue);
    check<aligned_double, 1>(Queue);
    check<aligned_double, 2>(Queue);
    check<aligned_double, 4>(Queue);
    check<aligned_double, 8>(Queue);
  }
  std::cout << "Test passed." << std::endl;
  return 0;
}
