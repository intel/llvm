//==--------------- reduce.hpp - SYCL sub_group reduce test ----*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "helper.hpp"
#include <iostream>
#include <sycl/sycl.hpp>
template <typename... Ts> class sycl_subgr;

using namespace sycl;

template <typename SpecializationKernelName, typename T, class BinaryOperation>
void check_op(queue &Queue, T init, BinaryOperation op, bool skip_init = false,
              size_t G = 256, size_t L = 64) {
  try {
    nd_range<1> NdRange(G, L);
    buffer<T> buf(G);
    buffer<size_t> sgsizebuf(1);
    Queue.submit([&](handler &cgh) {
      auto sgsizeacc = sgsizebuf.get_access<access::mode::read_write>(cgh);
      auto acc = buf.template get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<SpecializationKernelName>(
          NdRange, [=](nd_item<1> NdItem) {
            ext::oneapi::sub_group sg = NdItem.get_sub_group();
            if (skip_init) {
              acc[NdItem.get_global_id(0)] =
                  ext::oneapi::reduce(sg, T(NdItem.get_global_id(0)), op);
            } else {
              acc[NdItem.get_global_id(0)] =
                  ext::oneapi::reduce(sg, T(NdItem.get_global_id(0)), init, op);
            }
            if (NdItem.get_global_id(0) == 0)
              sgsizeacc[0] = sg.get_max_local_range()[0];
          });
    });
    auto acc = buf.template get_access<access::mode::read_write>();
    auto sgsizeacc = sgsizebuf.get_access<access::mode::read_write>();
    size_t sg_size = sgsizeacc[0];
    int WGid = -1, SGid = 0;
    T result = init;
    for (int j = 0; j < G; j++) {
      if (j % L % sg_size == 0) {
        SGid++;
        result = init;
        for (int i = j; (i % L && i % L % sg_size) || (i == j); i++) {
          result = op(result, T(i));
        }
      }
      if (j % L == 0) {
        WGid++;
        SGid = 0;
      }
      std::string name =
          std::string("reduce_") + typeid(BinaryOperation).name();
      exit_if_not_equal<T>(acc[j], result, name.c_str());
    }
  } catch (exception e) {
    std::cout << "SYCL exception caught: " << e.what();
    exit(1);
  }
}

template <typename SpecializationKernelName, typename T>
void check(queue &Queue, size_t G = 256, size_t L = 64) {
  // limit data range for half to avoid rounding issues
  if (std::is_same<T, sycl::half>::value) {
    G = 64;
    L = 32;
  }

  check_op<
      sycl_subgr<SpecializationKernelName, class KernelName_cNsJzXxSBQfEKY>, T>(
      Queue, T(L), ext::oneapi::plus<T>(), false, G, L);
  check_op<sycl_subgr<SpecializationKernelName, class KernelName_bWdCJaxe>, T>(
      Queue, T(0), ext::oneapi::plus<T>(), true, G, L);

  check_op<sycl_subgr<SpecializationKernelName, class KernelName_wjspvpHJtI>,
           T>(Queue, T(0), ext::oneapi::minimum<T>(), false, G, L);
  check_op<sycl_subgr<SpecializationKernelName, class KernelName_BUioaQYxhjN>,
           T>(Queue, T(G), ext::oneapi::minimum<T>(), true, G, L);

  check_op<sycl_subgr<SpecializationKernelName, class KernelName_bIHcoJBNpiB>,
           T>(Queue, T(G), ext::oneapi::maximum<T>(), false, G, L);
  check_op<sycl_subgr<SpecializationKernelName, class KernelName_bPPlfvdGShi>,
           T>(Queue, T(0), ext::oneapi::maximum<T>(), true, G, L);

  // Transparent operator functors.
  check_op<sycl_subgr<SpecializationKernelName,
                      class KernelName_fkOyLRYirfMnvBcnbRFy>,
           T>(Queue, T(L), ext::oneapi::plus<>(), false, G, L);
  check_op<sycl_subgr<SpecializationKernelName,
                      class KernelName_zhzfRmSAFlswKWShyecv>,
           T>(Queue, T(0), ext::oneapi::plus<>(), true, G, L);

  check_op<sycl_subgr<SpecializationKernelName,
                      class KernelName_NaOzDnOmDPiDIXnXvaGy>,
           T>(Queue, T(0), ext::oneapi::minimum<>(), false, G, L);
  check_op<sycl_subgr<SpecializationKernelName, class KernelName_XXAfdcNmCNX>,
           T>(Queue, T(G), ext::oneapi::minimum<>(), true, G, L);

  check_op<sycl_subgr<SpecializationKernelName, class KernelName_pLlvjjZsPv>,
           T>(Queue, T(G), ext::oneapi::maximum<>(), false, G, L);
  check_op<
      sycl_subgr<SpecializationKernelName, class KernelName_BaCGaWDMFeMFqvotbk>,
      T>(Queue, T(0), ext::oneapi::maximum<>(), true, G, L);
}

template <typename SpecializationKernelName, typename T>
void check_mul(queue &Queue, size_t G = 256, size_t L = 4) {
  check_op<sycl_subgr<SpecializationKernelName, class KernelName_MulF>, T>(
      Queue, T(G), ext::oneapi::multiplies<T>(), false, G, L);
  check_op<sycl_subgr<SpecializationKernelName, class KernelName_MulT>, T>(
      Queue, T(1), ext::oneapi::multiplies<T>(), true, G, L);

  // Transparent operator functors.
  check_op<sycl_subgr<SpecializationKernelName, class KernelName_MulFV>, T>(
      Queue, T(G), ext::oneapi::multiplies<>(), false, G, L);
  check_op<sycl_subgr<SpecializationKernelName, class KernelName_MulTV>, T>(
      Queue, T(1), ext::oneapi::multiplies<>(), true, G, L);
}

template <typename SpecializationKernelName, typename T>
void check_bit_ops(queue &Queue, size_t G = 256, size_t L = 4) {
  check_op<sycl_subgr<SpecializationKernelName, class KernelName_ORF>, T>(
      Queue, T(G), ext::oneapi::bit_or<T>(), false, G, L);
  check_op<sycl_subgr<SpecializationKernelName, class KernelName_ORT>, T>(
      Queue, T(0), ext::oneapi::bit_or<T>(), true, G, L);

  check_op<sycl_subgr<SpecializationKernelName, class KernelName_XORF>, T>(
      Queue, T(G), ext::oneapi::bit_xor<T>(), false, G, L);
  check_op<sycl_subgr<SpecializationKernelName, class KernelName_XORT>, T>(
      Queue, T(0), ext::oneapi::bit_xor<T>(), true, G, L);

  check_op<sycl_subgr<SpecializationKernelName, class KernelName_ANDF>, T>(
      Queue, T(G), ext::oneapi::bit_and<T>(), false, G, L);
  check_op<sycl_subgr<SpecializationKernelName, class KernelName_ANDT>, T>(
      Queue, ~T(0), ext::oneapi::bit_and<T>(), true, G, L);

  // Transparent operator functors
  check_op<sycl_subgr<SpecializationKernelName, class KernelName_ORFV>, T>(
      Queue, T(G), ext::oneapi::bit_or<T>(), false, G, L);
  check_op<sycl_subgr<SpecializationKernelName, class KernelName_ORTV>, T>(
      Queue, T(0), ext::oneapi::bit_or<T>(), true, G, L);

  check_op<sycl_subgr<SpecializationKernelName, class KernelName_XORFV>, T>(
      Queue, T(G), ext::oneapi::bit_xor<T>(), false, G, L);
  check_op<sycl_subgr<SpecializationKernelName, class KernelName_XORTV>, T>(
      Queue, T(0), ext::oneapi::bit_xor<T>(), true, G, L);

  check_op<sycl_subgr<SpecializationKernelName, class KernelName_ANDFV>, T>(
      Queue, T(G), ext::oneapi::bit_and<T>(), false, G, L);
  check_op<sycl_subgr<SpecializationKernelName, class KernelName_ANDTV>, T>(
      Queue, ~T(0), ext::oneapi::bit_and<T>(), true, G, L);
}
