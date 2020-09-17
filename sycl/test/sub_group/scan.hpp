//==--------------- scan.hpp - SYCL sub_group scan test --------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "helper.hpp"
#include <CL/sycl.hpp>
#include <limits>

template <typename... Ts> class sycl_subgr;

using namespace cl::sycl;

template <typename SpecializationKernelName, typename T, class BinaryOperation>
void check_op(queue &Queue, T init, BinaryOperation op, bool skip_init = false,
              size_t G = 256, size_t L = 64) {
  try {
    nd_range<1> NdRange(G, L);
    buffer<T> exbuf(G), inbuf(G);
    buffer<size_t> sgsizebuf(1);
    Queue.submit([&](handler &cgh) {
      auto sgsizeacc = sgsizebuf.get_access<access::mode::read_write>(cgh);
      auto exacc = exbuf.template get_access<access::mode::read_write>(cgh);
      auto inacc = inbuf.template get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<SpecializationKernelName>(
          NdRange, [=](nd_item<1> NdItem) {
            ONEAPI::sub_group sg = NdItem.get_sub_group();
            if (skip_init) {
              exacc[NdItem.get_global_id(0)] =
                  exclusive_scan(sg, T(NdItem.get_global_id(0)), op);
              inacc[NdItem.get_global_id(0)] =
                  inclusive_scan(sg, T(NdItem.get_global_id(0)), op);
            } else {
              exacc[NdItem.get_global_id(0)] =
                  exclusive_scan(sg, T(NdItem.get_global_id(0)), init, op);
              inacc[NdItem.get_global_id(0)] =
                  inclusive_scan(sg, T(NdItem.get_global_id(0)), op, init);
            }
            if (NdItem.get_global_id(0) == 0)
              sgsizeacc[0] = sg.get_max_local_range()[0];
          });
    });
    auto exacc = exbuf.template get_access<access::mode::read_write>();
    auto inacc = inbuf.template get_access<access::mode::read_write>();
    auto sgsizeacc = sgsizebuf.get_access<access::mode::read_write>();
    size_t sg_size = sgsizeacc[0];
    int WGid = -1, SGid = 0;
    T result = init;
    for (int j = 0; j < G; j++) {
      if (j % L % sg_size == 0) {
        SGid++;
        result = init;
      }
      if (j % L == 0) {
        WGid++;
        SGid = 0;
      }
      std::string exname =
          std::string("scan_exc_") + typeid(BinaryOperation).name();
      std::string inname =
          std::string("scan_inc_") + typeid(BinaryOperation).name();
      exit_if_not_equal<T>(exacc[j], result, exname.c_str());
      result = op(result, T(j));
      exit_if_not_equal<T>(inacc[j], result, inname.c_str());
    }
  } catch (exception e) {
    std::cout << "SYCL exception caught: " << e.what();
    exit(1);
  }
}

template <typename SpecializationKernelName, typename T>
void check(queue &Queue, size_t G = 256, size_t L = 64) {
  // limit data range for half to avoid rounding issues
  if (std::is_same<T, cl::sycl::half>::value) {
    G = 64;
    L = 32;
  }

  check_op<sycl_subgr<SpecializationKernelName, class KernelName_UdKabTMplbvM>,
           T>(Queue, T(L), ONEAPI::plus<T>(), false, G, L);
  check_op<sycl_subgr<SpecializationKernelName, class KernelName_hYvJ>, T>(
      Queue, T(0), ONEAPI::plus<T>(), true, G, L);

  check_op<
      sycl_subgr<SpecializationKernelName, class KernelName_eozPcciiaOmKkKUEp>,
      T>(Queue, T(0), ONEAPI::minimum<T>(), false, G, L);
  if (std::is_floating_point<T>::value ||
      std::is_same<T, cl::sycl::half>::value) {
    check_op<
        sycl_subgr<SpecializationKernelName, class KernelName_LylCkHSTmrFhMH>,
        T>(Queue, std::numeric_limits<T>::infinity(), ONEAPI::minimum<T>(),
           true, G, L);
  } else {
    check_op<sycl_subgr<SpecializationKernelName,
                        class KernelName_gYWXQQXGnzJEpaftEQly>,
             T>(Queue, std::numeric_limits<T>::max(), ONEAPI::minimum<T>(),
                true, G, L);
  }

  check_op<
      sycl_subgr<SpecializationKernelName, class KernelName_NEgmAHtvPAWDyXPoo>,
      T>(Queue, T(G), ONEAPI::maximum<T>(), false, G, L);
  if (std::is_floating_point<T>::value ||
      std::is_same<T, cl::sycl::half>::value) {
    check_op<
        sycl_subgr<SpecializationKernelName, class KernelName_EBNigvpxbxYEyRcl>,
        T>(Queue, -std::numeric_limits<T>::infinity(), ONEAPI::maximum<T>(),
           true, G, L);
  } else {
    check_op<sycl_subgr<SpecializationKernelName, class KernelName_KayihC>, T>(
        Queue, std::numeric_limits<T>::min(), ONEAPI::maximum<T>(), true, G, L);
  }

#if __cplusplus >= 201402L
  check_op<sycl_subgr<SpecializationKernelName, class KernelName_TPWS>, T>(
      Queue, T(L), ONEAPI::plus<>(), false, G, L);
  check_op<sycl_subgr<SpecializationKernelName, class KernelName_hWZv>, T>(
      Queue, T(0), ONEAPI::plus<>(), true, G, L);

  check_op<
      sycl_subgr<SpecializationKernelName, class KernelName_MdoesLriZMCljse>,
      T>(Queue, T(0), ONEAPI::minimum<>(), false, G, L);
  if (std::is_floating_point<T>::value ||
      std::is_same<T, cl::sycl::half>::value) {
    check_op<
        sycl_subgr<SpecializationKernelName, class KernelName_fgMMknFqTMGts>,
        T>(Queue, std::numeric_limits<T>::infinity(), ONEAPI::minimum<>(), true,
           G, L);
  } else {
    check_op<sycl_subgr<SpecializationKernelName,
                        class KernelName_FVbXDSctbMnggHMCz>,
             T>(Queue, std::numeric_limits<T>::max(), ONEAPI::minimum<>(), true,
                G, L);
  }

  check_op<sycl_subgr<SpecializationKernelName, class KernelName_zzvRru>, T>(
      Queue, T(G), ONEAPI::maximum<>(), false, G, L);
  if (std::is_floating_point<T>::value ||
      std::is_same<T, cl::sycl::half>::value) {
    check_op<sycl_subgr<SpecializationKernelName, class KernelName_NJh>, T>(
        Queue, -std::numeric_limits<T>::infinity(), ONEAPI::maximum<>(), true,
        G, L);
  } else {
    check_op<
        sycl_subgr<SpecializationKernelName, class KernelName_XjMHvRfLSQerFi>,
        T>(Queue, std::numeric_limits<T>::min(), ONEAPI::maximum<>(), true, G,
           L);
  }
#endif
}
