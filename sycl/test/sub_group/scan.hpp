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

template <typename T, class BinaryOperation> class sycl_subgr;

using namespace cl::sycl;

template <typename T, class BinaryOperation>
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
      cgh.parallel_for<sycl_subgr<T, BinaryOperation>>(
          NdRange, [=](nd_item<1> NdItem) {
            intel::sub_group sg = NdItem.get_sub_group();
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

template <typename T> void check(queue &Queue, size_t G = 256, size_t L = 64) {
  // limit data range for half to avoid rounding issues
  if (std::is_same<T, cl::sycl::half>::value) {
    G = 64;
    L = 32;
  }

  check_op<T>(Queue, T(L), intel::plus<T>(), false, G, L);
  check_op<T>(Queue, T(0), intel::plus<T>(), true, G, L);

  check_op<T>(Queue, T(0), intel::minimum<T>(), false, G, L);
  if (std::is_floating_point<T>::value ||
      std::is_same<T, cl::sycl::half>::value) {
    check_op<T>(Queue, std::numeric_limits<T>::infinity(), intel::minimum<T>(),
                true, G, L);
  } else {
    check_op<T>(Queue, std::numeric_limits<T>::max(), intel::minimum<T>(), true,
                G, L);
  }

  check_op<T>(Queue, T(G), intel::maximum<T>(), false, G, L);
  if (std::is_floating_point<T>::value ||
      std::is_same<T, cl::sycl::half>::value) {
    check_op<T>(Queue, -std::numeric_limits<T>::infinity(), intel::maximum<T>(),
                true, G, L);
  } else {
    check_op<T>(Queue, std::numeric_limits<T>::min(), intel::maximum<T>(), true,
                G, L);
  }

#if __cplusplus >= 201402L
  check_op<T>(Queue, T(L), intel::plus<>(), false, G, L);
  check_op<T>(Queue, T(0), intel::plus<>(), true, G, L);

  check_op<T>(Queue, T(0), intel::minimum<>(), false, G, L);
  if (std::is_floating_point<T>::value ||
      std::is_same<T, cl::sycl::half>::value) {
    check_op<T>(Queue, std::numeric_limits<T>::infinity(), intel::minimum<>(),
                true, G, L);
  } else {
    check_op<T>(Queue, std::numeric_limits<T>::max(), intel::minimum<>(), true,
                G, L);
  }

  check_op<T>(Queue, T(G), intel::maximum<>(), false, G, L);
  if (std::is_floating_point<T>::value ||
      std::is_same<T, cl::sycl::half>::value) {
    check_op<T>(Queue, -std::numeric_limits<T>::infinity(), intel::maximum<>(),
                true, G, L);
  } else {
    check_op<T>(Queue, std::numeric_limits<T>::min(), intel::maximum<>(), true,
                G, L);
  }
#endif
}
