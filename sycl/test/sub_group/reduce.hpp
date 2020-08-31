//==--------------- reduce.hpp - SYCL sub_group reduce test ----*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "helper.hpp"
#include <CL/sycl.hpp>

template <typename T, class BinaryOperation> class sycl_subgr;

using namespace cl::sycl;

template <typename T, class BinaryOperation>
void check_op(queue &Queue, T init, BinaryOperation op, bool skip_init = false,
              size_t G = 256, size_t L = 64) {
  try {
    nd_range<1> NdRange(G, L);
    buffer<T> buf(G);
    buffer<size_t> sgsizebuf(1);
    Queue.submit([&](handler &cgh) {
      auto sgsizeacc = sgsizebuf.get_access<access::mode::read_write>(cgh);
      auto acc = buf.template get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<sycl_subgr<T, BinaryOperation>>(
          NdRange, [=](nd_item<1> NdItem) {
            ONEAPI::sub_group sg = NdItem.get_sub_group();
            if (skip_init) {
              acc[NdItem.get_global_id(0)] =
                  reduce(sg, T(NdItem.get_global_id(0)), op);
            } else {
              acc[NdItem.get_global_id(0)] =
                  reduce(sg, T(NdItem.get_global_id(0)), init, op);
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

template <typename T> void check(queue &Queue, size_t G = 256, size_t L = 64) {
  // limit data range for half to avoid rounding issues
  if (std::is_same<T, cl::sycl::half>::value) {
    G = 64;
    L = 32;
  }

  check_op<T>(Queue, T(L), ONEAPI::plus<T>(), false, G, L);
  check_op<T>(Queue, T(0), ONEAPI::plus<T>(), true, G, L);

  check_op<T>(Queue, T(0), ONEAPI::minimum<T>(), false, G, L);
  check_op<T>(Queue, T(G), ONEAPI::minimum<T>(), true, G, L);

  check_op<T>(Queue, T(G), ONEAPI::maximum<T>(), false, G, L);
  check_op<T>(Queue, T(0), ONEAPI::maximum<T>(), true, G, L);

#if __cplusplus >= 201402L
  check_op<T>(Queue, T(L), ONEAPI::plus<>(), false, G, L);
  check_op<T>(Queue, T(0), ONEAPI::plus<>(), true, G, L);

  check_op<T>(Queue, T(0), ONEAPI::minimum<>(), false, G, L);
  check_op<T>(Queue, T(G), ONEAPI::minimum<>(), true, G, L);

  check_op<T>(Queue, T(G), ONEAPI::maximum<>(), false, G, L);
  check_op<T>(Queue, T(0), ONEAPI::maximum<>(), true, G, L);
#endif
}
