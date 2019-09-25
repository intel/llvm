// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %clangxx -fsycl -D SG_GPU %s -o %t_gpu.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t_gpu.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//==--------------- reduce.cpp - SYCL sub_group reduce test ----*- C++ -*---==//
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
              size_t G = 240, size_t L = 60) {
  try {
    nd_range<1> NdRange(G, L);
    buffer<T> buf(G);
    Queue.submit([&](handler &cgh) {
      auto acc = buf.template get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<sycl_subgr<T, BinaryOperation>>(
          NdRange, [=](nd_item<1> NdItem) {
            intel::sub_group sg = NdItem.get_sub_group();
            if (skip_init) {
              acc[NdItem.get_global_id(0)] =
                  sg.reduce(T(NdItem.get_global_id(0)), op);
            } else {
              acc[NdItem.get_global_id(0)] =
                  sg.reduce(T(NdItem.get_global_id(0)), init, op);
            }
          });
    });
    auto acc = buf.template get_access<access::mode::read_write>();
    size_t sg_size = get_sg_size(Queue.get_device());
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

template <typename T> void check(queue &Queue, size_t G = 240, size_t L = 60) {
  // limit data range for half to avoid rounding issues
  if (std::is_same<T, cl::sycl::half>::value) {
    G = 64;
    L = 32;
  }

  check_op<T>(Queue, T(L), intel::plus<T>(), false, G, L);
  check_op<T>(Queue, T(L), intel::plus<>(), false, G, L);
  check_op<T>(Queue, T(0), intel::plus<T>(), true, G, L);
  check_op<T>(Queue, T(0), intel::plus<>(), true, G, L);

  check_op<T>(Queue, T(0), intel::minimum<T>(), false, G, L);
  check_op<T>(Queue, T(0), intel::minimum<>(), false, G, L);
  check_op<T>(Queue, T(G), intel::minimum<T>(), true, G, L);
  check_op<T>(Queue, T(G), intel::minimum<>(), true, G, L);

  check_op<T>(Queue, T(G), intel::maximum<T>(), false, G, L);
  check_op<T>(Queue, T(G), intel::maximum<>(), false, G, L);
  check_op<T>(Queue, T(0), intel::maximum<T>(), true, G, L);
  check_op<T>(Queue, T(0), intel::maximum<>(), true, G, L);
}

int main() {
  queue Queue;
  if (!core_sg_supported(Queue.get_device())) {
    std::cout << "Skipping test\n";
    return 0;
  }

  check<int>(Queue);
  check<unsigned int>(Queue);
  check<long>(Queue);
  check<unsigned long>(Queue);
  check<float>(Queue);
  // reduce half type is not supported in OCL CPU RT
#ifdef SG_GPU
  if (Queue.get_device().has_extension("cl_khr_fp16")) {
    check<cl::sycl::half>(Queue);
  }
#endif
  if (Queue.get_device().has_extension("cl_khr_fp64")) {
    check<double>(Queue);
  }
  std::cout << "Test passed." << std::endl;
  return 0;
}
