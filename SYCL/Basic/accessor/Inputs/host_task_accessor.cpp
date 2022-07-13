//==-------- host_task_accessor.cpp - SYCL accessor basic test -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <cassert>
#include <sycl/sycl.hpp>

int main() {
  // Non-placeholder accessors.
  {
    int data[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    sycl::buffer<int, 1> buf_data(data, sycl::range<1>(9),
                                  {cl::sycl::property::buffer::use_host_ptr()});

    sycl::queue Queue;

    Queue.submit([&](sycl::handler &cgh) {

#if defined(accessor_new_api_test)
      sycl::host_accessor acc_1(buf_data, cgh);
      sycl::host_accessor acc_2(buf_data, cgh, sycl::range<1>(8));
      sycl::host_accessor acc_3(buf_data, cgh, sycl::range<1>(8),
                                sycl::id<1>(1));
      sycl::host_accessor acc_4(buf_data, cgh, sycl::read_only);
      sycl::host_accessor acc_5(buf_data, cgh, sycl::range<1>(8),
                                sycl::read_only);
      sycl::host_accessor acc_6(buf_data, cgh, sycl::range<1>(8),
                                sycl::id<1>(1), sycl::read_only);
      sycl::host_accessor acc_7(buf_data, cgh, sycl::write_only);
      sycl::host_accessor acc_8(buf_data, cgh, sycl::range<1>(8),
                                sycl::write_only);
      sycl::host_accessor acc_9(buf_data, cgh, sycl::range<1>(8),
                                sycl::id<1>(1), sycl::write_only);
#elif defined(buffer_new_api_test)
      auto acc_1 = buf_data.get_host_access(cgh);
      auto acc_2 = buf_data.get_host_access(cgh, sycl::range<1>(8));
      auto acc_3 =
          buf_data.get_host_access(cgh, sycl::range<1>(8), sycl::id<1>(1));
      auto acc_4 = buf_data.get_host_access(cgh, sycl::read_only);
      auto acc_5 =
          buf_data.get_host_access(cgh, sycl::range<1>(8), sycl::read_only);
      auto acc_6 = buf_data.get_host_access(cgh, sycl::range<1>(8),
                                            sycl::id<1>(1), sycl::read_only);
      auto acc_7 = buf_data.get_host_access(cgh, sycl::write_only);
      auto acc_8 =
          buf_data.get_host_access(cgh, sycl::range<1>(8), sycl::write_only);
      auto acc_9 = buf_data.get_host_access(cgh, sycl::range<1>(8),
                                            sycl::id<1>(1), sycl::write_only);
#endif

#if defined(accessor_new_api_test) || defined(buffer_new_api_test)
      cgh.host_task([=]() {
        acc_7[6] = acc_1[0];
        acc_8[7] = acc_2[1];
        acc_9[7] = acc_3[1];
        acc_1[0] = acc_4[3];
        acc_2[1] = acc_5[4];
        acc_3[1] = acc_6[4];
      });
    });
#endif
    Queue.wait();

#if defined(accessor_new_api_test)
    sycl::host_accessor host_acc(buf_data, sycl::read_only);
#elif defined(buffer_new_api_test)
    auto host_acc = buf_data.get_host_access(sycl::read_only);
#endif
    assert(host_acc[0] == 4 && host_acc[1] == 5 && host_acc[2] == 6);
    assert(host_acc[3] == 4 && host_acc[4] == 5 && host_acc[5] == 6);
    assert(host_acc[6] == 1 && host_acc[7] == 2 && host_acc[8] == 3);
  }

  // noinit accessors.
  {
    int data[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    sycl::buffer<int, 1> buf_data(data, sycl::range<1>(9),
                                  {cl::sycl::property::buffer::use_host_ptr()});

    sycl::queue Queue;

    Queue.submit([&](sycl::handler &cgh) {

#if defined(accessor_new_api_test)
      sycl::host_accessor acc_1(buf_data, cgh, sycl::noinit);
      sycl::host_accessor acc_2(buf_data, cgh, sycl::range<1>(8), sycl::noinit);
      sycl::host_accessor acc_3(buf_data, cgh, sycl::range<1>(8),
                                sycl::id<1>(1), sycl::noinit);
      sycl::host_accessor acc_7(buf_data, cgh, sycl::write_only, sycl::noinit);
      sycl::host_accessor acc_8(buf_data, cgh, sycl::range<1>(8),
                                sycl::write_only, sycl::noinit);
      sycl::host_accessor acc_9(buf_data, cgh, sycl::range<1>(8),
                                sycl::id<1>(1), sycl::write_only, sycl::noinit);
#elif defined(buffer_new_api_test)
      auto acc_1 = buf_data.get_host_access(cgh, sycl::noinit);
      auto acc_2 =
          buf_data.get_host_access(cgh, sycl::range<1>(8), sycl::noinit);
      auto acc_3 = buf_data.get_host_access(cgh, sycl::range<1>(8),
                                            sycl::id<1>(1), sycl::noinit);
      auto acc_7 =
          buf_data.get_host_access(cgh, sycl::write_only, sycl::noinit);
      auto acc_8 = buf_data.get_host_access(cgh, sycl::range<1>(8),
                                            sycl::write_only, sycl::noinit);
      auto acc_9 =
          buf_data.get_host_access(cgh, sycl::range<1>(8), sycl::id<1>(1),
                                   sycl::write_only, sycl::noinit);
#endif

#if defined(accessor_new_api_test) || defined(buffer_new_api_test)
      cgh.host_task([=]() {
        acc_7[6] = acc_1[0];
        acc_8[7] = acc_2[1];
        acc_9[7] = acc_3[1];
        acc_1[0] = 4;
        acc_2[1] = 5;
        acc_3[1] = 6;
      });
#endif
    });
    Queue.wait();

#if defined(accessor_new_api_test)
    sycl::host_accessor host_acc(buf_data, sycl::read_only);
#elif defined(buffer_new_api_test)
    auto host_acc = buf_data.get_host_access(sycl::read_only);
#endif
    assert(host_acc[0] == 4 && host_acc[1] == 5 && host_acc[2] == 6);
    assert(host_acc[3] == 4 && host_acc[4] == 5 && host_acc[5] == 6);
    assert(host_acc[6] == 1 && host_acc[7] == 2 && host_acc[8] == 3);
  }
}
