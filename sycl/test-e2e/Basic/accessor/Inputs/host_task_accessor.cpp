//==-------- host_task_accessor.cpp - SYCL accessor basic test -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <cassert>
#include <sycl/detail/core.hpp>

#if !defined(accessor_new_api_test) && !defined(buffer_new_api_test) &&        \
    !defined(accessor_placeholder_new_api_test) &&                             \
    !defined(buffer_placeholder_new_api_test)
#error Missing definition
#endif

int main() {
  // Non-placeholder accessors.
  {
    int data[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    sycl::buffer<int, 1> buf_data(data, sycl::range<1>(9),
                                  {sycl::property::buffer::use_host_ptr()});

    sycl::queue Queue;

#if defined(accessor_placeholder_new_api_test)
    sycl::accessor acc_1(buf_data, sycl::read_write_host_task);
    sycl::accessor acc_2(buf_data, sycl::range<1>(8),
                         sycl::read_write_host_task);
    sycl::accessor acc_3(buf_data, sycl::range<1>(8), sycl::id<1>(1),
                         sycl::read_write_host_task);
    sycl::accessor acc_4(buf_data, sycl::read_only_host_task);
    sycl::accessor acc_5(buf_data, sycl::range<1>(8),
                         sycl::read_only_host_task);
    sycl::accessor acc_6(buf_data, sycl::range<1>(8), sycl::id<1>(1),
                         sycl::read_only_host_task);
    sycl::accessor acc_7(buf_data, sycl::write_only_host_task);
    sycl::accessor acc_8(buf_data, sycl::range<1>(8),
                         sycl::write_only_host_task);
    sycl::accessor acc_9(buf_data, sycl::range<1>(8), sycl::id<1>(1),
                         sycl::write_only_host_task);
#elif defined(buffer_placeholder_new_api_test)
    auto acc_1 = buf_data.get_access(sycl::read_write_host_task);
    auto acc_2 =
        buf_data.get_access(sycl::range<1>(8), sycl::read_write_host_task);
    auto acc_3 = buf_data.get_access(sycl::range<1>(8), sycl::id<1>(1),
                                     sycl::read_write_host_task);
    auto acc_4 = buf_data.get_access(sycl::read_only_host_task);
    auto acc_5 =
        buf_data.get_access(sycl::range<1>(8), sycl::read_only_host_task);
    auto acc_6 = buf_data.get_access(sycl::range<1>(8), sycl::id<1>(1),
                                     sycl::read_only_host_task);
    auto acc_7 = buf_data.get_access(sycl::write_only_host_task);
    auto acc_8 =
        buf_data.get_access(sycl::range<1>(8), sycl::write_only_host_task);
    auto acc_9 = buf_data.get_access(sycl::range<1>(8), sycl::id<1>(1),
                                     sycl::write_only_host_task);
#endif

    Queue.submit([&](sycl::handler &cgh) {

#if defined(accessor_new_api_test)
      sycl::accessor acc_1(buf_data, cgh, sycl::read_write_host_task);
      sycl::accessor acc_2(buf_data, cgh, sycl::range<1>(8),
                           sycl::read_write_host_task);
      sycl::accessor acc_3(buf_data, cgh, sycl::range<1>(8), sycl::id<1>(1),
                           sycl::read_write_host_task);
      sycl::accessor acc_4(buf_data, cgh, sycl::read_only_host_task);
      sycl::accessor acc_5(buf_data, cgh, sycl::range<1>(8),
                           sycl::read_only_host_task);
      sycl::accessor acc_6(buf_data, cgh, sycl::range<1>(8), sycl::id<1>(1),
                           sycl::read_only_host_task);
      sycl::accessor acc_7(buf_data, cgh, sycl::write_only_host_task);
      sycl::accessor acc_8(buf_data, cgh, sycl::range<1>(8),
                           sycl::write_only_host_task);
      sycl::accessor acc_9(buf_data, cgh, sycl::range<1>(8), sycl::id<1>(1),
                           sycl::write_only_host_task);
#elif defined(buffer_new_api_test)
      auto acc_1 = buf_data.get_access(cgh, sycl::read_write_host_task);
      auto acc_2 = buf_data.get_access(cgh, sycl::range<1>(8),
                                       sycl::read_write_host_task);
      auto acc_3 = buf_data.get_access(cgh, sycl::range<1>(8), sycl::id<1>(1),
                                       sycl::read_write_host_task);
      auto acc_4 = buf_data.get_access(cgh, sycl::read_only_host_task);
      auto acc_5 = buf_data.get_access(cgh, sycl::range<1>(8),
                                       sycl::read_only_host_task);
      auto acc_6 = buf_data.get_access(cgh, sycl::range<1>(8), sycl::id<1>(1),
                                       sycl::read_only_host_task);
      auto acc_7 = buf_data.get_access(cgh, sycl::write_only_host_task);
      auto acc_8 = buf_data.get_access(cgh, sycl::range<1>(8),
                                       sycl::write_only_host_task);
      auto acc_9 = buf_data.get_access(cgh, sycl::range<1>(8), sycl::id<1>(1),
                                       sycl::write_only_host_task);
#elif defined(accessor_placeholder_new_api_test) ||                            \
    defined(buffer_placeholder_new_api_test)
      cgh.require(acc_1);
      cgh.require(acc_2);
      cgh.require(acc_3);
      cgh.require(acc_4);
      cgh.require(acc_5);
      cgh.require(acc_6);
      cgh.require(acc_7);
      cgh.require(acc_8);
      cgh.require(acc_9);
#endif

      cgh.host_task([=]() {
        acc_7[6] = acc_1[0];
        acc_8[7] = acc_2[1];
        acc_9[7] = acc_3[1];
        acc_1[0] = acc_4[3];
        acc_2[1] = acc_5[4];
        acc_3[1] = acc_6[4];
      });
    });
    Queue.wait();

    sycl::host_accessor host_acc(buf_data, sycl::read_only);
    assert(host_acc[0] == 4 && host_acc[1] == 5 && host_acc[2] == 6);
    assert(host_acc[3] == 4 && host_acc[4] == 5 && host_acc[5] == 6);
    assert(host_acc[6] == 1 && host_acc[7] == 2 && host_acc[8] == 3);
  }

  // noinit accessors.
  {
    int data[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    sycl::buffer<int, 1> buf_data(data, sycl::range<1>(9),
                                  {sycl::property::buffer::use_host_ptr()});

    sycl::queue Queue;

#if defined(accessor_placeholder_new_api_test)
    sycl::accessor acc_1(buf_data, sycl::read_write_host_task, sycl::no_init);
    sycl::accessor acc_2(buf_data, sycl::range<1>(8),
                         sycl::read_write_host_task, sycl::no_init);
    sycl::accessor acc_3(buf_data, sycl::range<1>(8), sycl::id<1>(1),
                         sycl::read_write_host_task, sycl::no_init);
    sycl::accessor acc_7(buf_data, sycl::write_only_host_task, sycl::no_init);
    sycl::accessor acc_8(buf_data, sycl::range<1>(8),
                         sycl::write_only_host_task, sycl::no_init);
    sycl::accessor acc_9(buf_data, sycl::range<1>(8), sycl::id<1>(1),
                         sycl::write_only_host_task, sycl::no_init);
#elif defined(buffer_placeholder_new_api_test)
    auto acc_1 = buf_data.get_access(sycl::read_write_host_task, sycl::no_init);
    auto acc_2 = buf_data.get_access(sycl::range<1>(8),
                                     sycl::read_write_host_task, sycl::no_init);
    auto acc_3 = buf_data.get_access(sycl::range<1>(8), sycl::id<1>(1),
                                     sycl::read_write_host_task, sycl::no_init);
    auto acc_7 = buf_data.get_access(sycl::write_only_host_task, sycl::no_init);
    auto acc_8 = buf_data.get_access(sycl::range<1>(8),
                                     sycl::write_only_host_task, sycl::no_init);
    auto acc_9 = buf_data.get_access(sycl::range<1>(8), sycl::id<1>(1),
                                     sycl::write_only_host_task, sycl::no_init);
#endif

    Queue.submit([&](sycl::handler &cgh) {

#if defined(accessor_new_api_test)
      sycl::accessor acc_1(buf_data, cgh, sycl::read_write_host_task,
                           sycl::no_init);
      sycl::accessor acc_2(buf_data, cgh, sycl::range<1>(8),
                           sycl::read_write_host_task, sycl::no_init);
      sycl::accessor acc_3(buf_data, cgh, sycl::range<1>(8), sycl::id<1>(1),
                           sycl::read_write_host_task, sycl::no_init);
      sycl::accessor acc_7(buf_data, cgh, sycl::write_only_host_task,
                           sycl::no_init);
      sycl::accessor acc_8(buf_data, cgh, sycl::range<1>(8),
                           sycl::write_only_host_task, sycl::no_init);
      sycl::accessor acc_9(buf_data, cgh, sycl::range<1>(8), sycl::id<1>(1),
                           sycl::write_only_host_task, sycl::no_init);
#elif defined(buffer_new_api_test)
      auto acc_1 =
          buf_data.get_access(cgh, sycl::read_write_host_task, sycl::no_init);
      auto acc_2 = buf_data.get_access(
          cgh, sycl::range<1>(8), sycl::read_write_host_task, sycl::no_init);
      auto acc_3 =
          buf_data.get_access(cgh, sycl::range<1>(8), sycl::id<1>(1),
                              sycl::read_write_host_task, sycl::no_init);
      auto acc_7 =
          buf_data.get_access(cgh, sycl::write_only_host_task, sycl::no_init);
      auto acc_8 = buf_data.get_access(
          cgh, sycl::range<1>(8), sycl::write_only_host_task, sycl::no_init);
      auto acc_9 =
          buf_data.get_access(cgh, sycl::range<1>(8), sycl::id<1>(1),
                              sycl::write_only_host_task, sycl::no_init);
#elif defined(accessor_placeholder_new_api_test) ||                            \
    defined(buffer_placeholder_new_api_test)
      cgh.require(acc_1);
      cgh.require(acc_2);
      cgh.require(acc_3);
      cgh.require(acc_7);
      cgh.require(acc_8);
      cgh.require(acc_9);
#endif

      cgh.host_task([=]() {
        acc_7[6] = acc_1[0];
        acc_8[7] = acc_2[1];
        acc_9[7] = acc_3[1];
        acc_1[0] = 4;
        acc_2[1] = 5;
        acc_3[1] = 6;
      });
    });
    Queue.wait();

    sycl::host_accessor host_acc(buf_data, sycl::read_only);
    assert(host_acc[0] == 4 && host_acc[1] == 5 && host_acc[2] == 6);
    assert(host_acc[3] == 4 && host_acc[4] == 5 && host_acc[5] == 6);
    assert(host_acc[6] == 1 && host_acc[7] == 2 && host_acc[8] == 3);
  }
}
