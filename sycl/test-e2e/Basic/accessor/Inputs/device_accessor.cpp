//==---------- device_accessor.cpp - SYCL accessor basic test --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <cassert>
#include <sycl/detail/core.hpp>

int main() {
  // Non-placeholder accessors.
  {
    int data[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    sycl::buffer<int, 1> buf_data(data, sycl::range<1>(9),
                                  {sycl::property::buffer::use_host_ptr()});

    sycl::queue Queue;

    Queue.submit([&](sycl::handler &cgh) {

#if defined(accessor_new_api_test)
      sycl::accessor acc_1(buf_data, cgh);
      sycl::accessor acc_2(buf_data, cgh, sycl::range<1>(8));
      sycl::accessor acc_3(buf_data, cgh, sycl::range<1>(8), sycl::id<1>(1));
      sycl::accessor acc_4(buf_data, cgh, sycl::read_only);
      sycl::accessor acc_5(buf_data, cgh, sycl::range<1>(8), sycl::read_only);
      sycl::accessor acc_6(buf_data, cgh, sycl::range<1>(8), sycl::id<1>(1),
                           sycl::read_only);
      sycl::accessor acc_7(buf_data, cgh, sycl::write_only);
      sycl::accessor acc_8(buf_data, cgh, sycl::range<1>(8), sycl::write_only);
      sycl::accessor acc_9(buf_data, cgh, sycl::range<1>(8), sycl::id<1>(1),
                           sycl::write_only);
#elif defined(buffer_new_api_test)
      auto acc_1 = buf_data.get_access(cgh);
      auto acc_2 = buf_data.get_access(cgh, sycl::range<1>(8));
      auto acc_3 = buf_data.get_access(cgh, sycl::range<1>(8), sycl::id<1>(1));
      auto acc_4 = buf_data.get_access(cgh, sycl::read_only);
      auto acc_5 = buf_data.get_access(cgh, sycl::range<1>(8), sycl::read_only);
      auto acc_6 = buf_data.get_access(cgh, sycl::range<1>(8), sycl::id<1>(1),
                                       sycl::read_only);
      auto acc_7 = buf_data.get_access(cgh, sycl::write_only);
      auto acc_8 =
          buf_data.get_access(cgh, sycl::range<1>(8), sycl::write_only);
      auto acc_9 = buf_data.get_access(cgh, sycl::range<1>(8), sycl::id<1>(1),
                                       sycl::write_only);
#endif

      assert(!acc_1.is_placeholder());
      assert(!acc_2.is_placeholder());
      assert(!acc_3.is_placeholder());
      assert(!acc_4.is_placeholder());
      assert(!acc_5.is_placeholder());
      assert(!acc_6.is_placeholder());
      assert(!acc_7.is_placeholder());
      assert(!acc_8.is_placeholder());
      assert(!acc_9.is_placeholder());

      cgh.single_task<class nonplaceholder_kernel>([=]() {
        acc_7[6] = acc_1[0];
        acc_8[7] = acc_2[1];
        acc_9[7] = acc_3[1];
        acc_1[0] = acc_4[3];
        acc_2[1] = acc_5[4];
        acc_3[1] = acc_6[4];
      });
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

  // Placeholder accessors.
  {
    int data[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    sycl::buffer<int, 1> buf_data(data, sycl::range<1>(9),
                                  {sycl::property::buffer::use_host_ptr()});

#if defined(accessor_new_api_test)
    sycl::accessor acc_1(buf_data);
    sycl::accessor acc_2(buf_data, sycl::range<1>(8));
    sycl::accessor acc_3(buf_data, sycl::range<1>(8), sycl::id<1>(1));
    sycl::accessor acc_4(buf_data, sycl::read_only);
    sycl::accessor acc_5(buf_data, sycl::range<1>(8), sycl::read_only);
    sycl::accessor acc_6(buf_data, sycl::range<1>(8), sycl::id<1>(1),
                         sycl::read_only);
    sycl::accessor acc_7(buf_data, sycl::write_only);
    sycl::accessor acc_8(buf_data, sycl::range<1>(8), sycl::write_only);
    sycl::accessor acc_9(buf_data, sycl::range<1>(8), sycl::id<1>(1),
                         sycl::write_only);
#elif defined(buffer_new_api_test)
    auto acc_1 = buf_data.get_access();
    auto acc_2 = buf_data.get_access(sycl::range<1>(8));
    auto acc_3 = buf_data.get_access(sycl::range<1>(8), sycl::id<1>(1));
    auto acc_4 = buf_data.get_access(sycl::read_only);
    auto acc_5 = buf_data.get_access(sycl::range<1>(8), sycl::read_only);
    auto acc_6 =
        buf_data.get_access(sycl::range<1>(8), sycl::id<1>(1), sycl::read_only);
    auto acc_7 = buf_data.get_access(sycl::write_only);
    auto acc_8 = buf_data.get_access(sycl::range<1>(8), sycl::write_only);
    auto acc_9 = buf_data.get_access(sycl::range<1>(8), sycl::id<1>(1),
                                     sycl::write_only);
#endif

    assert(acc_1.is_placeholder());
    assert(acc_2.is_placeholder());
    assert(acc_3.is_placeholder());
    assert(acc_4.is_placeholder());
    assert(acc_5.is_placeholder());
    assert(acc_6.is_placeholder());
    assert(acc_7.is_placeholder());
    assert(acc_8.is_placeholder());
    assert(acc_9.is_placeholder());

    sycl::queue Queue;

    Queue.submit([&](sycl::handler &cgh) {
      cgh.require(acc_1);
      cgh.require(acc_2);
      cgh.require(acc_3);
      cgh.require(acc_4);
      cgh.require(acc_5);
      cgh.require(acc_6);
      cgh.require(acc_7);
      cgh.require(acc_8);
      cgh.require(acc_9);

      cgh.single_task<class placeholder_kernel>([=]() {
        acc_7[6] = acc_1[0];
        acc_8[7] = acc_2[1];
        acc_9[7] = acc_3[1];
        acc_1[0] = acc_4[3];
        acc_2[1] = acc_5[4];
        acc_3[1] = acc_6[4];
      });
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

  // Non-placeholder no_init and constant_buffer accessors.
  {
    int data[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    sycl::buffer<int, 1> buf_data(data, sycl::range<1>(9),
                                  {sycl::property::buffer::use_host_ptr()});

    sycl::queue Queue;

    Queue.submit([&](sycl::handler &cgh) {

#if defined(accessor_new_api_test)
      sycl::accessor acc_1(buf_data, cgh, sycl::no_init);
      sycl::accessor acc_2(buf_data, cgh, sycl::range<1>(8), sycl::no_init);
      sycl::accessor acc_3(buf_data, cgh, sycl::range<1>(8), sycl::id<1>(1),
                           sycl::no_init);
      sycl::accessor acc_4(buf_data, cgh, sycl::read_constant);
      sycl::accessor acc_5(buf_data, cgh, sycl::range<1>(8),
                           sycl::read_constant);
      sycl::accessor acc_6(buf_data, cgh, sycl::range<1>(8), sycl::id<1>(1),
                           sycl::read_constant);
      sycl::accessor acc_7(buf_data, cgh, sycl::write_only, sycl::no_init);
      sycl::accessor acc_8(buf_data, cgh, sycl::range<1>(8), sycl::write_only,
                           sycl::no_init);
      sycl::accessor acc_9(buf_data, cgh, sycl::range<1>(8), sycl::id<1>(1),
                           sycl::write_only, sycl::no_init);
#elif defined(buffer_new_api_test)
      auto acc_1 = buf_data.get_access(cgh, sycl::no_init);
      auto acc_2 = buf_data.get_access(cgh, sycl::range<1>(8), sycl::no_init);
      auto acc_3 = buf_data.get_access(cgh, sycl::range<1>(8), sycl::id<1>(1),
                                       sycl::no_init);
      auto acc_4 = buf_data.get_access(cgh, sycl::read_constant);
      auto acc_5 =
          buf_data.get_access(cgh, sycl::range<1>(8), sycl::read_constant);
      auto acc_6 = buf_data.get_access(cgh, sycl::range<1>(8), sycl::id<1>(1),
                                       sycl::read_constant);
      auto acc_7 = buf_data.get_access(cgh, sycl::write_only, sycl::no_init);
      auto acc_8 = buf_data.get_access(cgh, sycl::range<1>(8), sycl::write_only,
                                       sycl::no_init);
      auto acc_9 = buf_data.get_access(cgh, sycl::range<1>(8), sycl::id<1>(1),
                                       sycl::write_only, sycl::no_init);
#endif

      assert(!acc_1.is_placeholder());
      assert(!acc_2.is_placeholder());
      assert(!acc_3.is_placeholder());
      assert(!acc_4.is_placeholder());
      assert(!acc_5.is_placeholder());
      assert(!acc_6.is_placeholder());
      assert(!acc_7.is_placeholder());
      assert(!acc_8.is_placeholder());
      assert(!acc_9.is_placeholder());

      cgh.single_task<class nonplaceholder_no_init_kernel>([=]() {
        acc_7[6] = 1;
        acc_8[7] = 2;
        acc_9[7] = 3;
        acc_1[0] = acc_4[3];
        acc_2[1] = acc_5[4];
        acc_3[1] = acc_6[4];
      });
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

  // Placeholder no_init and constant_buffer accessors.
  {
    int data[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    sycl::buffer<int, 1> buf_data(data, sycl::range<1>(9),
                                  {sycl::property::buffer::use_host_ptr()});

#if defined(accessor_new_api_test)
    sycl::accessor acc_1(buf_data, sycl::no_init);
    sycl::accessor acc_2(buf_data, sycl::range<1>(8), sycl::no_init);
    sycl::accessor acc_3(buf_data, sycl::range<1>(8), sycl::id<1>(1),
                         sycl::no_init);
    sycl::accessor acc_4(buf_data, sycl::read_constant);
    sycl::accessor acc_5(buf_data, sycl::range<1>(8), sycl::read_constant);
    sycl::accessor acc_6(buf_data, sycl::range<1>(8), sycl::id<1>(1),
                         sycl::read_constant);
    sycl::accessor acc_7(buf_data, sycl::write_only, sycl::no_init);
    sycl::accessor acc_8(buf_data, sycl::range<1>(8), sycl::write_only,
                         sycl::no_init);
    sycl::accessor acc_9(buf_data, sycl::range<1>(8), sycl::id<1>(1),
                         sycl::write_only, sycl::no_init);
#elif defined(buffer_new_api_test)
    auto acc_1 = buf_data.get_access(sycl::no_init);
    auto acc_2 = buf_data.get_access(sycl::range<1>(8), sycl::no_init);
    auto acc_3 =
        buf_data.get_access(sycl::range<1>(8), sycl::id<1>(1), sycl::no_init);
    auto acc_4 = buf_data.get_access(sycl::read_constant);
    auto acc_5 = buf_data.get_access(sycl::range<1>(8), sycl::read_constant);
    auto acc_6 = buf_data.get_access(sycl::range<1>(8), sycl::id<1>(1),
                                     sycl::read_constant);
    auto acc_7 = buf_data.get_access(sycl::write_only, sycl::no_init);
    auto acc_8 =
        buf_data.get_access(sycl::range<1>(8), sycl::write_only, sycl::no_init);
    auto acc_9 = buf_data.get_access(sycl::range<1>(8), sycl::id<1>(1),
                                     sycl::write_only, sycl::no_init);
#endif

    assert(acc_1.is_placeholder());
    assert(acc_2.is_placeholder());
    assert(acc_3.is_placeholder());
    assert(acc_4.is_placeholder());
    assert(acc_5.is_placeholder());
    assert(acc_6.is_placeholder());
    assert(acc_7.is_placeholder());
    assert(acc_8.is_placeholder());
    assert(acc_9.is_placeholder());

    sycl::queue Queue;

    Queue.submit([&](sycl::handler &cgh) {
      cgh.require(acc_1);
      cgh.require(acc_2);
      cgh.require(acc_3);
      cgh.require(acc_4);
      cgh.require(acc_5);
      cgh.require(acc_6);
      cgh.require(acc_7);
      cgh.require(acc_8);
      cgh.require(acc_9);

      cgh.single_task<class placeholder_no_init_kernel>([=]() {
        acc_7[6] = 1;
        acc_8[7] = 2;
        acc_9[7] = 3;
        acc_1[0] = acc_4[3];
        acc_2[1] = acc_5[4];
        acc_3[1] = acc_6[4];
      });
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
