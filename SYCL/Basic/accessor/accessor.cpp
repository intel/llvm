// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

//==----------------accessor.cpp - SYCL accessor basic test ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <cassert>
#include <iostream>
#include <sycl/sycl.hpp>

struct IdxID1 {
  int x;

  IdxID1(int x) : x(x) {}
  operator sycl::id<1>() { return x; }
};

struct IdxID3 {
  int x;
  int y;
  int z;

  IdxID3(int x, int y, int z) : x(x), y(y), z(z) {}
  operator sycl::id<3>() { return sycl::id<3>(x, y, z); }
};

template <typename T>
using AccAlias =
    sycl::accessor<T, 1, sycl::access::mode::write, sycl::target::device>;

template <typename T> struct InheritedAccessor : public AccAlias<T> {

  using AccAlias<T>::AccAlias;
};

template <typename Acc> struct AccWrapper { Acc accessor; };

template <typename Acc1, typename Acc2> struct AccsWrapper {
  int a;
  Acc1 accessor1;
  int b;
  Acc2 accessor2;
};

struct Wrapper1 {
  int a;
  int b;
};

template <typename Acc> struct Wrapper2 {
  Wrapper1 w1;
  AccWrapper<Acc> wrapped;
};

template <typename Acc> struct Wrapper3 { Wrapper2<Acc> w2; };

int main() {
  // Host accessor.
  {
    int src[2] = {3, 7};
    int dst[2];

    sycl::buffer<int, 1> buf_src(src, sycl::range<1>(2),
                                 {sycl::property::buffer::use_host_ptr()});
    sycl::buffer<int, 1> buf_dst(dst, sycl::range<1>(2),
                                 {sycl::property::buffer::use_host_ptr()});

    sycl::id<1> id1(1);
    auto acc_src = buf_src.get_access<sycl::access::mode::read>();
    auto acc_dst = buf_dst.get_access<sycl::access::mode::read_write>();

    assert(!acc_src.is_placeholder());
    assert(acc_src.get_size() == sizeof(src));
    assert(acc_src.get_count() == 2);
    assert(acc_src.get_range() == sycl::range<1>(2));

    // operator[] overload for size_t was intentionally removed
    // to remove ambiguity, when passing item to operator[].
    // Implicit conversion from IdxSzT to size_t guarantees that no
    // implicit conversion from size_t to id<1> will happen,
    // thus `acc_src[IdxSzT(0)]` will no longer compile.
    // Replaced with acc_src[0].
    assert(acc_src[0] + acc_src[IdxID1(1)] == 10);

    acc_dst[0] = acc_src[0] + acc_src[IdxID1(0)];
    acc_dst[id1] = acc_src[1] + acc_src[1];
    assert(dst[0] == 6 && dst[1] == 14);
  }

  // Three-dimensional host accessor.
  {
    int data[24];
    for (int i = 0; i < 24; ++i)
      data[i] = i;
    {
      sycl::buffer<int, 3> buf(data, sycl::range<3>(2, 3, 4));

      auto acc = buf.get_access<sycl::access::mode::read_write>();

      assert(!acc.is_placeholder());
      assert(acc.get_size() == sizeof(data));
      assert(acc.get_count() == 24);
      assert(acc.get_range() == sycl::range<3>(2, 3, 4));

      for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
          for (int k = 0; k < 4; ++k)
            acc[IdxID3(i, j, k)] += acc[sycl::id<3>(i, j, k)];
    }
    for (int i = 0; i < 24; ++i) {
      assert(data[i] == 2 * i);
    }
  }
  int data = 5;
  // Device accessor.
  {
    sycl::queue Queue;

    sycl::buffer<int, 1> buf(&data, sycl::range<1>(1),
                             {sycl::property::buffer::use_host_ptr()});

    Queue.submit([&](sycl::handler &cgh) {
      auto acc = buf.get_access<sycl::access::mode::read_write>(cgh);
      assert(!acc.is_placeholder());
      assert(acc.get_size() == sizeof(int));
      assert(acc.get_count() == 1);
      assert(acc.get_range() == sycl::range<1>(1));
      cgh.single_task<class kernel>([=]() { acc[0] += acc[IdxID1(0)]; });
    });
    Queue.wait();
  }
  assert(data == 10);

  // Device accessor with 2-dimensional subscript operators.
  {
    sycl::queue Queue;
    if (!Queue.is_host()) {
      int array[2][3] = {0};
      {
        sycl::range<2> Range(2, 3);
        sycl::buffer<int, 2> buf((int *)array, Range,
                                 {sycl::property::buffer::use_host_ptr()});

        Queue.submit([&](sycl::handler &cgh) {
          auto acc = buf.get_access<sycl::access::mode::read_write>(cgh);
          cgh.parallel_for<class dim2_subscr>(Range, [=](sycl::item<2> itemID) {
            acc[itemID.get_id(0)][itemID.get_id(1)] += itemID.get_linear_id();
          });
        });
        Queue.wait();
      }
      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
          if (array[i][j] != i * 3 + j) {
            std::cerr << array[i][j] << " != " << (i * 3 + j) << std::endl;
            assert(0);
            return 1;
          }
        }
      }
    }
  }

  // Device accessor with 2-dimensional subscript operators for atomic accessor
  // check compile error
  {
    sycl::queue queue;
    if (!queue.is_host()) {
      sycl::range<2> range(1, 1);
      int Arr[] = {2};
      {
        sycl::buffer<int, 1> Buf(Arr, 1);
        queue.submit([&](sycl::handler &cgh) {
          auto acc = sycl::accessor<int, 2, sycl::access::mode::atomic,
                                    sycl::target::local>(range, cgh);
          cgh.parallel_for<class dim2_subscr_atomic>(
              sycl::nd_range<2>{range, range}, [=](sycl::nd_item<2>) {
                sycl::atomic<int, sycl::access::address_space::local_space>
                    value = acc[0][0];
              });
        });
      }
    }
  }

  // Device accessor with 3-dimensional subscript operators.
  {
    sycl::queue Queue;
    if (!Queue.is_host()) {
      int array[2][3][4] = {0};
      {
        sycl::range<3> Range(2, 3, 4);
        sycl::buffer<int, 3> buf((int *)array, Range,
                                 {sycl::property::buffer::use_host_ptr()});

        Queue.submit([&](sycl::handler &cgh) {
          auto acc = buf.get_access<sycl::access::mode::read_write>(cgh);
          cgh.parallel_for<class dim3_subscr>(Range, [=](sycl::item<3> itemID) {
            acc[itemID.get_id(0)][itemID.get_id(1)][itemID.get_id(2)] +=
                itemID.get_linear_id();
          });
        });
        Queue.wait();
      }
      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
          for (int k = 0; k < 4; k++) {
            int expected = k + 4 * (j + 3 * i);
            if (array[i][j][k] != expected) {
              std::cerr << array[i][j][k] << " != " << expected << std::endl;
              assert(0);
              return 1;
            }
          }
        }
      }
    }
  }

  // Local accessor
  {
    sycl::queue queue;

    constexpr int dims = 1;

    using data_loc = int;
    constexpr auto mode_loc = sycl::access::mode::read_write;
    constexpr auto target_loc = sycl::target::local;
    const auto range_loc = sycl::range<1>(1);

    {
      queue.submit([&](sycl::handler &cgh) {
        auto properties = sycl::property_list{};

        auto acc_loc_p = sycl::accessor<data_loc, dims, mode_loc, target_loc>(
            range_loc, cgh, properties);
        auto acc_loc = sycl::accessor<data_loc, dims, mode_loc, target_loc>(
            range_loc, cgh);

        cgh.single_task<class loc_img_acc>([=]() {});
      });
    }
  }

  // Discard write accessor.
  {
    try {
      sycl::queue Queue;
      sycl::buffer<int, 1> buf(sycl::range<1>(3));

      Queue.submit([&](sycl::handler &cgh) {
        auto dev_acc = buf.get_access<sycl::access::mode::discard_write>(cgh);

        cgh.parallel_for<class test_discard_write>(
            sycl::range<1>{3}, [=](sycl::id<1> index) { dev_acc[index] = 42; });
      });

      auto host_acc = buf.get_access<sycl::access::mode::read>();
      for (int i = 0; i != 3; ++i)
        assert(host_acc[i] == 42);

    } catch (sycl::exception e) {
      std::cout << "SYCL exception caught: " << e.what();
      return 1;
    }
  }

  // Discard read-write accessor.
  {
    try {
      sycl::queue Queue;
      sycl::buffer<int, 1> buf(sycl::range<1>(3));

      Queue.submit([&](sycl::handler &cgh) {
        auto dev_acc = buf.get_access<sycl::access::mode::write>(cgh);

        cgh.parallel_for<class test_discard_read_write>(
            sycl::range<1>{3}, [=](sycl::id<1> index) { dev_acc[index] = 42; });
      });

      auto host_acc = buf.get_access<sycl::access::mode::discard_read_write>();
    } catch (sycl::exception e) {
      std::cout << "SYCL exception caught: " << e.what();
      return 1;
    }
  }

  // Check that accessor is initialized when accessor is wrapped to some class.
  {
    sycl::queue queue;
    if (!queue.is_host()) {
      int array[10] = {0};
      {
        sycl::buffer<int, 1> buf((int *)array, sycl::range<1>(10),
                                 {sycl::property::buffer::use_host_ptr()});
        queue.submit([&](sycl::handler &cgh) {
          auto acc = buf.get_access<sycl::access::mode::read_write>(cgh);
          auto acc_wrapped = AccWrapper<decltype(acc)>{acc};
          cgh.parallel_for<class wrapped_access1>(
              sycl::range<1>(buf.get_count()), [=](sycl::item<1> it) {
                auto idx = it.get_linear_id();
                acc_wrapped.accessor[idx] = 333;
              });
        });
        queue.wait();
      }
      for (int i = 0; i < 10; i++) {
        if (array[i] != 333) {
          std::cerr << array[i] << " != 333" << std::endl;
          assert(0);
          return 1;
        }
      }
    }
  }

  // Case when several accessors are wrapped to some class. Check that they are
  // initialized in proper way and value is assigned.
  {
    sycl::queue queue;
    if (!queue.is_host()) {
      int array1[10] = {0};
      int array2[10] = {0};
      {
        sycl::buffer<int, 1> buf1((int *)array1, sycl::range<1>(10),
                                  {sycl::property::buffer::use_host_ptr()});
        sycl::buffer<int, 1> buf2((int *)array2, sycl::range<1>(10),
                                  {sycl::property::buffer::use_host_ptr()});
        queue.submit([&](sycl::handler &cgh) {
          auto acc1 = buf1.get_access<sycl::access::mode::read_write>(cgh);
          auto acc2 = buf2.get_access<sycl::access::mode::read_write>(cgh);
          auto acc_wrapped =
              AccsWrapper<decltype(acc1), decltype(acc2)>{10, acc1, 5, acc2};
          cgh.parallel_for<class wrapped_access2>(
              sycl::range<1>(10), [=](sycl::item<1> it) {
                auto idx = it.get_linear_id();
                acc_wrapped.accessor1[idx] = 333;
                acc_wrapped.accessor2[idx] = 777;
              });
        });
        queue.wait();
      }
      for (int i = 0; i < 10; i++) {
        for (int i = 0; i < 10; i++) {
          if (array1[i] != 333) {
            std::cerr << array1[i] << " != 333" << std::endl;
            assert(0);
            return 1;
          }
          if (array2[i] != 777) {
            std::cerr << array2[i] << " != 777" << std::endl;
            assert(0);
            return 1;
          }
        }
      }
    }
  }

  // Several levels of wrappers for accessor.
  {
    sycl::queue queue;
    if (!queue.is_host()) {
      int array[10] = {0};
      {
        sycl::buffer<int, 1> buf((int *)array, sycl::range<1>(10),
                                 {sycl::property::buffer::use_host_ptr()});
        queue.submit([&](sycl::handler &cgh) {
          auto acc = buf.get_access<sycl::access::mode::read_write>(cgh);
          auto acc_wrapped = AccWrapper<decltype(acc)>{acc};
          Wrapper1 wr1;
          auto wr2 = Wrapper2<decltype(acc)>{wr1, acc_wrapped};
          auto wr3 = Wrapper3<decltype(acc)>{wr2};
          cgh.parallel_for<class wrapped_access3>(
              sycl::range<1>(buf.get_count()), [=](sycl::item<1> it) {
                auto idx = it.get_linear_id();
                wr3.w2.wrapped.accessor[idx] = 333;
              });
        });
        queue.wait();
      }
      for (int i = 0; i < 10; i++) {
        if (array[i] != 333) {
          std::cerr << array[i] << " != 333" << std::endl;
          assert(0);
          return 1;
        }
      }
    }
  }

  // Two accessors to the same buffer.
  {
    try {
      sycl::queue queue;
      int array[3] = {1, 1, 1};
      sycl::buffer<int, 1> buf(array, sycl::range<1>(3));

      queue.submit([&](sycl::handler &cgh) {
        auto acc1 = buf.get_access<sycl::access::mode::read>(cgh);
        auto acc2 = buf.get_access<sycl::access::mode::read_write>(cgh);

        cgh.parallel_for<class two_accessors_to_buf>(
            sycl::range<1>{3},
            [=](sycl::id<1> index) { acc2[index] = 41 + acc1[index]; });
      });

      auto host_acc = buf.get_access<sycl::access::mode::read>();
      for (int i = 0; i != 3; ++i)
        assert(host_acc[i] == 42);

    } catch (sycl::exception e) {
      std::cout << "SYCL exception caught: " << e.what();
      return 1;
    }
  }

  // Accessor with dimensionality 0.
  {
    try {
      int data = -1;
      {
        sycl::buffer<int, 1> b(&data, sycl::range<1>(1));
        sycl::queue queue;
        queue.submit([&](sycl::handler &cgh) {
          sycl::accessor<int, 0, sycl::access::mode::read_write,
                         sycl::target::device>
              B(b, cgh);
          cgh.single_task<class acc_with_zero_dim>([=]() {
            auto B2 = B;
            (int &)B2 = 399;
          });
        });
      }
      assert(data == 399);
    } catch (sycl::exception e) {
      std::cout << "SYCL exception caught: " << e.what();
      return 1;
    }
  }

  {
    // Call every available accessor's constructor to ensure that they work with
    // a buffer with a non-default allocator.
    int data[] = {1, 2, 3};

    using allocator_type = std::allocator<int>;

    sycl::buffer<int, 1, allocator_type> buf1(&data[0], sycl::range<1>(1),
                                              allocator_type{});
    sycl::buffer<int, 1, allocator_type> buf2(&data[1], sycl::range<1>(1),
                                              allocator_type{});
    sycl::buffer<int, 1, allocator_type> buf3(&data[2], sycl::range<1>(1),
                                              allocator_type{});

    sycl::queue queue;
    queue.submit([&](sycl::handler &cgh) {
      sycl::accessor<int, 0, sycl::access::mode::read_write,
                     sycl::target::device>
          acc1(buf1, cgh);
      sycl::accessor<int, 1, sycl::access::mode::read_write,
                     sycl::target::device>
          acc2(buf2, cgh);
      sycl::accessor<int, 1, sycl::access::mode::read_write,
                     sycl::target::device>
          acc3(buf3, cgh, sycl::range<1>(1));
      cgh.single_task<class acc_alloc_buf>([=]() {
        acc1 *= 2;
        acc2[0] *= 2;
        acc3[0] *= 2;
      });
    });

    sycl::accessor<int, 0, sycl::access::mode::read, sycl::target::host_buffer>
        acc4(buf1);
    sycl::accessor<int, 1, sycl::access::mode::read, sycl::target::host_buffer>
        acc5(buf2);
    sycl::accessor<int, 1, sycl::access::mode::read, sycl::target::host_buffer>
        acc6(buf3, sycl::range<1>(1));

    assert(acc4 == 2);
    assert(acc5[0] == 4);
    assert(acc6[0] == 6);
  }

  // Constant buffer accessor
  {
    try {
      int data = -1;
      int cnst = 399;

      {
        sycl::buffer<int, 1> d(&data, sycl::range<1>(1));
        sycl::buffer<int, 1> c(&cnst, sycl::range<1>(1));

        sycl::queue queue;
        queue.submit([&](sycl::handler &cgh) {
          sycl::accessor<int, 1, sycl::access::mode::write,
                         sycl::target::device>
              D(d, cgh);
          sycl::accessor<int, 1, sycl::access::mode::read,
                         sycl::target::constant_buffer>
              C(c, cgh);

          cgh.single_task<class acc_with_const>([=]() { D[0] = C[0]; });
        });

        auto host_acc = d.get_access<sycl::access::mode::read>();
        assert(host_acc[0] == 399);
      }

    } catch (sycl::exception e) {
      std::cout << "SYCL exception caught: " << e.what();
      return 1;
    }
  }

  // Placeholder accessor
  {
    try {
      int data = -1;
      int cnst = 399;

      {
        sycl::buffer<int, 1> d(&data, sycl::range<1>(1));
        sycl::buffer<int, 1> c(&cnst, sycl::range<1>(1));

        sycl::accessor<int, 1, sycl::access::mode::write, sycl::target::device,
                       sycl::access::placeholder::true_t>
            D(d);
        sycl::accessor<int, 1, sycl::access::mode::read,
                       sycl::target::constant_buffer,
                       sycl::access::placeholder::true_t>
            C(c);

        sycl::queue queue;
        queue.submit([&](sycl::handler &cgh) {
          cgh.require(D);
          cgh.require(C);

          cgh.single_task<class placeholder_acc>([=]() { D[0] = C[0]; });
        });

        auto host_acc = d.get_access<sycl::access::mode::read>();
        assert(host_acc[0] == 399);
      }

    } catch (sycl::exception e) {
      std::cout << "SYCL exception caught: " << e.what();
      return 1;
    }
  }

  // placeholder accessor exception  // SYCL2020 4.7.6.9
  {
    sycl::queue q;
    // host device executes kernels via a different method and there
    // is no good way to throw an exception at this time.
    if (!q.is_host()) {
      sycl::range<1> r(4);
      sycl::buffer<int, 1> b(r);
      try {
        sycl::accessor<int, 1, sycl::access::mode::read_write,
                       sycl::access::target::device,
                       sycl::access::placeholder::true_t>
            acc(b);

        q.submit([&](sycl::handler &cgh) {
          // we do NOT call .require(acc) without which we should throw a
          // synchronous exception with errc::kernel_argument
          cgh.parallel_for<class ph>(
              r, [=](sycl::id<1> index) { acc[index] = 0; });
        });
        q.wait_and_throw();
        assert(false && "we should not be here, missing exception");
      } catch (sycl::exception &e) {
        std::cout << "exception received: " << e.what() << std::endl;
        assert(e.code() == sycl::errc::kernel_argument &&
               "incorrect error code");
      } catch (...) {
        std::cout << "some other exception" << std::endl;
        return 1;
      }
    }
  }

  {
    try {
      int data = -1;
      int cnst = 399;

      {
        sycl::buffer<int, 1> A(&cnst, sycl::range<1>(1));
        sycl::buffer<int, 1> B(&cnst, sycl::range<1>(1));
        sycl::buffer<int, 1> C(&data, sycl::range<1>(1));

        sycl::queue queue;
        queue.submit([&](sycl::handler &cgh) {
          sycl::accessor<int, 1, sycl::access::mode::write,
                         sycl::target::device>
              AccA(A, cgh);
          sycl::accessor<int, 1, sycl::access::mode::read,
                         sycl::target::constant_buffer>
              AccB(B, cgh);
          InheritedAccessor<int> AccC(C, cgh);
          cgh.single_task<class acc_base>(
              [=]() { AccC[0] = AccA[0] + AccB[0]; });
        });

#ifndef simplification_test
        auto host_acc = C.get_access<sycl::access::mode::read>();
#else
        sycl::host_accessor host_acc(C, sycl::read_only);
#endif
        assert(host_acc[0] == 798);
      }

    } catch (sycl::exception e) {
      std::cout << "SYCL exception caught: " << e.what();
      return 1;
    }
  }

  // Accessor with buffer size 0.
  {
    try {
      int data[10] = {0};
      {
        sycl::buffer<int, 1> b{&data[0], 10};
        sycl::buffer<int, 1> b1{0};

        sycl::queue queue;
        queue.submit([&](sycl::handler &cgh) {
          sycl::accessor<int, 1, sycl::access::mode::read_write,
                         sycl::target::device>
              B(b, cgh);
          auto B1 = b1.template get_access<sycl::access::mode::read_write>(cgh);

          cgh.single_task<class acc_with_zero_sized_buffer>(
              [=]() { B[0] = 1; });
        });
      }
      assert(!"invalid device accessor buffer size exception wasn't caught");
    } catch (const sycl::invalid_object_error &e) {
      assert(e.get_cl_code() == CL_INVALID_VALUE);
    } catch (sycl::exception e) {
      std::cout << "SYCL exception caught: " << e.what();
      return 1;
    }
  }

  // Accessor to fixed size array type.
  {
    try {
      using array_t = int[3];

      array_t *array = (array_t *)malloc(3 * sizeof(array_t));
      sycl::queue q;
      static_assert(std::is_trivially_copyable<array_t>::value);
      {
        sycl::buffer buf(array, sycl::range<1>(3));
        q.submit([&](sycl::handler &h) {
          auto acc = buf.get_access<sycl::access::mode::write>(h);
          h.parallel_for<class A>(3, [=](sycl::id<1> i) {
            for (int j = 0; j < 3; ++j) {
              acc[i][j] = j + i * 10;
            }
          });
        });
      }
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          int expected = j + i * 10;
          if (array[i][j] != expected) {
            std::cerr << "Accessor to array fail: expected = " << expected
                      << ", computed = " << array[i][j] << std::endl;
            assert(0);
            return 1;
          }
        }
      }
      free(array);
    } catch (sycl::exception e) {
      std::cout << "SYCL exception caught: " << e.what();
      return 1;
    }
  }

  // exceptions with illegal ranges or no_init
  {
    const size_t bufSize = 10;
    std::vector<int> res(bufSize);
    sycl::range<1> r(bufSize);
    sycl::buffer<int, 1> b(res.data(), r);
    sycl::range<1> illegalR(bufSize + 1);
    sycl::id<1> offset(bufSize);

    // illegal ranges
    try {
      auto acc = b.get_access<sycl::access::mode::read_write>(illegalR, offset);
      assert(false && "operation should not have succeeded");
    } catch (sycl::exception &e) {
      assert(e.code() == sycl::errc::invalid && "errc should be errc::invalid");
    }
    try {
      sycl::queue q;
      q.submit([&](sycl::handler &cgh) {
        auto acc = b.get_access<sycl::access::mode::read_write>(cgh, illegalR);
      });
      q.wait_and_throw();
      assert(false &&
             "we should not be here. operation should not have succeeded");
    } catch (sycl::exception &e) {
      assert(e.code() == sycl::errc::invalid && "errc should be errc::invalid");
    }

    // no_init incompatible with read_only
    try {
      sycl::host_accessor out{b, sycl::read_only, sycl::no_init};
      assert(false && "operation should have failed");
    } catch (sycl::exception &e) {
      assert(e.code() == sycl::errc::invalid && "errc should be errc::invalid");
    }
    try {
      sycl::queue q;
      q.submit([&](sycl::handler &cgh) {
        sycl::accessor out{b, cgh, sycl::read_only, sycl::no_init};
      });
      q.wait_and_throw();
      assert(false && "we should not be here. operation should have failed");
    } catch (sycl::exception &e) {
      assert(e.code() == sycl::errc::invalid && "errc should be errc::invalid");
    }
  }

  std::cout << "Test passed" << std::endl;
}
