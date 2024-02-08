// RUN: %{build} -DSYCL2020_DISABLE_DEPRECATION_WARNINGS -o %t.out
// RUN: %{run} %t.out

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

template <typename Acc> struct AccWrapper {
  Acc accessor;
};

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

template <typename Acc> struct Wrapper3 {
  Wrapper2<Acc> w2;
};

using ResAccT = sycl::accessor<int, 1, sycl::access::mode::read_write>;
using AccT = sycl::accessor<int, 1, sycl::access::mode::read>;
using AccCT = sycl::accessor<const int, 1, sycl::access::mode::read>;

void implicit_conversion(const AccCT &acc, const ResAccT &res_acc) {
  auto v = acc[0];
  res_acc[0] = v;
}

void implicit_conversion(const AccT &acc, const ResAccT &res_acc) {
  auto v = acc[0];
  res_acc[0] = v;
}

void implicit_conversion(const sycl::local_accessor<const int, 1> &acc,
                         const ResAccT &res_acc) {
  auto v = acc[0];
  res_acc[0] = v;
}

int implicit_conversion(
    const sycl::host_accessor<const int, 1, sycl::access_mode::read> &acc) {
  auto v = acc[0];
  return v;
}

template <typename T> void TestAccSizeFuncs(const std::vector<T> &vec) {
  auto test = [=](auto &Res, const auto &Acc) {
    Res[0] = Acc.byte_size();
    Res[1] = Acc.size();
    Res[2] = Acc.max_size();
    Res[3] = size_t(Acc.empty());
  };
  auto checkResult = [=](const std::vector<size_t> &Res, size_t MaxSize) {
    assert(Res[0] == vec.size() * sizeof(T));
    assert(Res[1] == vec.size());
    assert(Res[2] == MaxSize);
    assert(Res[3] == vec.empty());
  };
  std::vector<size_t> res(4); // for 4 functions results

  sycl::buffer<T> bufInput(vec.data(), vec.size());
  sycl::host_accessor accInput(bufInput);
  test(res, accInput);
  checkResult(res,
              std::numeric_limits<
                  typename sycl::host_accessor<T>::difference_type>::max());

  sycl::queue q;
  {
    sycl::buffer<T> bufInput(vec.data(), vec.size());
    sycl::buffer<size_t> bufRes(res.data(), res.size());

    q.submit([&](sycl::handler &cgh) {
      sycl::accessor accInput(bufInput, cgh);
      sycl::accessor accRes(bufRes, cgh);
      cgh.single_task([=]() { test(accRes, accInput); });
    });
    q.wait();
  }
  checkResult(
      res,
      std::numeric_limits<typename sycl::accessor<T>::difference_type>::max());

  {
    sycl::buffer<size_t> bufRes(res.data(), res.size());
    q.submit([&](sycl::handler &cgh) {
      sycl::accessor accRes(bufRes, cgh);
      sycl::local_accessor<T, 1> locAcc(vec.size(), cgh);
      cgh.parallel_for(sycl::nd_range<1>{1, 1},
                       [=](sycl::nd_item<1>) { test(accRes, locAcc); });
    });
    q.wait();
  }
  checkResult(res,
              std::numeric_limits<
                  typename sycl::local_accessor<T>::difference_type>::max());
}

template <typename GlobAcc, typename LocAcc>
void testLocalAccItersImpl(sycl::handler &cgh, GlobAcc &globAcc, LocAcc &locAcc,
                           bool testConstIter) {
  if (testConstIter) {
    cgh.parallel_for(sycl::nd_range<1>{1, 1}, [=](sycl::nd_item<1>) {
      size_t Idx = 0;
      for (auto &It : locAcc) {
        It = globAcc[Idx++];
      }
      Idx = 0;
      for (auto It = locAcc.cbegin(); It != locAcc.cend(); It++)
        globAcc[Idx++] = *It * 2 + 1;
      Idx = locAcc.size() - 1;
      for (auto It = locAcc.crbegin(); It != locAcc.crend(); It++)
        globAcc[Idx--] += *It;
    });
  } else {
    cgh.parallel_for(sycl::nd_range<1>{1, 1}, [=](sycl::nd_item<1>) {
      size_t Idx = 0;
      for (auto It = locAcc.begin(); It != locAcc.end(); It++)
        *It = globAcc[Idx++] * 2;
      for (auto &It : locAcc) {
        It++;
      }
      for (auto It = locAcc.rbegin(); It != locAcc.rend(); It++) {
        *It *= 2;
        *It += 1;
      }
      Idx = 0;
      for (auto &It : locAcc) {
        globAcc[Idx++] = It;
      }
    });
  }
}

void testLocalAccIters(std::vector<int> &vec, bool testConstIter = false,
                       bool test2D = false) {
  try {
    sycl::queue queue;
    sycl::buffer<int, 1> buf(vec.data(), vec.size());
    queue.submit([&](sycl::handler &cgh) {
      auto globAcc = buf.get_access<sycl::access::mode::read_write>(cgh);
      if (test2D) {
        sycl::local_accessor<int, 2> locAcc(sycl::range<2>{2, 16}, cgh);
        testLocalAccItersImpl(cgh, globAcc, locAcc, testConstIter);
      } else {
        sycl::local_accessor<int, 1> locAcc(32, cgh);
        testLocalAccItersImpl(cgh, globAcc, locAcc, testConstIter);
      }
    });
  } catch (sycl::exception &e) {
    std::cout << e.what() << std::endl;
  }
}

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
    sycl::host_accessor acc_src(buf_src, sycl::read_only);
    sycl::host_accessor acc_dst(buf_dst);

    assert(!acc_src.is_placeholder());
    assert(acc_src.byte_size() == sizeof(src));
    assert(acc_src.size() == 2);
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

      sycl::host_accessor acc(buf);

      assert(!acc.is_placeholder());
      assert(acc.byte_size() == sizeof(data));
      assert(acc.size() == 24);
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
      assert(acc.byte_size() == sizeof(int));
      assert(acc.size() == 1);
      assert(acc.get_range() == sycl::range<1>(1));
      cgh.single_task<class kernel>([=]() { acc[0] += acc[IdxID1(0)]; });
    });
    Queue.wait();
  }
  assert(data == 10);

  // Device accessor with 2-dimensional subscript operators.
  {
    sycl::queue Queue;
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

  // Device accessor with 2-dimensional subscript operators for atomic accessor
  // check compile error
  {
    sycl::queue queue;
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

  // Device accessor with 3-dimensional subscript operators.
  {
    sycl::queue Queue;
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

      sycl::host_accessor host_acc(buf, sycl::read_only);
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

      sycl::host_accessor host_acc(buf, sycl::no_init);
    } catch (sycl::exception e) {
      std::cout << "SYCL exception caught: " << e.what();
      return 1;
    }
  }

  // Check that accessor is initialized when accessor is wrapped to some class.
  {
    sycl::queue queue;
    int array[10] = {0};
    {
      sycl::buffer<int, 1> buf((int *)array, sycl::range<1>(10),
                               {sycl::property::buffer::use_host_ptr()});
      queue.submit([&](sycl::handler &cgh) {
        auto acc = buf.get_access<sycl::access::mode::read_write>(cgh);
        auto acc_wrapped = AccWrapper<decltype(acc)>{acc};
        cgh.parallel_for<class wrapped_access1>(
            sycl::range<1>(buf.size()), [=](sycl::item<1> it) {
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

  // Case when several accessors are wrapped to some class. Check that they are
  // initialized in proper way and value is assigned.
  {
    sycl::queue queue;
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

  // Several levels of wrappers for accessor.
  {
    sycl::queue queue;
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
            sycl::range<1>(buf.size()), [=](sycl::item<1> it) {
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

      sycl::host_accessor host_acc(buf, sycl::read_only);
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

        sycl::host_accessor host_acc(d, sycl::read_only);
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

        sycl::host_accessor host_acc(d, sycl::read_only);
        assert(host_acc[0] == 399);
      }

    } catch (sycl::exception e) {
      std::cout << "SYCL exception caught: " << e.what();
      return 1;
    }
  }

  // placeholder accessor exception (1)  // SYCL2020 4.7.6.9
  {
    sycl::queue q;
    // host device executes kernels via a different method and there
    // is no good way to throw an exception at this time.
    sycl::range<1> r(4);
    sycl::buffer<int, 1> b(r);
    try {
      sycl::accessor<int, 1, sycl::access::mode::read_write,
                     sycl::access::target::device,
                     sycl::access::placeholder::true_t>
          acc(b);

      q.submit([&](sycl::handler &cgh) {
        // We do NOT call .require(acc) without which we should throw a
        // synchronous exception with errc::kernel_argument
        cgh.parallel_for<class ph1>(r,
                                    [=](sycl::id<1> index) { acc[index] = 0; });
      });
      q.wait_and_throw();
      assert(false && "we should not be here, missing exception");
    } catch (sycl::exception &e) {
      std::cout << "exception received: " << e.what() << std::endl;
      assert(e.code() == sycl::errc::kernel_argument && "incorrect error code");
    } catch (...) {
      std::cout << "Some other exception (line " << __LINE__ << ")"
                << std::endl;
      return 1;
    }
  }

  // placeholder accessor exception (2) // SYCL2020 4.7.6.9
  {
    sycl::queue q;
    // host device executes kernels via a different method and there
    // is no good way to throw an exception at this time.
    sycl::range<1> r(4);
    sycl::buffer<int, 1> b(r);
    try {
      using AccT = sycl::accessor<int, 1, sycl::access::mode::read_write,
                                  sycl::access::target::device,
                                  sycl::access::placeholder::true_t>;
      AccT acc(b);

      q.submit([&](sycl::handler &cgh) {
        // We do NOT call .require(acc) without which we should throw a
        // synchronous exception with errc::kernel_argument
        // The difference with the previous test is that the use of acc
        // is usually optimized away for this particular scenario, but the
        // exception should be thrown because of passing it, not because of
        // using it
        cgh.single_task<class ph2>([=] { int x = acc[0]; });
      });
      q.wait_and_throw();
      assert(false && "we should not be here, missing exception");
    } catch (sycl::exception &e) {
      std::cout << "exception received: " << e.what() << std::endl;
      assert(e.code() == sycl::errc::kernel_argument && "incorrect error code");
    } catch (...) {
      std::cout << "Some other exception (line " << __LINE__ << ")"
                << std::endl;
      return 1;
    }
  }

  // placeholder accessor exception (3)  // SYCL2020 4.7.6.9
  {
    sycl::queue q;
    // host device executes kernels via a different method and there
    // is no good way to throw an exception at this time.
    sycl::range<1> r(4);
    sycl::buffer<int, 1> b(r);
    try {
      using AccT = sycl::accessor<int, 1, sycl::access::mode::read_write,
                                  sycl::access::target::device,
                                  sycl::access::placeholder::true_t>;
      AccT acc(b);

      q.submit([&](sycl::handler &cgh) {
        AccT acc2(b, cgh);
        // We do NOT call .require(acc) without which we should throw a
        // synchronous exception with errc::kernel_argument
        // The particularity of this test is that it passes to a command
        // one bound accessor and one unbound accessor. In the past, this
        // has led to throw the wrong exception.
        cgh.single_task<class ph3>([=] {
          int x = acc[0];
          int y = acc2[0];
        });
      });
      q.wait_and_throw();
      assert(false && "we should not be here, missing exception");
    } catch (sycl::exception &e) {
      std::cout << "exception received: " << e.what() << std::endl;
      assert(e.code() == sycl::errc::kernel_argument && "incorrect error code");
    } catch (...) {
      std::cout << "Some other exception (line " << __LINE__ << ")"
                << std::endl;
      return 1;
    }
  }

  // placeholder accessor exception (4)  // SYCL2020 4.7.6.9
  {
    sycl::queue q;
    // host device executes kernels via a different method and there
    // is no good way to throw an exception at this time.
    sycl::range<1> r(4);
    sycl::buffer<int, 1> b(r);
    try {
      using AccT = sycl::accessor<int, 1, sycl::access::mode::read_write,
                                  sycl::access::target::device,
                                  sycl::access::placeholder::true_t>;
      AccT acc(b);

      q.submit([&](sycl::handler &cgh) {
        AccT acc2(b, cgh);
        // Pass placeholder accessor to command, but having required a different
        // accessor in the command. In past versions, we used to compare the
        // number of accessors with the number of requirements, and if they
        // matched, we did not throw, allowing this scenario that shouldn't be
        // allowed.
        cgh.single_task<class ph4>([=] { int x = acc[0]; });
      });
      q.wait_and_throw();
      assert(false && "we should not be here, missing exception");
    } catch (sycl::exception &e) {
      std::cout << "exception received: " << e.what() << std::endl;
      assert(e.code() == sycl::errc::kernel_argument && "incorrect error code");
    } catch (...) {
      std::cout << "Some other exception (line " << __LINE__ << ")"
                << std::endl;
      return 1;
    }
  }

  // SYCL2020 4.9.4.1: calling require() on empty accessor should not throw.
  {
    sycl::queue q;
    try {
      AccT acc;

      q.submit([&](sycl::handler &cgh) { cgh.require(acc); });
      q.wait_and_throw();
    } catch (sycl::exception &e) {
      assert("Unexpected exception");
    } catch (...) {
      std::cout << "Some other unexpected exception (line " << __LINE__ << ")"
                << std::endl;
      return 1;
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
        auto host_acc = C.get_host_access(sycl::read_only);
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

  // Accessor common property interface
  {
    using namespace sycl::ext::oneapi;
    int data[1] = {0};

    // host accessor
    try {
      sycl::buffer<int, 1> buf_data(data, sycl::range<1>(1),
                                    {sycl::property::buffer::use_host_ptr()});
      accessor_property_list PL{no_alias, no_offset, sycl::no_init};
      sycl::accessor acc_1(buf_data, PL);
      static_assert(acc_1.has_property<property::no_alias>());
      static_assert(acc_1.has_property<property::no_offset>());
      assert(acc_1.has_property<sycl::property::no_init>());

      static_assert(acc_1.get_property<property::no_alias>() == no_alias);
      static_assert(acc_1.get_property<property::no_offset>() == no_offset);
      // Should not throw "The property is not found"
      auto noInit = acc_1.get_property<sycl::property::no_init>();
    } catch (sycl::exception e) {
      std::cout << "SYCL exception caught: " << e.what() << std::endl;
    }

    // base accessor
    try {
      sycl::buffer<int, 1> buf_data(data, sycl::range<1>(1),
                                    {sycl::property::buffer::use_host_ptr()});
      sycl::queue q;
      accessor_property_list PL{no_alias, no_offset, sycl::no_init};

      q.submit([&](sycl::handler &cgh) {
        sycl::accessor acc_1(buf_data, PL);
        static_assert(acc_1.has_property<property::no_alias>());
        static_assert(acc_1.has_property<property::no_offset>());
        assert(acc_1.has_property<sycl::property::no_init>());

        static_assert(acc_1.get_property<property::no_alias>() == no_alias);
        static_assert(acc_1.get_property<property::no_offset>() == no_offset);
        // Should not throw "The property is not found"
        auto noInit = acc_1.get_property<sycl::property::no_init>();
      });
      q.wait();
    } catch (sycl::exception e) {
      std::cout << "SYCL exception caught: " << e.what() << std::endl;
    }
  }

  // Test byte_size(), size(), max_size(), empty()
  {
    std::vector<char> vecChar(64, 'a');
    std::vector<int> vecInt(32, 1);
    std::vector<float> vecFloat(16, 1.0);
    std::vector<IdxID3> vecCustom(8, {0, 0, 0});
    TestAccSizeFuncs(vecChar);
    TestAccSizeFuncs(vecInt);
    TestAccSizeFuncs(vecFloat);
    TestAccSizeFuncs(vecCustom);
  }
  // Test swap() on host_accessor
  {
    std::vector<int> vec1(8), vec2(16);
    {
      sycl::buffer<int> buf1(vec1.data(), vec1.size());
      sycl::buffer<int> buf2(vec2.data(), vec2.size());
      sycl::host_accessor acc1(buf1);
      sycl::host_accessor acc2(buf2);
      acc1.swap(acc2);
      acc1[15] = 4;
      acc2[7] = 4;
    }
    assert(vec1[7] == 4 && vec2[15] == 4);
  }

  // 0-dim host_accessor iterator
  {
    std::vector<int> vec1(8);
    {
      sycl::buffer<int> buf1(vec1.data(), vec1.size());
      sycl::host_accessor<int, 0> acc1(buf1);
      *acc1.begin() = 4;
      auto value = *acc1.cbegin();
      value += *acc1.crbegin();
      *acc1.rbegin() += value;
    }
    assert(vec1[0] == 12);
  }

  // Test swap() on basic accessor
  {
    std::vector<int> vec1(8), vec2(16);
    {
      sycl::buffer<int> buf1(vec1.data(), vec1.size());
      sycl::buffer<int> buf2(vec2.data(), vec2.size());
      sycl::queue q;
      q.submit([&](sycl::handler &cgh) {
        sycl::accessor acc1(buf1, cgh);
        sycl::accessor acc2(buf2, cgh);
        acc1.swap(acc2);
        cgh.parallel_for<class swap1>(sycl::nd_range<1>{1, 1},
                                      [=](sycl::nd_item<1>) {
                                        acc1[15] = 4;
                                        acc2[7] = 4;
                                      });
      });
    }
    assert(vec1[7] == 4 && vec2[15] == 4);
  }
  // Test swap on local_accessor
  {
    size_t size1 = 0, size2 = 0;
    {
      sycl::buffer<size_t> buf1(&size1, 1);
      sycl::buffer<size_t> buf2(&size2, 1);

      sycl::queue q;
      q.submit([&](sycl::handler &cgh) {
        sycl::accessor acc1(buf1, cgh);
        sycl::accessor acc2(buf2, cgh);
        sycl::local_accessor<int, 1> locAcc1(8, cgh), locAcc2(16, cgh);
        locAcc1.swap(locAcc2);
        cgh.parallel_for<class swap2>(sycl::nd_range<1>{1, 1},
                                      [=](sycl::nd_item<1>) {
                                        acc1[0] = locAcc1.size();
                                        acc2[0] = locAcc2.size();
                                      });
      });
    }
    assert(size1 == 16 && size2 == 8);
  }
  // Test iterator methods with 1D local_accessor
  {
    std::vector<int> v(32);
    for (int i = 0; i < v.size(); ++i) {
      v[i] = i;
    }
    testLocalAccIters(v);
    for (int i = 0; i < v.size(); ++i)
      assert(v[i] == ((i * 2 + 1) * 2 + 1));

    for (int i = 0; i < v.size(); ++i) {
      v[i] = i;
    }
    testLocalAccIters(v, true);
    for (int i = 0; i < v.size(); ++i)
      assert(v[i] == ((i * 2 + 1) + i));
  }
  // Test iterator methods with 2D local_accessor
  {
    std::vector<int> v(32);
    for (int i = 0; i < v.size(); ++i) {
      v[i] = i;
    }
    testLocalAccIters(v, false, true);
    for (int i = 0; i < v.size(); ++i)
      assert(v[i] == ((i * 2 + 1) * 2 + 1));

    for (int i = 0; i < v.size(); ++i) {
      v[i] = i;
    }
    testLocalAccIters(v, true, true);
    for (int i = 0; i < v.size(); ++i)
      assert(v[i] == ((i * 2 + 1) + i));
  }

  // Assignment operator test for 0-dim buffer accessor
  {
    sycl::queue Queue;
    int Data = 32;

    // Explicit block to prompt copy-back to Data
    {
      sycl::buffer<int, 1> DataBuffer(&Data, sycl::range<1>(1));

      Queue.submit([&](sycl::handler &CGH) {
        sycl::accessor<int, 0> Acc(DataBuffer, CGH);
        CGH.single_task<class acc_0_dim_assignment>([=]() { Acc = 64; });
      });
      Queue.wait();
    }

    assert(Data == 64);
  }

  // iterator operations test for 0-dim buffer accessor
  {
    sycl::queue Queue;
    int Data[] = {32, 32};

    // Explicit block to prompt copy-back to Data
    {
      sycl::buffer<int, 1> DataBuffer(Data, sycl::range<1>(2));

      Queue.submit([&](sycl::handler &CGH) {
        sycl::accessor<int, 0> Acc(DataBuffer, CGH);
        CGH.single_task<class acc_0_dim_iter_assignment>([=]() {
          *Acc.begin() = 64;
          auto value = *Acc.cbegin();
          value += *Acc.crbegin();
          *Acc.rbegin() += value;
        });
      });
      Queue.wait();
    }

    assert(Data[0] == 64 * 3);
    assert(Data[1] == 32);
  }

  // iterator operations test for 0-dim buffer accessor with target::host_task
  {
    sycl::queue Queue;
    int Data[] = {32, 32};

    using HostTaskAcc = sycl::accessor<int, 0, sycl::access::mode::read_write,
                                       sycl::access::target::host_task>;

    // Explicit block to prompt copy-back to Data
    {
      sycl::buffer<int, 1> DataBuffer(Data, sycl::range<1>(2));

      Queue.submit([&](sycl::handler &CGH) {
        HostTaskAcc Acc(DataBuffer, CGH);
        CGH.host_task([=]() {
          *Acc.begin() = 64;
          auto value = *Acc.cbegin();
          value += *Acc.crbegin();
          *Acc.rbegin() += value;
        });
      });
      Queue.wait();
    }

    assert(Data[0] == 64 * 3);
    assert(Data[1] == 32);
  }

  // Assignment operator test for 0-dim local accessor
  {
    sycl::queue Queue;
    int Data = 0;

    // Explicit block to prompt copy-back to Data
    {
      sycl::buffer<int, 1> DataBuffer(&Data, sycl::range<1>(1));
      Queue.submit([&](sycl::handler &CGH) {
        sycl::accessor<int, 0> Acc(DataBuffer, CGH);
        sycl::local_accessor<int, 0> LocalAcc(CGH);
        CGH.parallel_for<class copyblock>(sycl::nd_range<1>{1, 1},
                                          [=](sycl::nd_item<1>) {
                                            LocalAcc = 64;
                                            Acc = LocalAcc;
                                          });
      });
    }

    assert(Data == 64);
  }

  // Throws exception on local_accessors used in single_task
  {
    constexpr static int size = 1;
    sycl::queue Queue;

    try {
      Queue.submit([&](sycl::handler &cgh) {
        auto local_acc = sycl::local_accessor<int, 1>({size}, cgh);
        cgh.single_task<class local_acc_exception>([=]() { (void)local_acc; });
      });
      assert(0 && "local accessor must not be used in single task.");
    } catch (sycl::exception e) {
      std::cout << "SYCL exception caught: " << e.what() << std::endl;
    }
  }

  // Throws exception on local_accessors used in parallel_for taking a range
  // parameter.
  {
    constexpr static int size = 1;
    sycl::queue Queue;

    try {
      Queue.submit([&](sycl::handler &cgh) {
        auto local_acc = sycl::local_accessor<int, 1>({size}, cgh);
        cgh.parallel_for<class parallel_for_exception>(
            sycl::range<1>{size}, [=](sycl::id<1> ID) { (void)local_acc; });
      });
      assert(0 &&
             "local accessor must not be used in parallel for with range.");
    } catch (sycl::exception e) {
      std::cout << "SYCL exception caught: " << e.what() << std::endl;
    }
  }

  // local_accessor::operator& and local_accessor::operator[] with const DataT
  {
    using AccT_zero = sycl::local_accessor<const int, 0>;
    using AccT_non_zero = sycl::local_accessor<const int, 1>;

    sycl::queue queue;
    {
      queue.submit([&](sycl::handler &cgh) {
        AccT_zero acc_zero(cgh);
        AccT_non_zero acc_non_zero(sycl::range<1>(5), cgh);
        cgh.parallel_for<class local_acc_const_type>(
            sycl::nd_range<1>{1, 1}, [=](sycl::nd_item<1> ID) {
              const int &ref_zero = acc_zero;
              const int &ref_non_zero = acc_non_zero[0];
            });
      });
    }
  }

  // Assignment operator test for 0-dim local accessor iterator
  {
    sycl::queue Queue;
    int Data = 0;

    // Explicit block to prompt copy-back to Data
    {
      sycl::buffer<int, 1> DataBuffer(&Data, sycl::range<1>(1));

      Queue.submit([&](sycl::handler &CGH) {
        sycl::accessor<int, 0> Acc(DataBuffer, CGH);
        sycl::local_accessor<int, 0> LocalAcc(CGH);
        CGH.parallel_for<class local_acc_0_dim_iter_assignment>(
            sycl::nd_range<1>{1, 1}, [=](sycl::nd_item<1> ID) {
              *LocalAcc.begin() = 32;
              auto value = *LocalAcc.cbegin();
              value += *LocalAcc.crbegin();
              *LocalAcc.rbegin() += value;
              Acc = LocalAcc;
            });
      });
    }

    assert(Data == 96);
  }

  // host_accessor hash
  {
    sycl::buffer<int> buffer1{sycl::range<1>{1}};
    sycl::buffer<int> buffer2{sycl::range<1>{1}};
    sycl::host_accessor<int> host_acc1{buffer1};
    auto host_acc2(host_acc1);
    sycl::host_accessor<int> host_acc3{buffer2};

    auto host_acc1_hash = std::hash<sycl::host_accessor<int>>{}(host_acc1);
    auto host_acc2_hash = std::hash<sycl::host_accessor<int>>{}(host_acc2);
    auto host_acc3_hash = std::hash<sycl::host_accessor<int>>{}(host_acc3);

    assert(host_acc1_hash == host_acc2_hash && "Identical hash expected.");
    assert(host_acc1_hash != host_acc3_hash &&
           "Identical hash was not expected.");
  }

  // local_accessor hash
  {
    sycl::queue queue;

    queue.submit([&](sycl::handler &cgh) {
      sycl::local_accessor<int, 1> local_acc1(1, cgh);
      auto local_acc2 = local_acc1;
      sycl::local_accessor<int, 1> local_acc3(1, cgh);

      auto local_acc1_hash =
          std::hash<sycl::local_accessor<int, 1>>{}(local_acc1);
      auto local_acc2_hash =
          std::hash<sycl::local_accessor<int, 1>>{}(local_acc2);
      auto local_acc3_hash =
          std::hash<sycl::local_accessor<int, 1>>{}(local_acc3);

      assert(local_acc1_hash == local_acc2_hash && "Identical hash expected.");
      assert(local_acc1_hash != local_acc3_hash &&
             "Identical hash was not expected.");
    });
  }

  // accessor<T> to accessor<const T> implicit conversion.
  {
    int data = 123;
    int result = 0;
    {
      sycl::buffer<int, 1> data_buf(&data, 1);
      sycl::buffer<int, 1> res_buf(&result, 1);
      sycl::queue queue;
      queue
          .submit([&](sycl::handler &cgh) {
            ResAccT res_acc = res_buf.get_access(cgh);
            AccT acc(data_buf, cgh);
            cgh.single_task([=]() { implicit_conversion(acc, res_acc); });
          })
          .wait_and_throw();
    }
    assert(result == 123 && "Expected value not seen.");
  }

  {
    const int data = 123;
    int result = 0;

    // accessor<const T, read_only> to accessor<T, read_only> implicit
    // conversion.
    {
      sycl::buffer<const int, 1> data_buf(&data, 1);
      sycl::buffer<int, 1> res_buf(&result, 1);

      sycl::queue queue;
      queue
          .submit([&](sycl::handler &cgh) {
            ResAccT res_acc = res_buf.get_access(cgh);
            AccCT acc(data_buf, cgh);
            cgh.parallel_for_work_group(sycl::range(1), [=](sycl::group<1>) {
              implicit_conversion(acc, res_acc);
            });
          })
          .wait_and_throw();
    }
    assert(result == 123 && "Expected value not seen.");
  }

  // local_accessor<T> to local_accessor<const T> implicit conversion.
  {
    int data = 123;
    int result = 0;
    {
      sycl::buffer<int, 1> res_buf(&result, 1);
      sycl::queue queue;
      queue
          .submit([&](sycl::handler &cgh) {
            ResAccT res_acc = res_buf.get_access(cgh);
            sycl::local_accessor<int, 1> locAcc(1, cgh);
            cgh.parallel_for(sycl::nd_range<1>{1, 1}, [=](sycl::nd_item<1>) {
              locAcc[0] = 123;
              implicit_conversion(locAcc, res_acc);
            });
          })
          .wait_and_throw();
    }
    assert(result == 123 && "Expected value not seen.");
  }

  // host_accessor<T, read_write> to host_accessor<const T, read> implicit
  // conversion.
  {
    int data = -1;
    sycl::buffer<int, 1> d(&data, sycl::range<1>(1));
    sycl::host_accessor host_acc(d, sycl::read_write);
    host_acc[0] = 399;
    assert(implicit_conversion(host_acc) == 399);
  }

  // accessor swap
  {
    int data[2] = {2, 100};
    int data2[2] = {23, 4};
    int results[2] = {0, 0};
    {
      sycl::buffer<int, 1> data_buf(data, 2);
      sycl::buffer<int, 1> data_buf2(data2, 2);
      sycl::buffer<int, 1> res_buf(results, 2);
      sycl::queue queue;
      queue
          .submit([&](sycl::handler &cgh) {
            ResAccT res_acc = res_buf.get_access(cgh);
            AccT acc1(data_buf, cgh);
            AccT acc2(data_buf2, cgh);
            std::swap(acc1, acc2);
            cgh.single_task([=]() {
              res_acc[0] = acc1[0]; // data2[0] == 23
              res_acc[1] = acc2[0]; // data1[0] == 2
              AccT acc1_copy(acc1);
              AccT acc2_copy(acc2);
              std::swap(acc1_copy, acc2_copy);
              res_acc[0] += acc1_copy[1]; // data1[1] == 100
              res_acc[1] += acc2_copy[1]; // data2[0] == 4
            });
          })
          .wait_and_throw();
    }
    assert(results[0] == 123 && "Unexpected value!");
    assert(results[1] == 6 && "Unexpected value!");
  }

  // accessor with buffer size 0.
  {
    sycl::buffer<int, 1> Buf{0};
    sycl::buffer<int, 1> Buf2{200};

    {
      sycl::queue queue;
      for (auto IBuf : {Buf, Buf2}) {
        queue
            .submit([&](sycl::handler &cgh) {
              auto B =
                  IBuf.template get_access<sycl::access::mode::read_write>(cgh);

              cgh.single_task<class fill_with_potentially_zero_size>([=]() {
                for (size_t I = 0; I < B.size(); ++I)
                  B[I] = 1;
              });
            })
            .wait();
      }
    }
  }

  // default constructed accessor is not a placeholder
  {
    AccT acc;
    assert(!acc.is_placeholder());
    sycl::queue q;
    bool result;
    {
      sycl::buffer<bool, 1> Buf{&result, sycl::range<1>{1}};
      // As a non-placeholder accessor, make sure no exception is thrown when
      // passed to a command. However, we cannot access it, because there's
      // no underlying storage.
      try {
        q.submit([&](sycl::handler &cgh) {
          sycl::accessor res_acc{Buf, cgh};
          cgh.single_task<class def_ctor>(
              [=] { res_acc[0] = acc.is_placeholder(); });
        });
        q.wait_and_throw();
      } catch (sycl::exception &e) {
        assert("Unexpected exception");
      } catch (...) {
        std::cout << "Some other unexpected exception (line " << __LINE__ << ")"
                  << std::endl;
        return 1;
      }
    }
    assert(!result);
  }

  // default constructed accessor can be passed to a kernel.
  {
    AccT acc;
    sycl::queue q;
    bool result = false;
    {
      sycl::buffer<bool, 1> Buf{&result, sycl::range<1>{1}};
      // We are passing a default constructed accessor and a non default
      // constructed accessor with storage. Default constructed accessors can be
      // passed to commands, but trying to access the (non-existing) underlying
      // storage is UB. This test should work, since the access to the default
      // constructed accessor must never be reached.
      try {
        q.submit([&](sycl::handler &cgh) {
          sycl::accessor res_acc{Buf, cgh};
          cgh.single_task<class def_ctor_kernel>([=] {
            if (false)
              res_acc[0] = acc[0];
          });
        });
        q.wait_and_throw();
      } catch (sycl::exception &e) {
        assert("Unexpected exception");
      } catch (...) {
        std::cout << "Some other unexpected exception (line " << __LINE__ << ")"
                  << std::endl;
        return 1;
      }
    }
    assert(!result);
  }

  // default constructed accessor can be passed to a kernel (2).
  {
    using AccT = sycl::accessor<int, 1, sycl::access::mode::read_write>;
    AccT acc;
    assert(acc.empty());
    sycl::queue q;
    bool result = false;
    {
      // We are passing only a default constructed accessor. Default constructed
      // accessors can be passed to commands, but trying to access the
      // (non-existing) underlying storage is UB. This test should work, since
      // the access to the default constructed accessor must never be reached.
      // The difference with the previous test case is that in this case the
      // task will not have any requirements, while the previous one does have
      // one requirement for the non default constructed accessor, testing
      // different code paths.
      try {
        q.submit([&](sycl::handler &cgh) {
          cgh.single_task<class def_ctor_kernel2>([=] {
            if (!acc.empty())
              acc[0] = 1;
          });
        });
        q.wait_and_throw();
      } catch (sycl::exception &e) {
        assert("Unexpected exception");
      } catch (...) {
        std::cout << "Some other unexpected exception (line " << __LINE__ << ")"
                  << std::endl;
        return 1;
      }
    }
    assert(!result);
  }

  std::cout << "Test passed" << std::endl;
}
