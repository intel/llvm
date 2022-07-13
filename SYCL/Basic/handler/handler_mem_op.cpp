// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// UNSUPPORTED: cuda

// FIXME: the test fails on multiple back-ends
// REQUIRES: TEMPORARY_DISABLED

//==- handler.cpp - SYCL handler explicit memory operations test -*- C++-*--==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

#include <cassert>
#include <iostream>
#include <numeric>

using namespace cl::sycl;

template <typename T> struct point {
  point(const point &rhs) = default;
  point(T x, T y) : x(x), y(y) {}
  point(T v) : x(v), y(v) {}
  point() : x(0), y(0) {}
  bool operator==(const T &rhs) const { return rhs == x && rhs == y; }
  bool operator==(const point<T> &rhs) const {
    return rhs.x == x && rhs.y == y;
  }
  T x;
  T y;
};

template <typename T> void test_fill(T Val);
template <typename T> void test_copy_ptr_acc();
template <typename T> void test_3D_copy_ptr_acc();
template <typename T> void test_copy_acc_ptr();
template <typename T> void test_3D_copy_acc_ptr();
template <typename T> void test_copy_shared_ptr_acc();
template <typename T> void test_copy_shared_ptr_const_acc();
template <typename T> void test_copy_acc_shared_ptr();
template <typename T> void test_copy_acc_acc();
template <typename T> void test_update_host();
template <typename T> void test_2D_copy_acc_acc();
template <typename T> void test_3D_copy_acc_acc();
template <typename T> void test_0D1D_copy_acc_acc();
template <typename T> void test_1D2D_copy_acc_acc();
template <typename T> void test_1D3D_copy_acc_acc();
template <typename T> void test_2D1D_copy_acc_acc();
template <typename T> void test_2D3D_copy_acc_acc();
template <typename T> void test_3D1D_copy_acc_acc();
template <typename T> void test_3D2D_copy_acc_acc();

int main() {
  // handler.fill
  {
    test_fill<int>(888);
    test_fill<int>(777);
    test_fill<float>(888.0f);
    test_fill<point<int>>(point<int>(111.0f, 222.0f));
    test_fill<point<int>>(point<int>(333.0f));
    test_fill<point<float>>(point<float>(444.0f, 555.0f));
  }

  // handler.copy(ptr, acc)
  {
    test_copy_ptr_acc<int>();
    test_copy_ptr_acc<int>();
    test_copy_ptr_acc<point<int>>();
    test_copy_ptr_acc<point<int>>();
    test_copy_ptr_acc<point<float>>();
  }
  // handler.copy(ptr, acc) 3D
  {
    test_3D_copy_ptr_acc<int>();
    test_3D_copy_ptr_acc<int>();
    test_3D_copy_ptr_acc<point<int>>();
    test_3D_copy_ptr_acc<point<int>>();
    test_3D_copy_ptr_acc<point<float>>();
  }
  // handler.copy(acc, ptr)
  {
    test_copy_acc_ptr<int>();
    test_copy_acc_ptr<int>();
    test_copy_acc_ptr<point<int>>();
    test_copy_acc_ptr<point<int>>();
    test_copy_acc_ptr<point<float>>();
  }
  // handler.copy(acc, ptr) 3D
  {
    test_3D_copy_acc_ptr<int>();
    test_3D_copy_acc_ptr<int>();
    test_3D_copy_acc_ptr<point<int>>();
    test_3D_copy_acc_ptr<point<int>>();
    test_3D_copy_acc_ptr<point<float>>();
  }
  // handler.copy(shared_ptr, acc)
  {
    test_copy_shared_ptr_acc<int>();
    test_copy_shared_ptr_acc<int>();
    test_copy_shared_ptr_acc<point<int>>();
    test_copy_shared_ptr_acc<point<int>>();
    test_copy_shared_ptr_acc<point<float>>();
  }
  // handler.copy(const shared_ptr, acc)
  {
    test_copy_shared_ptr_const_acc<int>();
    test_copy_shared_ptr_const_acc<int>();
    test_copy_shared_ptr_const_acc<point<int>>();
    test_copy_shared_ptr_const_acc<point<int>>();
    test_copy_shared_ptr_const_acc<point<float>>();
  }
  // handler.copy(acc, shared_ptr)
  {
    test_copy_acc_shared_ptr<int>();
    test_copy_acc_shared_ptr<int>();
    test_copy_acc_shared_ptr<point<int>>();
    test_copy_acc_shared_ptr<point<int>>();
    test_copy_acc_shared_ptr<point<float>>();
  }
  // handler.copy(acc, acc)
  {
    test_copy_acc_acc<int>();
    test_copy_acc_acc<int>();
    test_copy_acc_acc<point<int>>();
    test_copy_acc_acc<point<int>>();
    test_copy_acc_acc<point<float>>();
  }

  // handler.update_host(acc)
  {
    test_update_host<int>();
    test_update_host<int>();
    test_update_host<point<int>>();
    test_update_host<point<int>>();
    test_update_host<point<float>>();
  }

  // handler.copy(acc, acc) 2D
  {
    test_2D_copy_acc_acc<int>();
    test_2D_copy_acc_acc<int>();
    test_2D_copy_acc_acc<point<int>>();
    test_2D_copy_acc_acc<point<int>>();
    test_2D_copy_acc_acc<point<float>>();
  }

  // handler.copy(acc, acc) 3D
  {
    test_3D_copy_acc_acc<int>();
    test_3D_copy_acc_acc<int>();
    test_3D_copy_acc_acc<point<int>>();
    test_3D_copy_acc_acc<point<int>>();
    test_3D_copy_acc_acc<point<float>>();
  }

  // handler.copy(acc, acc) 0D to/from 1D
  {
    test_0D1D_copy_acc_acc<int>();
    test_0D1D_copy_acc_acc<point<int>>();
    test_0D1D_copy_acc_acc<point<float>>();
  }

  // handler.copy(acc, acc) 1D to 2D
  {
    test_1D2D_copy_acc_acc<int>();
    test_1D2D_copy_acc_acc<int>();
    test_1D2D_copy_acc_acc<point<int>>();
    test_1D2D_copy_acc_acc<point<int>>();
    test_1D2D_copy_acc_acc<point<float>>();
  }

  // handler.copy(acc, acc) 1D to 3D
  {
    test_1D3D_copy_acc_acc<int>();
    test_1D3D_copy_acc_acc<int>();
    test_1D3D_copy_acc_acc<point<int>>();
    test_1D3D_copy_acc_acc<point<int>>();
    test_1D3D_copy_acc_acc<point<float>>();
  }

  // handler.copy(acc, acc) 2D to 1D
  {
    test_2D1D_copy_acc_acc<int>();
    test_2D1D_copy_acc_acc<int>();
    test_2D1D_copy_acc_acc<point<int>>();
    test_2D1D_copy_acc_acc<point<int>>();
    test_2D1D_copy_acc_acc<point<float>>();
  }

  // handler.copy(acc, acc) 2D to 3D
  {
    test_2D3D_copy_acc_acc<int>();
    test_2D3D_copy_acc_acc<int>();
    test_2D3D_copy_acc_acc<point<int>>();
    test_2D3D_copy_acc_acc<point<int>>();
    test_2D3D_copy_acc_acc<point<float>>();
  }

  // handler.copy(acc, acc) 3D to 1D
  {
    test_3D1D_copy_acc_acc<int>();
    test_3D1D_copy_acc_acc<int>();
    test_3D1D_copy_acc_acc<point<int>>();
    test_3D1D_copy_acc_acc<point<int>>();
    test_3D1D_copy_acc_acc<point<float>>();
  }

  // handler.copy(acc, acc) 3D to 2D
  {
    test_3D2D_copy_acc_acc<int>();
    test_3D2D_copy_acc_acc<int>();
    test_3D2D_copy_acc_acc<point<int>>();
    test_3D2D_copy_acc_acc<point<int>>();
    test_3D2D_copy_acc_acc<point<float>>();
  }
  std::cout << "finish" << std::endl;
  return 0;
}

template <typename T> void test_fill(T Val) {
  const size_t Size = 10;
  T Data[Size] = {0};
  const T Value = Val;
  {
    buffer<T, 1> Buffer(Data, range<1>(Size));
    queue Queue;
    Queue.submit([&](handler &Cgh) {
      accessor<T, 1, access::mode::write, access::target::device> Accessor(
          Buffer, Cgh, range<1>(Size));
      Cgh.fill(Accessor, Value);
    });
  }
  for (size_t I = 0; I < Size; ++I) {
    assert(Data[I] == Value);
  }
}

template <typename T> void test_copy_ptr_acc() {
  const size_t Size = 10;
  T Data[Size] = {0};
  T Values[Size] = {0};
  for (size_t I = 0; I < Size; ++I) {
    Values[I] = I;
  }
  {
    buffer<T, 1> Buffer(Data, range<1>(Size));
    queue Queue;
    Queue.submit([&](handler &Cgh) {
      accessor<T, 1, access::mode::write, access::target::device> Accessor(
          Buffer, Cgh, range<1>(Size));
      Cgh.copy(Values, Accessor);
    });
  }
  for (size_t I = 0; I < Size; ++I) {
    assert(Data[I] == Values[I]);
  }

  // Check copy from 'const void *' memory to accessor.
  {
    buffer<T, 1> Buffer(Size);
    queue Queue;
    Queue.submit([&](handler &Cgh) {
      auto Acc = Buffer.template get_access<access::mode::discard_write>(Cgh);
      Cgh.copy(reinterpret_cast<const void *>(Values), Acc);
    });

    auto Acc = Buffer.template get_access<access::mode::read>();
    for (int I = 0; I < Size; ++I)
      assert(Acc[I] == Values[I]);
  }

  // Check copy from memory to 0-dimensional accessor.
  T SrcValue = 99;
  T DstValue = 0;
  {
    buffer<T, 1> DstBuf(&DstValue, range<1>(1));
    queue Queue;
    Queue.submit([&](handler &Cgh) {
      accessor<T, 0, access::mode::discard_write, access::target::device>
          DstAcc(DstBuf, Cgh);
      Cgh.copy(&SrcValue, DstAcc);
    });
  }
  assert(DstValue == 99);
}

template <typename T> void test_3D_copy_ptr_acc() {
  const range<3> Range{2, 3, 4};
  const size_t Size = 2 * 3 * 4;
  T Data[Size] = {0};
  T Values[Size] = {0};
  for (size_t I = 0; I < Size; ++I)
    Values[I] = I;

  {
    buffer<T, 3> Buffer(Data, Range);
    queue Queue;
    Queue.submit([&](handler &Cgh) {
      accessor<T, 3, access::mode::write, access::target::device> Accessor(
          Buffer, Cgh, Range);
      Cgh.copy(Values, Accessor);
    });
  }

  for (int I = 0; I < Size; ++I)
    assert(Data[I] == Values[I]);
}

template <typename T> void test_copy_acc_ptr() {
  const size_t Size = 10;
  T Data[Size] = {0};
  for (size_t I = 0; I < Size; ++I) {
    Data[I] = I;
  }
  T Values[Size] = {0};
  {
    buffer<T, 1> Buffer(Data, range<1>(Size));
    queue Queue;
    Queue.submit([&](handler &Cgh) {
      accessor<T, 1, access::mode::read, access::target::device> Accessor(
          Buffer, Cgh, range<1>(Size));
      Cgh.copy(Accessor, Values);
    });
  }
  for (size_t I = 0; I < Size; ++I) {
    assert(Data[I] == Values[I]);
  }

  // Check copy from 'const T' accessor to 'void *' memory.
  {
    buffer<const T, 1> Buffer(reinterpret_cast<const T *>(Data),
                              range<1>(Size));
    T Dst[Size] = {0};
    queue Queue;
    Queue.submit([&](handler &Cgh) {
      auto Acc = Buffer.template get_access<access::mode::read>(Cgh);
      Cgh.copy(Acc, reinterpret_cast<void *>(Dst));
    });
    Queue.wait();

    for (int I = 0; I < Size; ++I)
      assert(Dst[I] == Data[I]);
  }

  // Check copy from 0-dimensional accessor to memory
  T SrcValue = 99;
  T DstValue = 0;
  {
    buffer<T, 1> SrcBuf(&SrcValue, range<1>(1));
    queue Queue;
    Queue.submit([&](handler &Cgh) {
      accessor<T, 0, access::mode::read, access::target::device> SrcAcc(SrcBuf,
                                                                        Cgh);
      Cgh.copy(SrcAcc, &DstValue);
    });
  }
  assert(DstValue == 99);

  // Check copy from 0-dimensional placeholder accessor to memory
  SrcValue = 77;
  DstValue = 0;
  {
    buffer<T, 1> SrcBuf(&SrcValue, range<1>(1));
    accessor<T, 0, access::mode::read, access::target::device,
             access::placeholder::true_t>
        SrcAcc(SrcBuf);
    {
      queue Queue;
      Queue.submit([&](handler &Cgh) {
        Cgh.require(SrcAcc);
        Cgh.copy(SrcAcc, &DstValue);
      });
    }
  }
  assert(DstValue == 77);
}

template <typename T> void test_3D_copy_acc_ptr() {
  const range<3> Range{2, 3, 4};
  const size_t Size = 2 * 3 * 4;
  T Data[Size] = {0};
  T Values[Size] = {0};
  for (size_t I = 0; I < Size; ++I)
    Data[I] = I;

  {
    buffer<T, 3> Buffer(Data, Range);
    queue Queue;
    Queue.submit([&](handler &Cgh) {
      accessor<T, 3, access::mode::read, access::target::device> Accessor(
          Buffer, Cgh, Range);
      Cgh.copy(Accessor, Values);
    });
  }

  for (size_t I = 0; I < Size; ++I)
    assert(Data[I] == Values[I]);
}

template <typename T> void test_copy_shared_ptr_acc() {
  const size_t Size = 10;
  T Data[Size] = {0};
  std::shared_ptr<T[]> Values(new T[Size]());
  for (size_t I = 0; I < Size; ++I) {
    Values.get()[I] = I;
  }
  {
    buffer<T, 1> Buffer(Data, range<1>(Size));
    queue Queue;
    Queue.submit([&](handler &Cgh) {
      accessor<T, 1, access::mode::write, access::target::device> Accessor(
          Buffer, Cgh, range<1>(Size));
      Cgh.copy(Values, Accessor);
    });
  }
  for (size_t I = 0; I < Size; ++I) {
    assert(Data[I] == Values.get()[I]);
  }
}

template <typename T> void test_copy_shared_ptr_const_acc() {
  constexpr size_t Size = 10;
  T Data[Size] = {0};
  std::shared_ptr<const T[]> Values(new T[Size]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  {
    buffer<T, 1> Buffer(Data, range<1>(Size));
    queue Queue;
    Queue.submit([&](handler &Cgh) {
      accessor<T, 1, access::mode::write, access::target::device> Accessor(
          Buffer, Cgh, range<1>(Size));
      Cgh.copy(Values, Accessor);
    });
  }
  for (size_t I = 0; I < Size; ++I) {
    assert(Data[I] == Values.get()[I]);
  }
}

template <typename T> void test_copy_acc_shared_ptr() {
  const size_t Size = 10;
  T Data[Size] = {0};
  for (size_t I = 0; I < Size; ++I) {
    Data[I] = I;
  }
  std::shared_ptr<T[]> Values(new T[Size]());
  {
    buffer<T, 1> Buffer(Data, range<1>(Size));
    queue Queue;
    Queue.submit([&](handler &Cgh) {
      accessor<T, 1, access::mode::read, access::target::device> Accessor(
          Buffer, Cgh, range<1>(Size));
      Cgh.copy(Accessor, Values);
    });
  }
  for (size_t I = 0; I < Size; ++I) {
    assert(Data[I] == Values.get()[I]);
  }
}

template <typename T> void test_copy_acc_acc() {
  const size_t Size = 10;
  T Data[Size] = {0};
  for (size_t I = 0; I < Size; ++I) {
    Data[I] = I;
  }
  T Values[Size] = {0};
  {
    buffer<T, 1> BufferFrom(Data, range<1>(Size));
    buffer<T, 1> BufferTo(Values, range<1>(Size));
    queue Queue;
    Queue.submit([&](handler &Cgh) {
      accessor<T, 1, access::mode::read, access::target::device> AccessorFrom(
          BufferFrom, Cgh, range<1>(Size));
      accessor<T, 1, access::mode::write, access::target::device> AccessorTo(
          BufferTo, Cgh, range<1>(Size));
      Cgh.copy(AccessorFrom, AccessorTo);
    });
  }
  for (size_t I = 0; I < Size; ++I) {
    assert(Data[I] == Values[I]);
  }
}

/* This is the class used to name the kernel for the runtime.
 * This must be done when the kernel is expressed as a lambda. */
template <typename T> class rawPointer;

template <typename T> void test_update_host() {
  const size_t Size = 10;
  T Data[Size] = {0};
  {
    auto Buffer =
        buffer<T, 1>(Data, range<1>(Size), {property::buffer::use_host_ptr()});
    Buffer.set_final_data(nullptr);
    queue Queue;
    Queue.submit([&](handler &Cgh) {
      accessor<T, 1, access::mode::write, access::target::device> Accessor(
          Buffer, Cgh, range<1>(Size));
      Cgh.parallel_for<class rawPointer<T>>(range<1>{Size},
          [=](id<1> Index) {
        Accessor[Index] = Index.get(0); });
    });
    Queue.submit([&](handler &Cgh) {
      accessor<T, 1, access::mode::write, access::target::device> Accessor(
          Buffer, Cgh, range<1>(Size));
      Cgh.update_host(Accessor);
    });
  }
  for (size_t I = 0; I < Size; ++I) {
    assert(Data[I] == I);
  }
}

template <typename T> void test_2D_copy_acc_acc() {
  const size_t Size = 20;
  T Data[Size][Size] = {{0}};
  for (size_t I = 0; I < Size; ++I) {
    for (size_t J = 0; J < Size; ++J) {
      Data[I][J] = I + J * Size;
    }
  }
  T Values[Size][Size] = {{0}};
  {
    buffer<T, 2> BufferFrom((T *)Data, range<2>(Size, Size));
    buffer<T, 2> BufferTo((T *)Values, range<2>(Size, Size));
    queue Queue;
    Queue.submit([&](handler &Cgh) {
      accessor<T, 2, access::mode::read, access::target::device> AccessorFrom(
          BufferFrom, Cgh, range<2>(Size, Size));
      accessor<T, 2, access::mode::write, access::target::device> AccessorTo(
          BufferTo, Cgh, range<2>(Size, Size));
      Cgh.copy(AccessorFrom, AccessorTo);
    });
  }

  for (size_t I = 0; I < Size; ++I) {
    for (size_t J = 0; J < Size; ++J) {
      assert(Data[I][J] == Values[I][J]);
    }
  }
}

template <typename T> void test_3D_copy_acc_acc() {
  const size_t Size = 20;
  T Data[Size][Size][Size] = {{{0}}};
  for (size_t I = 0; I < Size; ++I) {
    for (size_t J = 0; J < Size; ++J) {
      for (size_t K = 0; K < Size; ++K) {
        Data[I][J][K] = I + J * Size + K * Size * Size;
      }
    }
  }
  T Values[Size][Size][Size] = {{{0}}};
  {
    buffer<T, 3> BufferFrom((T *)Data, range<3>(Size, Size, Size));
    buffer<T, 3> BufferTo((T *)Values, range<3>(Size, Size, Size));
    queue Queue;
    Queue.submit([&](handler &Cgh) {
      accessor<T, 3, access::mode::read, access::target::device> AccessorFrom(
          BufferFrom, Cgh, range<3>(Size, Size, Size));
      accessor<T, 3, access::mode::write, access::target::device> AccessorTo(
          BufferTo, Cgh, range<3>(Size, Size, Size));
      Cgh.copy(AccessorFrom, AccessorTo);
    });
  }

  for (size_t I = 0; I < Size; ++I) {
    for (size_t J = 0; J < Size; ++J) {
      for (size_t K = 0; K < Size; ++K) {
        assert(Data[I][J][K] == Values[I][J][K]);
      }
    }
  }
}

template <typename T> void test_0D1D_copy_acc_acc() {
  // Copy 1 element from 0-dim accessor to 1-dim accessor
  T Src(1), Dst(0);
  {
    buffer<T, 1> BufferFrom(&Src, range<1>(1));
    buffer<T, 1> BufferTo(&Dst, range<1>(1));
    queue Queue;
    Queue.submit([&](handler &Cgh) {
      accessor<T, 0, access::mode::read, access::target::device> AccessorFrom(
          BufferFrom, Cgh);
      accessor<T, 1, access::mode::write, access::target::device> AccessorTo(
          BufferTo, Cgh);
      Cgh.copy(AccessorFrom, AccessorTo);
    });
  }
  assert(Dst == 1);

  // Copy 1 element from 1-dim accessor to 0-dim accessor
  Src = T(3);
  Dst = T(0);
  {
    buffer<T, 1> BufferFrom(&Src, range<1>(1));
    buffer<T, 1> BufferTo(&Dst, range<1>(1));
    queue Queue;
    Queue.submit([&](handler &Cgh) {
      accessor<T, 1, access::mode::read, access::target::device> AccessorFrom(
          BufferFrom, Cgh);
      accessor<T, 0, access::mode::write, access::target::device> AccessorTo(
          BufferTo, Cgh);
      Cgh.copy(AccessorFrom, AccessorTo);
    });
  }
  assert(Dst == 3);

  // Copy 1 element from 0-dim accessor to 0-dim accessor
  Src = T(7);
  Dst = T(0);
  {
    buffer<T, 1> BufferFrom(&Src, range<1>(1));
    buffer<T, 1> BufferTo(&Dst, range<1>(1));
    queue Queue;
    Queue.submit([&](handler &Cgh) {
      accessor<T, 0, access::mode::read, access::target::device> AccessorFrom(
          BufferFrom, Cgh);
      accessor<T, 0, access::mode::write, access::target::device> AccessorTo(
          BufferTo, Cgh);
      Cgh.copy(AccessorFrom, AccessorTo);
    });
  }
  assert(Dst == 7);
}

template <typename T> void test_1D2D_copy_acc_acc() {
  const size_t Size = 20;
  std::vector<T> Data(Size);
  std::iota(Data.begin(), Data.end(), 0);
  std::vector<T> Values(Size, T{});
  {
    buffer<T, 1> BufferFrom(&Data[0], range<1>(Size));
    buffer<T, 2> BufferTo(&Values[0], range<2>(Size / 2, 2));
    queue Queue;
    Queue.submit([&](handler &Cgh) {
      accessor<T, 1, access::mode::read, access::target::device> AccessorFrom(
          BufferFrom, Cgh, range<1>(Size));
      accessor<T, 2, access::mode::write, access::target::device> AccessorTo(
          BufferTo, Cgh, range<2>(Size / 2, 2));
      Cgh.copy(AccessorFrom, AccessorTo);
    });
  }
  assert(Data == Values);
}

template <typename T> void test_1D3D_copy_acc_acc() {
  const size_t Size = 20;
  std::vector<T> Data(Size);
  std::iota(Data.begin(), Data.end(), 0);
  std::vector<T> Values(Size, T{});
  {
    buffer<T, 1> BufferFrom(&Data[0], range<1>(Size));
    buffer<T, 3> BufferTo(&Values[0], range<3>(Size / 4, 2, 2));
    queue Queue;
    Queue.submit([&](handler &Cgh) {
      accessor<T, 1, access::mode::read, access::target::device> AccessorFrom(
          BufferFrom, Cgh, range<1>(Size));
      accessor<T, 3, access::mode::write, access::target::device> AccessorTo(
          BufferTo, Cgh, range<3>(Size / 4, 2, 2));
      Cgh.copy(AccessorFrom, AccessorTo);
    });
  }
  assert(Data == Values);
}

template <typename T> void test_2D1D_copy_acc_acc() {
  const size_t Size = 20;
  std::vector<T> Data(Size);
  std::iota(Data.begin(), Data.end(), 0);
  std::vector<T> Values(Size, T{});
  {
    buffer<T, 2> BufferFrom(&Data[0], range<2>(Size / 2, 2));
    buffer<T, 1> BufferTo(&Values[0], range<1>(Size));
    queue Queue;
    Queue.submit([&](handler &Cgh) {
      accessor<T, 2, access::mode::read, access::target::device> AccessorFrom(
          BufferFrom, Cgh, range<2>(Size / 2, 2));
      accessor<T, 1, access::mode::write, access::target::device> AccessorTo(
          BufferTo, Cgh, range<1>(Size));
      Cgh.copy(AccessorFrom, AccessorTo);
    });
  }
  assert(Data == Values);
}

template <typename T> void test_2D3D_copy_acc_acc() {
  const size_t Size = 20;
  std::vector<T> Data(Size);
  std::iota(Data.begin(), Data.end(), 0);
  std::vector<T> Values(Size, T{});
  {
    buffer<T, 2> BufferFrom(&Data[0], range<2>(Size / 2, 2));
    buffer<T, 3> BufferTo(&Values[0], range<3>(Size / 4, 2, 2));
    queue Queue;
    Queue.submit([&](handler &Cgh) {
      accessor<T, 2, access::mode::read, access::target::device> AccessorFrom(
          BufferFrom, Cgh, range<2>(Size / 2, 2));
      accessor<T, 3, access::mode::write, access::target::device> AccessorTo(
          BufferTo, Cgh, range<3>(Size / 4, 2, 2));
      Cgh.copy(AccessorFrom, AccessorTo);
    });
  }
  assert(Data == Values);
}

template <typename T> void test_3D1D_copy_acc_acc() {
  const size_t Size = 20;
  std::vector<T> Data(Size);
  std::iota(Data.begin(), Data.end(), 0);
  std::vector<T> Values(Size, T{});
  {
    buffer<T, 3> BufferFrom(&Data[0], range<3>(Size / 4, 2, 2));
    buffer<T, 1> BufferTo(&Values[0], range<1>(Size));
    queue Queue;
    Queue.submit([&](handler &Cgh) {
      accessor<T, 3, access::mode::read, access::target::device> AccessorFrom(
          BufferFrom, Cgh, range<3>(Size / 4, 2, 2));
      accessor<T, 1, access::mode::write, access::target::device> AccessorTo(
          BufferTo, Cgh, range<1>(Size));
      Cgh.copy(AccessorFrom, AccessorTo);
    });
  }
  assert(Data == Values);
}

template <typename T> void test_3D2D_copy_acc_acc() {
  const size_t Size = 20;
  std::vector<T> Data(Size);
  std::iota(Data.begin(), Data.end(), 0);
  std::vector<T> Values(Size, T{});
  {
    buffer<T, 3> BufferFrom(&Data[0], range<3>(Size / 4, 2, 2));
    buffer<T, 2> BufferTo(&Values[0], range<2>(Size / 2, 2));
    queue Queue;
    Queue.submit([&](handler &Cgh) {
      accessor<T, 3, access::mode::read, access::target::device> AccessorFrom(
          BufferFrom, Cgh, range<3>(Size / 4, 2, 2));
      accessor<T, 2, access::mode::write, access::target::device> AccessorTo(
          BufferTo, Cgh, range<2>(Size / 2, 2));
      Cgh.copy(AccessorFrom, AccessorTo);
    });
  }
  assert(Data == Values);
}
