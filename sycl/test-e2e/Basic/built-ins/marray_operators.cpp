// Verifies marray's raw operators produce identical results on host and device.
//
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/half_type.hpp>
#include <sycl/marray.hpp>

#include <algorithm>
#include <cassert>

using namespace sycl;

template <typename FnT> bool runOnDevice(queue &Q, FnT Fn) {
  buffer<bool, 1> ResultBuf{1};
  Q.submit([&](handler &Cgh) {
    accessor Result{ResultBuf, Cgh};
    Cgh.single_task([=]() { Result[0] = Fn(); });
  });
  return host_accessor{ResultBuf}[0];
}

template <typename MArrayT> bool equal(const MArrayT &A, const MArrayT &B) {
  return std::equal(A.begin(), A.end(), B.begin());
}

template <typename T, size_t N> marray<T, N> makeA() {
  marray<T, N> Ret;
  for (size_t I = 0; I < N; ++I)
    if constexpr (std::is_same_v<T, bool>)
      Ret[I] = (I % 2 == 0);
    else
      Ret[I] = T(I + 1);
  return Ret;
}

template <typename T, size_t N> marray<T, N> makeB() {
  if constexpr (std::is_same_v<T, bool>)
    return marray<T, N>(true);
  else
    return marray<T, N>(T(2));
}

// Binary op, marray-marray. The same Op runs on host (the oracle) and device;
// their results must match.
template <typename T, size_t N, typename OpT>
void checkBinary(queue &Q, OpT Op) {
  auto A = makeA<T, N>();
  auto B = makeB<T, N>();
  auto Expected = Op(A, B);
  assert(runOnDevice(Q, [=]() { return equal(Op(A, B), Expected); }));
}

template <typename T, size_t N, typename OpT>
void checkScalarOrders(queue &Q, OpT Op) {
  auto A = makeA<T, N>();
  T Scalar = T(2);
  auto ExpectedRhs = Op(A, Scalar);
  assert(runOnDevice(Q, [=]() { return equal(Op(A, Scalar), ExpectedRhs); }));
  auto ExpectedLhs = Op(Scalar, A);
  assert(runOnDevice(Q, [=]() { return equal(Op(Scalar, A), ExpectedLhs); }));
}

template <typename T, size_t N, typename OpT>
void checkUnary(queue &Q, OpT Op) {
  auto A = makeA<T, N>();
  auto Expected = Op(A);
  assert(runOnDevice(Q, [=]() { return equal(Op(A), Expected); }));
}

template <typename T, size_t N, typename OpT>
void checkAssign(queue &Q, OpT Op) {
  auto A = makeA<T, N>();
  auto B = makeB<T, N>();
  T Scalar = T(2);
  auto ExpectedRhs = A;
  Op(ExpectedRhs, B);
  assert(runOnDevice(Q, [=]() {
    auto Lhs = A;
    Op(Lhs, B);
    return equal(Lhs, ExpectedRhs);
  }));
  auto ExpectedScalar = A;
  Op(ExpectedScalar, Scalar);
  assert(runOnDevice(Q, [=]() {
    auto Lhs = A;
    Op(Lhs, Scalar);
    return equal(Lhs, ExpectedScalar);
  }));
}

template <typename T, size_t N, typename OpT>
void checkMutate(queue &Q, OpT Op) {
  auto A = makeA<T, N>();
  auto Expected = A;
  Op(Expected);
  assert(runOnDevice(Q, [=]() {
    auto Lhs = A;
    Op(Lhs);
    return equal(Lhs, Expected);
  }));
}

template <typename T, size_t N> void checkNumericAt(queue &Q) {
  checkBinary<T, N>(Q, [](auto A, auto B) { return A + B; });
  checkBinary<T, N>(Q, [](auto A, auto B) { return A == B; });
  checkUnary<T, N>(Q, [](auto A) { return -A; });
  checkMutate<T, N>(Q, [](auto &X) { ++X; });
}

template <typename T, size_t N> void checkIntegralAt(queue &Q) {
  checkBinary<T, N>(Q, [](auto A, auto B) { return A & B; });
  checkUnary<T, N>(Q, [](auto A) { return ~A; });
  checkUnary<T, N>(Q, [](auto A) { return !A; });
}

int main() {
  queue Q;

  // Run each op at a power-of-two N and at N=3, since the operators take
  // different code paths depending on whether N is vectorizable.
  checkNumericAt<int, 4>(Q);
  checkNumericAt<int, 3>(Q);
  checkIntegralAt<int, 4>(Q);
  checkIntegralAt<int, 3>(Q);
  checkScalarOrders<int, 4>(Q, [](auto A, auto B) { return A + B; });
  checkScalarOrders<int, 4>(Q, [](auto A, auto B) { return A & B; });
  checkAssign<int, 4>(Q, [](auto &X, auto Y) { X += Y; });
  checkAssign<int, 4>(Q, [](auto &X, auto Y) { X &= Y; });

  if (Q.get_device().has(aspect::fp16))
    checkBinary<half, 4>(Q, [](auto A, auto B) { return A + B; });

  // bool: operator~ has a special case (returns !Lhs). ++/-- are deprecated for
  // marray<bool>, so they are not exercised here.
  checkBinary<bool, 4>(Q, [](auto A, auto B) { return A & B; });
  checkUnary<bool, 4>(Q, [](auto A) { return ~A; });
  checkUnary<bool, 4>(Q, [](auto A) { return !A; });

  return 0;
}
