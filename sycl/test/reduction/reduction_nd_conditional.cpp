// UNSUPPORTED: cuda
// Reductions use work-group builtins not yet supported by CUDA.
//
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUNx: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// This test performs basic checks of parallel_for(nd_range, reduction, func)
// with reduction and conditional increment of the reduction variable.

#include <CL/sycl.hpp>
#include <cassert>

using namespace cl::sycl;

template <typename T, class BinaryOperation>
void initInputData(buffer<T, 1> &InBuf, T &ExpectedOut, T Identity,
                   BinaryOperation BOp, size_t N) {
  ExpectedOut = Identity;
  auto In = InBuf.template get_access<access::mode::write>();
  for (int I = 0; I < N; ++I) {
    if (std::is_same<BinaryOperation, std::multiplies<T>>::value)
      In[I] = 1 + (((I % 37) == 0) ? 1 : 0);
    else
      In[I] = I + 1 + 1.1;

    if (I < 2)
      ExpectedOut = BOp(ExpectedOut, 99);
    else if (I % 3)
      ExpectedOut = BOp(ExpectedOut, In[I]);
    else
      ; // do nothing.
  }
};

template <typename T, int Dim, class BinaryOperation>
class SomeClass;

template <typename T>
struct Vec {
  Vec() : X(0), Y(0) {}
  Vec(T X, T Y) : X(X), Y(Y) {}
  Vec(T V) : X(V), Y(V) {}
  bool operator==(const Vec &P) const {
    return P.X == X && P.Y == Y;
  }
  bool operator!=(const Vec &P) const {
    return !(*this == P);
  }
  T X;
  T Y;
};
template <typename T>
bool operator==(const Vec<T> &A, const Vec<T> &B) {
  return A.X == B.X && A.Y == B.Y;
}
template <typename T>
std::ostream &operator<<(std::ostream &OS, const Vec<T> &P) {
  return OS << "(" << P.X << ", " << P.Y << ")";
}

template <class T>
struct VecPlus {
  using P = Vec<T>;
  P operator()(const P &A, const P &B) const {
    return P(A.X + B.X, A.Y + B.Y);
  }
};

template <typename T, int Dim, class BinaryOperation>
void test(T Identity, size_t WGSize, size_t NWItems) {
  buffer<T, 1> InBuf(NWItems);
  buffer<T, 1> OutBuf(1);

  // Initialize.
  BinaryOperation BOp;
  T CorrectOut;
  initInputData(InBuf, CorrectOut, Identity, BOp, NWItems);

  // Compute.
  queue Q;
  Q.submit([&](handler &CGH) {
    auto In = InBuf.template get_access<access::mode::read>(CGH);
    accessor<T, Dim, access::mode::discard_write, access::target::global_buffer>
        Out(OutBuf, CGH);
    auto Redu = intel::reduction(Out, Identity, BOp);

    range<1> GlobalRange(NWItems);
    range<1> LocalRange(WGSize);
    nd_range<1> NDRange(GlobalRange, LocalRange);
    CGH.parallel_for<SomeClass<T, Dim, BinaryOperation>>(
        NDRange, Redu, [=](nd_item<1> NDIt, auto &Sum) {
          size_t I = NDIt.get_global_linear_id();
          if (I < 2)
            Sum.combine(T(99));
          else if (I % 3)
            Sum.combine(In[I]);
          else
            ; // do nothing.
        });
  });

  // Check correctness.
  auto Out = OutBuf.template get_access<access::mode::read>();
  T ComputedOut = *(Out.get_pointer());
  if (ComputedOut != CorrectOut) {
    std::cout << "NWItems = " << NWItems << ", WGSize = " << WGSize << "\n";
    std::cout << "Computed value: " << ComputedOut
              << ", Expected value: " << CorrectOut << "\n";
    assert(0 && "Wrong value.");
  }
}

int main() {
  test<int, 0, intel::plus<int>>(0, 2, 2);
  test<int, 1, intel::plus<int>>(0, 7, 7);
  test<int, 0, intel::plus<int>>(0, 2, 64);
  test<short, 1, intel::plus<short>>(0, 16, 256);

  std::cout << "Test passed\n";
  return 0;
}
