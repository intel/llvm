#include <CL/sycl.hpp>

using namespace cl::sycl;

// Initializes 'InBuf' buffer with pseudo-random values, computes the reduction
// value for the buffer and writes it to 'ExpectedOut'.
template <typename T, class BinaryOperation>
void initInputData(buffer<T, 1> &InBuf, T &ExpectedOut, T Identity,
                   BinaryOperation BOp, size_t N) {
  ExpectedOut = Identity;
  auto In = InBuf.template get_access<access::mode::write>();
  for (int I = 0; I < N; ++I) {
    if (std::is_same<BinaryOperation, std::multiplies<T>>::value)
      In[I] = 1 + (((I % 37) == 0) ? 1 : 0);
    else
      In[I] = ((I + 1) % 5) + 1.1;
    ExpectedOut = BOp(ExpectedOut, In[I]);
  }
};

// This type is needed only to check that custom types are properly handled
// in parallel_for() with reduction. For simplicity it needs a default
// constructor, a constructor with one argument, operators ==, != and
// printing to a stream.
template <typename T>
struct CustomVec {
  CustomVec() : X(0), Y(0) {}
  CustomVec(T X, T Y) : X(X), Y(Y) {}
  CustomVec(T V) : X(V), Y(V) {}
  bool operator==(const CustomVec &V) const {
    return V.X == X && V.Y == Y;
  }
  bool operator!=(const CustomVec &V) const {
    return !(*this == V);
  }
  T X;
  T Y;
};
template <typename T>
bool operator==(const CustomVec<T> &A, const CustomVec<T> &B) {
  return A.X == B.X && A.Y == B.Y;
}
template <typename T>
std::ostream &operator<<(std::ostream &OS, const CustomVec<T> &V) {
  return OS << "(" << V.X << ", " << V.Y << ")";
}

template <class T>
struct CustomVecPlus {
  using CV = CustomVec<T>;
  CV operator()(const CV &A, const CV &B) const {
    return CV(A.X + B.X, A.Y + B.Y);
  }
};

template <typename T>
T getMinimumFPValue() {
  return std::numeric_limits<T>::has_infinity
             ? static_cast<T>(-std::numeric_limits<T>::infinity())
             : std::numeric_limits<T>::lowest();
}

template <typename T>
T getMaximumFPValue() {
  return std::numeric_limits<T>::has_infinity
             ? std::numeric_limits<T>::infinity()
             : (std::numeric_limits<T>::max)();
}
