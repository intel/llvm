#include <CL/sycl.hpp>

using namespace cl::sycl;

/// Initializes the buffer<1> \p 'InBuf' buffer with pseudo-random values,
/// computes the write the reduction value \p 'ExpectedOut'.
template <typename T, class BinaryOperation>
void initInputData(buffer<T, 1> &InBuf, T &ExpectedOut, T Identity,
                   BinaryOperation BOp, range<1> Range) {
  ExpectedOut = Identity;
  size_t N = Range.size();
  auto In = InBuf.template get_access<access::mode::write>();
  for (int I = 0; I < N; ++I) {
    if (std::is_same<BinaryOperation, std::multiplies<T>>::value)
      In[I] = 1 + (((I % 37) == 0) ? 1 : 0);
    else
      In[I] = ((I + 1) % 5) + 1.1;
    ExpectedOut = BOp(ExpectedOut, In[I]);
  }
};

/// Initializes the buffer<2> \p 'InBuf' buffer with pseudo-random values,
/// computes the write the reduction value \p 'ExpectedOut'.
template <typename T, class BinaryOperation>
void initInputData(buffer<T, 2> &InBuf, T &ExpectedOut, T Identity,
                   BinaryOperation BOp, range<2> Range) {
  ExpectedOut = Identity;
  auto In = InBuf.template get_access<access::mode::write>();
  for (int J = 0; J < Range[0]; ++J) {
    for (int I = 0; I < Range[1]; ++I) {
      if (std::is_same<BinaryOperation, std::multiplies<T>>::value)
        In[J][I] = 1 + ((((I * 2 + J * 3) % 37) == 0) ? 1 : 0);
      else
        In[J][I] = ((I + 1 + J) % 5) + 1.1;
      ExpectedOut = BOp(ExpectedOut, In[J][I]);
    }
  }
};

/// Initializes the buffer<3> \p 'InBuf' buffer with pseudo-random values,
/// computes the write the reduction value \p 'ExpectedOut'.
template <typename T, class BinaryOperation>
void initInputData(buffer<T, 3> &InBuf, T &ExpectedOut, T Identity,
                   BinaryOperation BOp, range<3> Range) {
  ExpectedOut = Identity;
  auto In = InBuf.template get_access<access::mode::write>();
  for (int K = 0; K < Range[0]; ++K) {
    for (int J = 0; J < Range[1]; ++J) {
      for (int I = 0; I < Range[2]; ++I) {
        if (std::is_same<BinaryOperation, std::multiplies<T>>::value)
          In[K][J][I] = 1 + ((((I * 2 + J * 3 + K) % 37) == 0) ? 1 : 0);
        else
          In[K][J][I] = ((I + 1 + J + K * 3) % 5) + 1.1;
        ExpectedOut = BOp(ExpectedOut, In[K][J][I]);
      }
    }
  }
};

// This type is needed only to check that custom types are properly handled
// in parallel_for() with reduction. For simplicity it needs a default
// constructor, a constructor with one argument, operators ==, != and
// printing to a stream.
template <typename T> struct CustomVec {
  CustomVec() : X(0), Y(0) {}
  CustomVec(T X, T Y) : X(X), Y(Y) {}
  CustomVec(T V) : X(V), Y(V) {}
  bool operator==(const CustomVec &V) const { return V.X == X && V.Y == Y; }
  bool operator!=(const CustomVec &V) const { return !(*this == V); }
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

template <class T> struct CustomVecPlus {
  using CV = CustomVec<T>;
  CV operator()(const CV &A, const CV &B) const {
    return CV(A.X + B.X, A.Y + B.Y);
  }
};

template <typename T> T getMinimumFPValue() {
  return std::numeric_limits<T>::has_infinity
             ? static_cast<T>(-std::numeric_limits<T>::infinity())
             : std::numeric_limits<T>::lowest();
}

template <typename T> T getMaximumFPValue() {
  return std::numeric_limits<T>::has_infinity
             ? std::numeric_limits<T>::infinity()
             : (std::numeric_limits<T>::max)();
}

template <access::mode Mode> property_list getPropertyList() {
  if constexpr (Mode == access::mode::read_write)
    return property_list();
  return property_list(property::reduction::initialize_to_identity{});
}

void printDeviceInfo(queue &Q, bool ToCERR = false) {
  device D = Q.get_device();
  auto Name = D.get_info<sycl::info::device::name>();
  size_t MaxWGSize = D.get_info<info::device::max_work_group_size>();
  size_t LocalMemSize = D.get_info<info::device::local_mem_size>();
  if (ToCERR)
    std::cout << "Device: " << Name << ", MaxWGSize: " << MaxWGSize
              << ", LocalMemSize: " << LocalMemSize << std::endl;
  else
    std::cerr << "Device: " << Name << ", MaxWGSize: " << MaxWGSize
              << ", LocalMemSize: " << LocalMemSize << std::endl;
}
