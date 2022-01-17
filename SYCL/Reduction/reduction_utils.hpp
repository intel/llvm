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
    if (std::is_same_v<BinaryOperation, std::multiplies<T>> ||
        std::is_same_v<BinaryOperation, std::multiplies<>>)
      In[I] = 1.1 + (((I % 11) == 0) ? 1 : 0);
    else if (std::is_same_v<BinaryOperation, std::bit_and<T>> ||
             std::is_same_v<BinaryOperation, std::bit_and<>>)
      In[I] = (I + 1) | 0x10203040;
    else if (std::is_same_v<BinaryOperation, sycl::minimum<T>> ||
             std::is_same_v<BinaryOperation, sycl::minimum<>>)
      In[I] = Range[0] - I;
    else if (std::is_same_v<BinaryOperation, sycl::maximum<T>> ||
             std::is_same_v<BinaryOperation, sycl::maximum<>>)
      In[I] = I;
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
      if (std::is_same_v<BinaryOperation, std::multiplies<T>> ||
          std::is_same_v<BinaryOperation, std::multiplies<>>)
        In[J][I] = 1.1 + ((((I + J * 3) % 11) == 0) ? 1 : 0);
      else if (std::is_same_v<BinaryOperation, std::bit_and<T>> ||
               std::is_same_v<BinaryOperation, std::bit_and<>>)
        In[J][I] = (I + J + 1) | 0x10203040;
      else if (std::is_same_v<BinaryOperation, sycl::minimum<T>> ||
               std::is_same_v<BinaryOperation, sycl::minimum<>>)
        In[J][I] = Range[0] + Range[1] - I - J;
      else if (std::is_same_v<BinaryOperation, sycl::maximum<T>> ||
               std::is_same_v<BinaryOperation, sycl::maximum<>>)
        In[J][I] = I + J;
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
        if (std::is_same_v<BinaryOperation, std::multiplies<T>> ||
            std::is_same_v<BinaryOperation, std::multiplies<>>)
          In[K][J][I] = 1.1 + ((((I + J * 3 + K) % 11) == 0) ? 1 : 0);
        else if (std::is_same_v<BinaryOperation, std::bit_and<T>> ||
                 std::is_same_v<BinaryOperation, std::bit_and<>>)
          In[K][J][I] = (I + J + K + 1) | 0x10203040;
        else if (std::is_same_v<BinaryOperation, sycl::minimum<T>> ||
                 std::is_same_v<BinaryOperation, sycl::minimum<>>)
          In[K][J][I] = Range[0] + Range[1] + Range[2] - I - J - K;
        else if (std::is_same_v<BinaryOperation, sycl::maximum<T>> ||
                 std::is_same_v<BinaryOperation, sycl::maximum<>>)
          In[K][J][I] = I + J + K;
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
bool operator<(const CustomVec<T> &A, const CustomVec<T> &B) {
  return A.X < B.X && A.Y < B.Y;
}
template <typename T>
CustomVec<T> operator/(const CustomVec<T> &A, const CustomVec<T> &B) {
  return {A.X / B.X && A.Y / B.Y};
}
template <typename T>
CustomVec<T> operator-(const CustomVec<T> &A, const CustomVec<T> &B) {
  return {A.X - B.X && A.Y - B.Y};
}
namespace std {
template <typename T> CustomVec<T> abs(const CustomVec<T> &A) {
  return {std::abs(A.X), std::abs(A.Y)};
}
} // namespace std
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
  static int IsErrDeviceInfoPrinted = 0;
  if (IsErrDeviceInfoPrinted >= 2)
    return;
  IsErrDeviceInfoPrinted++;

  device D = Q.get_device();
  auto Name = D.get_info<sycl::info::device::name>();
  size_t MaxWGSize = D.get_info<info::device::max_work_group_size>();
  size_t LocalMemSize = D.get_info<info::device::local_mem_size>();
  std::ostream &OS = ToCERR ? std::cerr : std::cout;
  OS << "Device: " << Name << ", MaxWGSize: " << MaxWGSize
     << ", LocalMemSize: " << LocalMemSize
     << ", Driver: " << D.get_info<info::device::driver_version>() << std::endl;
}

template <int Dims>
std::ostream &operator<<(std::ostream &OS, const range<Dims> &Range) {
  OS << "{" << Range[0];
  if constexpr (Dims > 1)
    OS << ", " << Range[1];
  if constexpr (Dims > 2)
    OS << ", " << Range[2];
  OS << "}";
  return OS;
}

template <int Dims>
std::ostream &operator<<(std::ostream &OS, const nd_range<Dims> &Range) {
  OS << "{" << Range.get_global_range() << ", " << Range.get_local_range()
     << "}";
  return OS;
}

template <typename T, typename BinaryOperation, typename RangeT>
void printTestLabel(bool IsSYCL2020, const RangeT &Range, bool ToCERR = false) {
  std::ostream &OS = ToCERR ? std::cerr : std::cout;
  std::string Mode = IsSYCL2020 ? "SYCL2020" : "ext::oneapi  ";
  OS << (ToCERR ? "Error" : "Start") << ": Mode=" << Mode
     << ", T=" << typeid(T).name() << ", BOp=" << typeid(BinaryOperation).name()
     << ", Range=" << Range;
}

template <typename BOp, typename T> constexpr bool isPreciseResultFP() {
  return (std::is_floating_point_v<T> || std::is_same_v<T, sycl::half>)&&(
      std::is_same_v<ext::oneapi::minimum<>, BOp> ||
      std::is_same_v<ext::oneapi::minimum<T>, BOp> ||
      std::is_same_v<ext::oneapi::maximum<>, BOp> ||
      std::is_same_v<ext::oneapi::maximum<T>, BOp>);
}

template <typename BinaryOperation, typename T, typename RangeT>
int checkResults(queue &Q, bool IsSYCL2020, BinaryOperation,
                 const RangeT &Range, const T &ComputedRes, const T &CorrectRes,
                 std::string AddInfo = "") {
  std::string ErrorStr;
  bool Passed;

  if constexpr (std::is_floating_point_v<T> || std::is_same_v<T, sycl::half>) {
    // It is a pretty simple and naive FP diff check here, which though
    // should work reasonably well for most of cases handled in reduction
    // tests.
    T MaxDiff = std::numeric_limits<T>::epsilon() * std::fabs(CorrectRes);
    if constexpr (std::is_same_v<RangeT, range<1>> ||
                  std::is_same_v<RangeT, range<2>> ||
                  std::is_same_v<RangeT, range<3>>)
      MaxDiff *= Range.size();
    else
      MaxDiff *= Range.get_global_range().size();

    if (isPreciseResultFP<BinaryOperation, T>())
      MaxDiff = 0;
    T Diff = std::abs(CorrectRes - ComputedRes);
    ErrorStr = ", Diff=" + std::to_string(Diff) +
               ", MaxDiff=" + std::to_string(MaxDiff);
    Passed = Diff <= MaxDiff;
  } else {
    Passed = ComputedRes == CorrectRes;
  }

  if (!AddInfo.empty())
    AddInfo = std::string(", ") + AddInfo;
  std::cout << AddInfo << (Passed ? ". PASSED" : ". FAILED") << std::endl;
  if (!Passed) {
    printDeviceInfo(Q, true);
    printTestLabel<T, BinaryOperation>(IsSYCL2020, Range, true);
    std::cerr << ", Computed value=" << ComputedRes
              << ", Expected value=" << CorrectRes << ErrorStr << AddInfo
              << std::endl;
  }
  return Passed ? 0 : 1;
}

void printFinalStatus(int NumErrors) {
  if (NumErrors == 0)
    std::cout << "Test passed" << std::endl;
  else
    std::cerr << NumErrors << " test-cases failed" << std::endl;
}

template <bool IsSYCL2020, access::mode AccMode = access::mode::read_write,
          typename T, typename BinaryOperation>
auto createReduction(T *USMPtr, T Identity, BinaryOperation BOp) {
  if constexpr (IsSYCL2020)
    return sycl::reduction(USMPtr, Identity, BOp, getPropertyList<AccMode>());
  else
    return ext::oneapi::reduction(USMPtr, Identity, BOp);
}

template <bool IsSYCL2020, access::mode AccMode = access::mode::read_write,
          typename T, typename BinaryOperation>
auto createReduction(T *USMPtr, BinaryOperation BOp) {
  if constexpr (IsSYCL2020)
    return sycl::reduction(USMPtr, BOp, getPropertyList<AccMode>());
  else
    return ext::oneapi::reduction(USMPtr, BOp);
}

template <bool IsSYCL2020, access::mode AccMode, int AccDim = 1, typename T,
          typename BinaryOperation, typename BufferT>
auto createReduction(BufferT ReduBuf, handler &CGH, T Identity,
                     BinaryOperation BOp) {
  if constexpr (IsSYCL2020) {
    property_list PropList = getPropertyList<AccMode>();
    return sycl::reduction(ReduBuf, CGH, Identity, BOp, PropList);
  } else {
    accessor<T, AccDim, AccMode, access::target::device> Out(ReduBuf, CGH);
    return ext::oneapi::reduction(Out, Identity, BOp);
  }
}

aspect getUSMAspect(usm::alloc Alloc) {
  if (Alloc == sycl::usm::alloc::host)
    return aspect::usm_host_allocations;

  if (Alloc == sycl::usm::alloc::device)
    return aspect::usm_device_allocations;

  assert(Alloc == usm::alloc::shared && "Unknown USM allocation type");
  return aspect::usm_shared_allocations;
}
