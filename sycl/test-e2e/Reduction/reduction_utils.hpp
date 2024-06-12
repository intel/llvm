#include <cmath>
#include <iostream>
#include <optional>

#include <sycl/detail/core.hpp>

#include <sycl/ext/oneapi/functional.hpp>
#include <sycl/reduction.hpp>

using namespace sycl;

struct AllIdOp {
  constexpr bool operator()(size_t Idx) const { return true; }
};

struct SkipAllOp {
  constexpr bool operator()(size_t Idx) const { return false; }
};

struct SkipEvenOp {
  constexpr bool operator()(size_t Idx) const { return Idx % 2; }
};

struct SkipOddOp {
  constexpr bool operator()(size_t Idx) const { return (Idx + 1) % 2; }
};

/// Initializes the buffer<1> \p 'InBuf' buffer with pseudo-random values,
/// computes the write the reduction value \p 'ExpectedOut'. Linearized IDs are
/// filtered in \p 'ExpectedOut' using \p 'IdFilterFunc'.
template <typename T, class BinaryOperation, typename IdFilterFuncT = AllIdOp>
void initInputData(buffer<T, 1> &InBuf, std::optional<T> &ExpectedOut,
                   BinaryOperation BOp, range<1> Range,
                   IdFilterFuncT IdFilterFunc = {}) {
  size_t N = Range.size();
  assert(N != 0);
  host_accessor In(InBuf, write_only);
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
    if (IdFilterFunc(I))
      ExpectedOut = ExpectedOut ? BOp(*ExpectedOut, In[I]) : In[I];
  }
};

/// Initializes the buffer<2> \p 'InBuf' buffer with pseudo-random values,
/// computes the write the reduction value \p 'ExpectedOut'. Linearized IDs are
/// filtered in \p 'ExpectedOut' using \p 'IdFilterFunc'.
template <typename T, class BinaryOperation, typename IdFilterFuncT = AllIdOp>
void initInputData(buffer<T, 2> &InBuf, std::optional<T> &ExpectedOut,
                   BinaryOperation BOp, range<2> Range,
                   IdFilterFuncT IdFilterFunc = {}) {
  assert(Range.size() != 0);
  host_accessor In(InBuf, write_only);
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
      if (IdFilterFunc(I + J * Range[1]))
        ExpectedOut = ExpectedOut ? BOp(*ExpectedOut, In[J][I]) : In[J][I];
    }
  }
};

/// Initializes the buffer<3> \p 'InBuf' buffer with pseudo-random values,
/// computes the write the reduction value \p 'ExpectedOut'. Linearized IDs are
/// filtered in \p 'ExpectedOut' using \p 'IdFilterFunc'.
template <typename T, class BinaryOperation, typename IdFilterFuncT = AllIdOp>
void initInputData(buffer<T, 3> &InBuf, std::optional<T> &ExpectedOut,
                   BinaryOperation BOp, range<3> Range,
                   IdFilterFuncT IdFilterFunc = {}) {
  assert(Range.size() != 0);
  host_accessor In(InBuf, write_only);
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
        if (IdFilterFunc(I + J * Range[2] + K * Range[1] * Range[2]))
          ExpectedOut =
              ExpectedOut ? BOp(*ExpectedOut, In[K][J][I]) : In[K][J][I];
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

template <class T> struct PlusWithoutIdentity {
  T operator()(const T &A, const T &B) const { return A + B; }
};

template <class T> struct MultipliesWithoutIdentity {
  T operator()(const T &A, const T &B) const { return A * B; }
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

template <typename T, bool HasIdentity = false> struct OptionalIdentity {
  OptionalIdentity() {}
  OptionalIdentity(T IdentityVal) : MValue{IdentityVal} {}

  T get() const { return *MValue; }

private:
  std::optional<T> MValue;
};
template <typename T> OptionalIdentity(T) -> OptionalIdentity<T, true>;

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
void printTestLabel(const RangeT &Range, bool ToCERR = false) {
  std::ostream &OS = ToCERR ? std::cerr : std::cout;
  OS << (ToCERR ? "Error" : "Start") << ", T=" << typeid(T).name()
     << ", BOp=" << typeid(BinaryOperation).name() << ", Range=" << Range;
}

template <typename BOp, typename T> constexpr bool isPreciseResultFP() {
  return (std::is_floating_point_v<T> || std::is_same_v<T, sycl::half>)&&(
      std::is_same_v<ext::oneapi::minimum<>, BOp> ||
      std::is_same_v<ext::oneapi::minimum<T>, BOp> ||
      std::is_same_v<ext::oneapi::maximum<>, BOp> ||
      std::is_same_v<ext::oneapi::maximum<T>, BOp>);
}

template <typename BinaryOperation, typename T, typename RangeT>
int checkResults(queue &Q, BinaryOperation, const RangeT &Range,
                 const T &ComputedRes, const T &CorrectRes,
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
    printTestLabel<T, BinaryOperation>(Range, true);
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

aspect getUSMAspect(usm::alloc Alloc) {
  if (Alloc == sycl::usm::alloc::host)
    return aspect::usm_host_allocations;

  if (Alloc == sycl::usm::alloc::device)
    return aspect::usm_device_allocations;

  assert(Alloc == usm::alloc::shared && "Unknown USM allocation type");
  return aspect::usm_shared_allocations;
}

template <typename T, bool B> class KName;
template <typename T, typename> class TName;

/// Helper to make the code slightly more readable.
auto init_to_identity() {
  return property_list{property::reduction::initialize_to_identity{}};
}

template <typename Name, typename T, bool HasIdentity, class BinaryOperation,
          template <int> typename RangeTy, int Dims,
          typename PropListTy = property_list, typename IdFilterFuncT = AllIdOp>
int testInner(queue &Q, OptionalIdentity<T, HasIdentity> Identity, T Init,
              BinaryOperation BOp, const RangeTy<Dims> &Range,
              PropListTy PropList = {}, IdFilterFuncT IdFilterFunc = {}) {
  constexpr bool IsRange = std::is_same_v<range<Dims>, RangeTy<Dims>>;
  constexpr bool IsNDRange = std::is_same_v<nd_range<Dims>, RangeTy<Dims>>;
  static_assert(IsRange || IsNDRange);

  printTestLabel<T, BinaryOperation>(Range);

  // It is a known problem with passing data that is close to 4Gb in size
  // to device. Such data breaks the execution pretty badly.
  // Some of test cases calling this function try to verify the correctness
  // of reduction with the global range bigger than the maximal work-group size
  // for the device. Maximal WG size for device may be very big, e.g. it is
  // 67108864 for ACC emulator. Multiplying that by some factor
  // (to exceed max WG-Size) and multiplying it by the element size may exceed
  // the safe size of data passed to device.
  // Let's set it to 1 GB for now, and just skip the test if it exceeds 1Gb.
  constexpr size_t OneGB = 1LL * 1024 * 1024 * 1024;
  range<Dims> GlobalRange = [&]() {
    if constexpr (IsRange)
      return Range;
    else
      return Range.get_global_range();
  }();

  if (GlobalRange.size() * sizeof(T) > OneGB) {
    std::cout << " SKIPPED due to too big data size" << std::endl;
    return 0;
  }

  // TODO: Perhaps, this is a _temporary_ fix for CI. The test may run
  // for too long when the range is big. That is especially bad on ACC.
  if (GlobalRange.size() > 65536 && Q.get_device().is_accelerator()) {
    std::cout << " SKIPPED due to risk of timeout in CI" << std::endl;
    return 0;
  }

  buffer<T, Dims> InBuf(GlobalRange);
  buffer<T, 1> OutBuf(1);

  // Initialize.
  std::optional<T> CorrectOut;
  initInputData(InBuf, CorrectOut, BOp, GlobalRange, IdFilterFunc);
  if (!PropList.template has_property<
          property::reduction::initialize_to_identity>()) {
    CorrectOut = CorrectOut ? BOp(*CorrectOut, Init) : Init;
  }

  // The value assigned here must be discarded (if IsReadWrite is true).
  // Verify that it is really discarded and assign some value.
  host_accessor(OutBuf, write_only)[0] = Init;

  // Compute.
  Q.submit([&](handler &CGH) {
    // Helper for creating the reductions depending on the existance of an
    // identity.
    auto CreateReduction = [&]() {
      if constexpr (HasIdentity) {
        return reduction(OutBuf, CGH, Identity.get(), BOp, PropList);
      } else {
        return reduction(OutBuf, CGH, BOp, PropList);
      }
    };

    auto In = InBuf.template get_access<access::mode::read>(CGH);
    auto Redu = CreateReduction();
    if constexpr (IsRange)
      CGH.parallel_for<Name>(Range, Redu, [=](item<Dims> Id, auto &Sum) {
        if (IdFilterFunc(Id.get_linear_id()))
          Sum.combine(In[Id]);
      });
    else
      CGH.parallel_for<Name>(Range, Redu, [=](nd_item<Dims> NDIt, auto &Sum) {
        if (IdFilterFunc(NDIt.get_global_linear_id()))
          Sum.combine(In[NDIt.get_global_linear_id()]);
      });
  });

  // Check correctness.
  host_accessor Out(OutBuf, read_only);
  T ComputedOut = *(Out.get_pointer());
  return checkResults(Q, BOp, Range, ComputedOut, *CorrectOut);
}

template <typename Name, typename T, class BinaryOperation,
          template <int> typename RangeTy, int Dims,
          typename PropListTy = property_list, typename IdFilterFuncT = AllIdOp>
int test(queue &Q, T Identity, T Init, BinaryOperation BOp,
         const RangeTy<Dims> &Range, PropListTy PropList = {},
         IdFilterFuncT IdFilterFunc = {}) {
  return testInner<Name>(Q, OptionalIdentity(Identity), Init, BOp, Range,
                         PropList);
}

template <typename Name, typename T, class BinaryOperation,
          template <int> typename RangeTy, int Dims,
          typename PropListTy = property_list, typename IdFilterFuncT = AllIdOp>
int test(queue &Q, T Init, BinaryOperation BOp, const RangeTy<Dims> &Range,
         PropListTy PropList = {}, IdFilterFuncT IdFilterFunc = {}) {
  return testInner<Name>(Q, OptionalIdentity<T>(), Init, BOp, Range, PropList);
}

template <typename Name, typename T, bool HasIdentity, class BinaryOperation,
          int Dims, typename PropListTy = property_list,
          typename IdFilterFuncT = AllIdOp>
int testUSMInner(queue &Q, OptionalIdentity<T, HasIdentity> Identity, T Init,
                 BinaryOperation BOp, const range<Dims> &Range,
                 usm::alloc AllocType, PropListTy PropList = {},
                 IdFilterFuncT IdFilterFunc = {}) {
  printTestLabel<T, BinaryOperation>(Range);

  auto Dev = Q.get_device();
  if (!Dev.has(getUSMAspect(AllocType))) {
    std::cout << " SKIPPED due to unsupported USM alloc type" << std::endl;
    return 0;
  }

  // It is a known problem with passing data that is close to 4Gb in size
  // to device. Such data breaks the execution pretty badly.
  // Some of test cases calling this function try to verify the correctness
  // of reduction with the global range bigger than the maximal work-group size
  // for the device. Maximal WG size for device may be very big, e.g. it is
  // 67108864 for ACC emulator. Multiplying that by some factor
  // (to exceed max WG-Size) and multiplying it by the element size may exceed
  // the safe size of data passed to device.
  // Let's set it to 1 GB for now, and just skip the test if it exceeds 1Gb.
  constexpr size_t OneGB = 1LL * 1024 * 1024 * 1024;
  if (Range.size() * sizeof(T) > OneGB) {
    std::cout << " SKIPPED due to too big data size" << std::endl;
    return 0;
  }

  // TODO: Perhaps, this is a _temporary_ fix for CI. The test may run
  // for too long when the range is big. That is especially bad on ACC.
  if (Range.size() > 65536) {
    std::cout << " SKIPPED due to risk of timeout in CI" << std::endl;
    return 0;
  }

  T *ReduVarPtr = (T *)malloc(sizeof(T), Dev, Q.get_context(), AllocType);
  if (ReduVarPtr == nullptr) {
    std::cout << " SKIPPED due to unrelated reason: alloc returned nullptr"
              << std::endl;
    return 0;
  }
  if (AllocType == usm::alloc::device) {
    Q.submit([&](handler &CGH) {
       CGH.single_task<TName<Name, class InitKernel>>(
           [=]() { *ReduVarPtr = Init; });
     }).wait();
  } else {
    *ReduVarPtr = Init;
  }

  // Initialize.
  std::optional<T> CorrectOut;
  buffer<T, Dims> InBuf(Range);
  initInputData(InBuf, CorrectOut, BOp, Range, IdFilterFunc);
  if (!PropList.template has_property<
          property::reduction::initialize_to_identity>()) {
    CorrectOut = CorrectOut ? BOp(*CorrectOut, Init) : Init;
  }

  // Compute.
  Q.submit([&](handler &CGH) {
     // Helper for creating the reductions depending on the existance of an
     // identity.
     auto CreateReduction = [&]() {
       if constexpr (HasIdentity) {
         return reduction(ReduVarPtr, Identity.get(), BOp, PropList);
       } else {
         return reduction(ReduVarPtr, BOp, PropList);
       }
     };

     auto In = InBuf.template get_access<access::mode::read>(CGH);
     auto Redu = CreateReduction();
     CGH.parallel_for<TName<Name, class Test>>(
         Range, Redu, [=](item<Dims> Id, auto &Sum) {
           if (IdFilterFunc(Id.get_linear_id()))
             Sum.combine(In[Id]);
         });
   }).wait();

  // Check correctness.
  T ComputedOut;
  if (AllocType == usm::alloc::device) {
    buffer<T, 1> Buf(&ComputedOut, range<1>(1));
    Q.submit([&](handler &CGH) {
       auto OutAcc = Buf.template get_access<access::mode::discard_write>(CGH);
       CGH.single_task<TName<Name, class Check>>(
           [=]() { OutAcc[0] = *ReduVarPtr; });
     }).wait();
    ComputedOut = host_accessor(Buf, read_only)[0];
  } else {
    ComputedOut = *ReduVarPtr;
  }

  std::string AllocStr =
      "AllocMode=" + std::to_string(static_cast<int>(AllocType));
  int Error = checkResults(Q, BOp, Range, ComputedOut, *CorrectOut, AllocStr);
  free(ReduVarPtr, Q.get_context());
  return Error;
}

template <typename Name, typename T, class BinaryOperation, int Dims,
          typename PropListTy = property_list, typename IdFilterFuncT = AllIdOp>
int testUSM(queue &Q, T Identity, T Init, BinaryOperation BOp,
            const range<Dims> &Range, usm::alloc AllocType,
            property_list PropList = {}, IdFilterFuncT IdFilterFunc = {}) {
  return testUSMInner<Name>(Q, OptionalIdentity(Identity), Init, BOp, Range,
                            AllocType, PropList);
}

template <typename Name, typename T, class BinaryOperation, int Dims,
          typename PropListTy = property_list, typename IdFilterFuncT = AllIdOp>
int testUSM(queue &Q, T Init, BinaryOperation BOp, const range<Dims> &Range,
            usm::alloc AllocType, property_list PropList = {},
            IdFilterFuncT IdFilterFunc = {}) {
  return testUSMInner<Name>(Q, OptionalIdentity<T>(), Init, BOp, Range,
                            AllocType, PropList);
}
