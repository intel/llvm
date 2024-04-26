// RUN: %{build} -std=c++17 -o %t.run -Wno-deprecated-declarations
// RUN: %{run} %t.run

// Variant of group_asymc_copy.cpp using legacy multi_ptr and the corresponding
// deprecated async_work_group_copy interfaces.

#include <iostream>
#include <sycl/detail/core.hpp>
#include <typeinfo>

using namespace sycl;

template <typename T> class TypeHelper;

template <typename T>
using KernelName = class TypeHelper<typename std::conditional<
    std::is_same<T, std::byte>::value, unsigned char, T>::type>;

// Define the number of work items to enqueue.
const size_t NElems = 32;
const size_t WorkGroupSize = 8;
const size_t NWorkGroups = NElems / WorkGroupSize;

template <typename T> void initInputBuffer(buffer<T, 1> &Buf, size_t Stride) {
  host_accessor Acc(Buf, sycl::write_only);
  for (size_t I = 0; I < Buf.size(); I += WorkGroupSize) {
    for (size_t J = 0; J < WorkGroupSize; J++)
      Acc[I + J] = static_cast<T>(I + J + ((J % Stride == 0) ? 100 : 0));
  }
}

template <typename T> void initOutputBuffer(buffer<T, 1> &Buf) {
  host_accessor Acc(Buf, sycl::write_only);
  for (size_t I = 0; I < Buf.size(); I++)
    Acc[I] = static_cast<T>(0);
}

template <typename T> struct is_vec : std::false_type {};
template <typename T, size_t N> struct is_vec<vec<T, N>> : std::true_type {};

template <typename T> bool checkEqual(vec<T, 1> A, size_t B) {
  T TB = B;
  return A.s0() == TB;
}

template <typename T> bool checkEqual(vec<T, 4> A, size_t B) {
  T TB = B;
  return A.x() == TB && A.y() == TB && A.z() == TB && A.w() == TB;
}

template <typename T>
typename std::enable_if_t<!is_vec<T>::value, bool> checkEqual(T A, size_t B) {
  T TB = static_cast<T>(B);
  return A == TB;
}

template <typename T> std::string toString(vec<T, 1> A) {
  std::string R("(");
  return R + std::to_string(A.s0()) + ")";
}

template <typename T> std::string toString(vec<T, 4> A) {
  std::string R("(");
  R += std::to_string(A.x()) + "," + std::to_string(A.y()) + "," +
       std::to_string(A.z()) + "," + std::to_string(A.w()) + ")";
  return R;
}

template <typename T = void>
typename std::enable_if_t<
    !is_vec<T>::value && std::is_same<T, std::byte>::value, std::string>
toString(T A) {
  return std::to_string((unsigned char)A);
}

template <typename T = void>
typename std::enable_if_t<
    !is_vec<T>::value && !std::is_same<T, std::byte>::value, std::string>
toString(T A) {
  return std::to_string(A);
}

template <typename T> int checkResults(buffer<T, 1> &OutBuf, size_t Stride) {
  host_accessor Out(OutBuf, sycl::read_only);
  int EarlyFailout = 20;

  for (size_t I = 0; I < OutBuf.size(); I += WorkGroupSize) {
    for (size_t J = 0; J < WorkGroupSize; J++) {
      size_t ExpectedVal = (J % Stride == 0) ? (100 + I + J) : 0;
      if (!checkEqual(Out[I + J], ExpectedVal)) {
        std::cerr << std::string(typeid(T).name()) + ": Stride=" << Stride
                  << " : Incorrect value at index " << I + J
                  << " : Expected: " << toString(ExpectedVal)
                  << ", Computed: " << toString(Out[I + J]) << "\n";
        if (--EarlyFailout == 0)
          return 1;
      }
    }
  }
  return EarlyFailout - 20;
}

template <typename T> int test(size_t Stride) {
  queue Q;

  buffer<T, 1> InBuf(NElems);
  buffer<T, 1> OutBuf(NElems);

  initInputBuffer(InBuf, Stride);
  initOutputBuffer(OutBuf);

  Q.submit([&](handler &CGH) {
     // In must be read_write for legacy multi_ptr as it would otherwise produce
     // a const multi_ptr which was not a valid source argument to
     // async_work_group_copy in SYCL 1.2.1.
     auto In = InBuf.template get_access<access::mode::read_write>(CGH);
     auto Out = OutBuf.template get_access<access::mode::write>(CGH);
     local_accessor<T, 1> Local(range<1>{WorkGroupSize}, CGH);

     nd_range<1> NDR{range<1>(NElems), range<1>(WorkGroupSize)};
     CGH.parallel_for<KernelName<T>>(NDR, [=](nd_item<1> NDId) {
       auto GrId = NDId.get_group_linear_id();
       auto Group = NDId.get_group();
       size_t NElemsToCopy =
           WorkGroupSize / Stride + ((WorkGroupSize % Stride) ? 1 : 0);
       size_t Offset = GrId * WorkGroupSize;
       if (Stride == 1) { // Check the version without stride arg.
         auto E = NDId.async_work_group_copy(
             Local.template get_multi_ptr<access::decorated::legacy>(),
             In.template get_multi_ptr<access::decorated::legacy>() + Offset,
             NElemsToCopy);
         E.wait();
       } else {
         auto E = NDId.async_work_group_copy(
             Local.template get_multi_ptr<access::decorated::legacy>(),
             In.template get_multi_ptr<access::decorated::legacy>() + Offset,
             NElemsToCopy, Stride);
         E.wait();
       }

       if (Stride == 1) { // Check the version without stride arg.
         auto E = Group.async_work_group_copy(
             Out.template get_multi_ptr<access::decorated::legacy>() + Offset,
             Local.template get_multi_ptr<access::decorated::legacy>(),
             NElemsToCopy);
         Group.wait_for(E);
       } else {
         auto E = Group.async_work_group_copy(
             Out.template get_multi_ptr<access::decorated::legacy>() + Offset,
             Local.template get_multi_ptr<access::decorated::legacy>(),
             NElemsToCopy, Stride);
         Group.wait_for(E);
       }
     });
   }).wait();

  return checkResults(OutBuf, Stride);
}

int main() {
  for (int Stride = 1; Stride < WorkGroupSize; Stride++) {
    if (test<int>(Stride))
      return 1;
    if (test<vec<int, 1>>(Stride))
      return 1;
    if (test<int4>(Stride))
      return 1;
    if (test<bool>(Stride))
      return 1;
    if (test<vec<bool, 1>>(Stride))
      return 1;
    if (test<vec<bool, 4>>(Stride))
      return 1;
    if (test<sycl::opencl::cl_bool>(Stride))
      return 1;
    if (test<std::byte>(Stride))
      return 1;
  }

  std::cout << "Test passed.\n";
  return 0;
}
