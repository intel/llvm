// REQUIRES: sg-8
// RUN: %{build} -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %{run} %t.out
// UNSUPPORTED: accelerator

// The test verifies sort API extension.
// Currently it checks the following combinations:
// For number of elements {18, 64}
//   For types {int, char, half, double, Custom}
//     For initial elements values {reversed, random}
//       For comparators {std::less, std::greater}
//         For dimensions {1, 2}
//           For group {work-group, sub-group}
//             For sorters {default_sorter, radix_sorter}
//               joint_sort with
//                   WG size = {16} or {1, 16}
//                   SG size = {8}
//                   elements per WI = 2
//               sort_over_group with
//                   WG size = {number_of_elements} or {1, number_of_elements}
//                   SG size = 8
//                   elements per WI = 1
//
// TODO: Test global memory for temporary storage
// TODO: Consider using USM instead of buffers
// TODO: Add support for sorting over workgroup for CUDA and HIP BE

#include <sycl/detail/core.hpp>

#include "common.hpp"
#include <sycl/builtins.hpp>
#include <sycl/ext/oneapi/experimental/group_sort.hpp>
#include <sycl/group_algorithm.hpp>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

template <UseGroupT UseGroup, int Dims, class T, class Compare>
void RunJointSort(sycl::queue &Q, const std::vector<T> &DataToSort,
                  const Compare &Comp) {

  constexpr size_t WGSize = 16;
  constexpr size_t ElemsPerWI = 2;
  const size_t NumOfElements = DataToSort.size();
  const size_t NumWGs = ((NumOfElements - 1) / WGSize * ElemsPerWI) + 1;

  constexpr size_t NumSubGroups = WGSize / ReqSubGroupSize;

  using RadixSorterT = oneapi_exp::radix_sorters::joint_sorter<
      typename ConvertToSimpleType<T>::Type,
      ConvertToSortingOrder<Compare>::Type>;

  std::size_t LocalMemorySizeDefault = 0;
  std::size_t LocalMemorySizeRadix = 0;
  if (UseGroup == UseGroupT::SubGroup) {
    // Each sub-group needs a piece of memory for sorting
    LocalMemorySizeDefault = oneapi_exp::default_sorters::joint_sorter<
        Compare>::template memory_required<T>(sycl::memory_scope::sub_group,
                                              ReqSubGroupSize * ElemsPerWI);
    LocalMemorySizeRadix = RadixSorterT::memory_required(
        sycl::memory_scope::sub_group, ReqSubGroupSize * ElemsPerWI);
  } else {
    // A single chunk of memory for each work-group
    LocalMemorySizeDefault = oneapi_exp::default_sorters::joint_sorter<
        Compare>::template memory_required<T>(sycl::memory_scope::work_group,
                                              WGSize * ElemsPerWI);
    LocalMemorySizeRadix = RadixSorterT::memory_required(
        sycl::memory_scope::sub_group, WGSize * ElemsPerWI);
  }

  const sycl::nd_range<Dims> NDRange = [&]() {
    if constexpr (Dims == 1)
      return sycl::nd_range<1>{{WGSize * NumWGs}, {WGSize}};
    else
      return sycl::nd_range<2>{{1, WGSize * NumWGs}, {1, WGSize}};
    static_assert(Dims < 3,
                  "Only one and two dimensional kernels are supported");
  }();

  std::vector<T> DataToSortCase0 = DataToSort;
  std::vector<T> DataToSortCase1 = DataToSort;
  std::vector<T> DataToSortCase2 = DataToSort;
  std::vector<T> DataToSortCase3 = DataToSort;

  // Sort data using 3 different versions of joint_sort API
  {
    sycl::buffer<T> BufToSort0(DataToSortCase0.data(), DataToSortCase0.size());
    sycl::buffer<T> BufToSort1(DataToSortCase1.data(), DataToSortCase1.size());
    sycl::buffer<T> BufToSort2(DataToSortCase2.data(), DataToSortCase2.size());
    sycl::buffer<T> BufToSort3(DataToSortCase3.data(), DataToSortCase3.size());

    Q.submit([&](sycl::handler &CGH) {
       auto AccToSort0 = sycl::accessor(BufToSort0, CGH);
       auto AccToSort1 = sycl::accessor(BufToSort1, CGH);
       auto AccToSort2 = sycl::accessor(BufToSort2, CGH);
       auto AccToSort3 = sycl::accessor(BufToSort3, CGH);

       // Allocate local memory for all sub-groups in a work-group
       const size_t TotalLocalMemSizeDefault =
           UseGroup == UseGroupT::SubGroup
               ? LocalMemorySizeDefault * NumSubGroups
               : LocalMemorySizeDefault;

       const size_t TotalLocalMemSizeRadix =
           UseGroup == UseGroupT::SubGroup ? LocalMemorySizeRadix * NumSubGroups
                                           : LocalMemorySizeRadix;

       sycl::local_accessor<std::byte, 1> ScratchDefault(
           {TotalLocalMemSizeDefault}, CGH);

       sycl::local_accessor<std::byte, 1> ScratchRadix({TotalLocalMemSizeRadix},
                                                       CGH);

       CGH.parallel_for<KernelNameJoint<IntWrapper<Dims>,
                                        UseGroupWrapper<UseGroup>, T, Compare>>(
           NDRange, [=](sycl::nd_item<Dims> ID) [[intel::reqd_sub_group_size(
                        ReqSubGroupSize)]] {
             auto Group = [&]() {
               if constexpr (UseGroup == UseGroupT::SubGroup)
                 return ID.get_sub_group();
               else
                 return ID.get_group();
             }();

             const size_t WGID = ID.get_group_linear_id();
             const size_t ChunkSize =
                 Group.get_max_local_range().size() * ElemsPerWI;
             const size_t PartID = UseGroup == UseGroupT::SubGroup
                                       ? WGID * Group.get_group_linear_range() +
                                             Group.get_group_linear_id()
                                       : WGID;
             const size_t LocalPartID =
                 UseGroup == UseGroupT::SubGroup
                     ? LocalMemorySizeDefault * Group.get_group_linear_id()
                     : 0;

             const size_t StartIdx = ChunkSize * PartID;
             const size_t EndIdx =
                 std::min(ChunkSize * (PartID + 1), NumOfElements);

             if (EndIdx <= StartIdx)
               return;

             // This version of API always sorts in ascending order
             if constexpr (std::is_same_v<Compare, std::less<T>>)
               oneapi_exp::joint_sort(
                   oneapi_exp::group_with_scratchpad(
                       Group, sycl::span{&ScratchDefault[LocalPartID],
                                         LocalMemorySizeDefault}),
                   &AccToSort0[StartIdx], &AccToSort0[EndIdx]);

             oneapi_exp::joint_sort(
                 oneapi_exp::group_with_scratchpad(
                     Group, sycl::span{&ScratchDefault[LocalPartID],
                                       LocalMemorySizeDefault}),
                 &AccToSort1[StartIdx], &AccToSort1[EndIdx], Comp);

             oneapi_exp::joint_sort(
                 Group, &AccToSort2[StartIdx], &AccToSort2[EndIdx],
                 oneapi_exp::default_sorters::joint_sorter<Compare>(sycl::span{
                     &ScratchDefault[LocalPartID], LocalMemorySizeDefault}));

             const size_t LocalPartIDRadix =
                 UseGroup == UseGroupT::SubGroup
                     ? LocalMemorySizeRadix * Group.get_group_linear_id()
                     : 0;

             // Radix doesn't support custom types
             if constexpr (!std::is_same_v<CustomType, T>)
               oneapi_exp::joint_sort(
                   Group, &AccToSort3[StartIdx], &AccToSort3[EndIdx],
                   RadixSorterT(sycl::span{&ScratchRadix[LocalPartIDRadix],
                                           LocalMemorySizeRadix}));
           });
     }).wait_and_throw();
  }

  // Verification
  {
    // Emulate independent sorting of each work-group and/or sub-group
    const size_t ChunkSize = UseGroup == UseGroupT::SubGroup
                                 ? ReqSubGroupSize * ElemsPerWI
                                 : WGSize * ElemsPerWI;
    std::vector<T> DataSorted = DataToSort;
    auto It = DataSorted.begin();
    for (; (It + ChunkSize) < DataSorted.end(); It += ChunkSize)
      std::sort(It, It + ChunkSize, Comp);

    // Sort reminder
    std::sort(It, DataSorted.end(), Comp);

    // This version of API always sorts in ascending order
    if constexpr (std::is_same_v<Compare, std::less<T>>)
      assert(DataToSortCase0 == DataSorted);

    assert(DataToSortCase1 == DataSorted);
    assert(DataToSortCase2 == DataSorted);
    // Radix doesn't support custom types
    if constexpr (!std::is_same_v<CustomType, T>)
      assert(DataToSortCase3 == DataSorted);
  }
}

template <UseGroupT UseGroup, int Dims, class T, class Compare>
void RunSortOVerGroup(sycl::queue &Q, const std::vector<T> &DataToSort,
                      const Compare &Comp) {

  const size_t NumOfElements = DataToSort.size();
  const size_t NumSubGroups = NumOfElements / ReqSubGroupSize + 1;

  const sycl::nd_range<Dims> NDRange = [&]() {
    if constexpr (Dims == 1)
      return sycl::nd_range<1>{{NumOfElements}, {NumOfElements}};
    else
      return sycl::nd_range<2>{{1, NumOfElements}, {1, NumOfElements}};
    static_assert(Dims < 3,
                  "Only one and two dimensional kernels are supported");
  }();

#if VERSION == 1
  using RadixSorterT = typename RadixSorterType<Compare, T>::Type;
#else
  using RadixSorterT = oneapi_exp::radix_sorters::group_sorter<
      typename ConvertToSimpleType<T>::Type,
      ConvertToSortingOrder<Compare>::Type>;
#endif

  std::size_t LocalMemorySizeDefault = 0;
  std::size_t LocalMemorySizeRadix = 0;
  if (UseGroup == UseGroupT::SubGroup) {
    // Each sub-group needs a piece of memory for sorting
    LocalMemorySizeDefault = oneapi_exp::default_sorters::group_sorter<
        T, Compare, 1>::memory_required(sycl::memory_scope::sub_group,
                                        ReqSubGroupSize);

    LocalMemorySizeRadix = RadixSorterT::memory_required(
        sycl::memory_scope::sub_group, ReqSubGroupSize);
  } else {
    // A single chunk of memory for each work-group
    LocalMemorySizeDefault = oneapi_exp::default_sorters::group_sorter<
        T, Compare, 1>::memory_required(sycl::memory_scope::work_group,
                                        NumOfElements);

    LocalMemorySizeRadix = RadixSorterT::memory_required(
        sycl::memory_scope::work_group, NumOfElements);
  }

  std::vector<T> DataToSortCase0 = DataToSort;
  std::vector<T> DataToSortCase1 = DataToSort;
  std::vector<T> DataToSortCase2 = DataToSort;
  std::vector<T> DataToSortCase3 = DataToSort;

  // Sort data using 3 different versions of sort_over_group API
  {
    sycl::buffer<T> BufToSort0(DataToSortCase0.data(), DataToSortCase0.size());
    sycl::buffer<T> BufToSort1(DataToSortCase1.data(), DataToSortCase1.size());
    sycl::buffer<T> BufToSort2(DataToSortCase2.data(), DataToSortCase2.size());
    sycl::buffer<T> BufToSort3(DataToSortCase3.data(), DataToSortCase3.size());

    Q.submit([&](sycl::handler &CGH) {
       auto AccToSort0 = sycl::accessor(BufToSort0, CGH);
       auto AccToSort1 = sycl::accessor(BufToSort1, CGH);
       auto AccToSort2 = sycl::accessor(BufToSort2, CGH);
       auto AccToSort3 = sycl::accessor(BufToSort3, CGH);

       // Allocate local memory for all sub-groups in a work-group
       const size_t TotalLocalMemSizeDefault =
           UseGroup == UseGroupT::SubGroup
               ? LocalMemorySizeDefault * NumSubGroups
               : LocalMemorySizeDefault;
       sycl::local_accessor<std::byte, 1> ScratchDefault(
           {TotalLocalMemSizeDefault}, CGH);

       const size_t TotalLocalMemSizeRadix =
           UseGroup == UseGroupT::SubGroup ? LocalMemorySizeRadix * NumSubGroups
                                           : LocalMemorySizeRadix;

       sycl::local_accessor<std::byte, 1> ScratchRadix({TotalLocalMemSizeRadix},
                                                       CGH);

       CGH.parallel_for<KernelNameOverGroup<
           IntWrapper<Dims>, UseGroupWrapper<UseGroup>, T, Compare>>(
           NDRange, [=](sycl::nd_item<Dims> id) [[intel::reqd_sub_group_size(
                        ReqSubGroupSize)]] {
             const size_t GlobalLinearID = id.get_global_linear_id();

             auto Group = [&]() {
               if constexpr (UseGroup == UseGroupT::SubGroup)
                 return id.get_sub_group();
               else
                 return id.get_group();
             }();

             // Each sub-group should use it's own part of the scratch pad
             const size_t ScratchShiftDefault =
                 UseGroup == UseGroupT::SubGroup
                     ? id.get_sub_group().get_group_linear_id() *
                           LocalMemorySizeDefault
                     : 0;
             std::byte *ScratchPtrDefault =
                 &ScratchDefault[0] + ScratchShiftDefault;

             if constexpr (std::is_same_v<Compare, std::less<T>>)
               AccToSort0[GlobalLinearID] = oneapi_exp::sort_over_group(
                   oneapi_exp::group_with_scratchpad(
                       Group,
                       sycl::span{ScratchPtrDefault, LocalMemorySizeDefault}),
                   AccToSort0[GlobalLinearID]);

             AccToSort1[GlobalLinearID] = oneapi_exp::sort_over_group(
                 oneapi_exp::group_with_scratchpad(
                     Group,
                     sycl::span{ScratchPtrDefault, LocalMemorySizeDefault}),
                 AccToSort1[GlobalLinearID], Comp);

             AccToSort2[GlobalLinearID] = oneapi_exp::sort_over_group(
                 Group, AccToSort2[GlobalLinearID],
                 oneapi_exp::default_sorters::group_sorter<T, Compare, 1>(
                     sycl::span{ScratchPtrDefault, LocalMemorySizeDefault}));

             // Each sub-group should use it's own part of the scratch pad
             const size_t ScratchShiftRadix =
                 UseGroup == UseGroupT::SubGroup
                     ? id.get_sub_group().get_group_linear_id() *
                           LocalMemorySizeRadix
                     : 0;
             std::byte *ScratchPtrRadix = &ScratchRadix[0] + ScratchShiftRadix;

             // Radix doesn't support custom types
             if constexpr (!std::is_same_v<CustomType, T>)
               AccToSort3[GlobalLinearID] = oneapi_exp::sort_over_group(
                   Group, AccToSort3[GlobalLinearID],
                   RadixSorterT(
                       sycl::span{ScratchPtrRadix, LocalMemorySizeRadix}));
           });
     }).wait_and_throw();
  }

  // Verification
  {
    // Emulate independent sorting of each work-group/sub-group
    const size_t ChunkSize = UseGroup == UseGroupT::SubGroup
                                 ? ReqSubGroupSize
                                 : NDRange.get_local_range().size();
    std::vector<T> DataSorted = DataToSort;
    auto It = DataSorted.begin();
    for (; (It + ChunkSize) < DataSorted.end(); It += ChunkSize)
      std::sort(It, It + ChunkSize, Comp);

    // Sort reminder
    std::sort(It, DataSorted.end(), Comp);

    if constexpr (std::is_same_v<Compare, std::less<T>>)
      assert(DataToSortCase0 == DataSorted);

    assert(DataToSortCase1 == DataSorted);
    assert(DataToSortCase2 == DataSorted);
    // Radix doesn't support custom types
    if constexpr (!std::is_same_v<CustomType, T>)
      assert(DataToSortCase3 == DataSorted);
  }
}

template <class T> void RunOverType(sycl::queue &Q, size_t DataSize) {
  std::vector<T> DataReversed(DataSize);
  std::vector<T> DataRandom(DataSize);

  std::iota(DataReversed.rbegin(), DataReversed.rend(), (size_t)0);

  // Fill using random numbers
  {
    std::default_random_engine generator;
    std::normal_distribution<float> distribution((10.0), (2.0));
    for (T &Elem : DataRandom)
      Elem = T(distribution(generator));
  }

  auto RunOnDataAndComp = [&](const std::vector<T> &Data,
                              const auto &Comparator) {
    RunSortOVerGroup<UseGroupT::WorkGroup, 1>(Q, Data, Comparator);
    RunSortOVerGroup<UseGroupT::WorkGroup, 2>(Q, Data, Comparator);

    RunJointSort<UseGroupT::WorkGroup, 1>(Q, Data, Comparator);
    RunJointSort<UseGroupT::WorkGroup, 2>(Q, Data, Comparator);

    if (Q.get_backend() == sycl::backend::ext_oneapi_cuda ||
        Q.get_backend() == sycl::backend::ext_oneapi_hip) {
      std::cout << "Note! Skipping sub group testing on CUDA BE" << std::endl;
      return;
    }

    RunSortOVerGroup<UseGroupT::SubGroup, 1>(Q, Data, Comparator);
    RunSortOVerGroup<UseGroupT::SubGroup, 2>(Q, Data, Comparator);

    RunJointSort<UseGroupT::SubGroup, 1>(Q, Data, Comparator);
    RunJointSort<UseGroupT::SubGroup, 2>(Q, Data, Comparator);
  };

  RunOnDataAndComp(DataReversed, std::greater<T>{});
  RunOnDataAndComp(DataReversed, std::less<T>{});
  RunOnDataAndComp(DataRandom, std::less<T>{});
  RunOnDataAndComp(DataRandom, std::greater<T>{});
}

int main() {
  try {
    sycl::queue Q;

    std::vector<size_t> Sizes{18, 64};

    for (size_t Size : Sizes) {
      RunOverType<std::int32_t>(Q, Size);
      RunOverType<char>(Q, Size);
      if (Q.get_device().has(sycl::aspect::fp16))
        RunOverType<sycl::half>(Q, Size);
      if (Q.get_device().has(sycl::aspect::fp64))
        RunOverType<double>(Q, Size);
      RunOverType<CustomType>(Q, Size);
    }

    std::cout << "Test passed." << std::endl;
    return 0;
  } catch (std::exception &E) {
    std::cout << "Test failed" << std::endl;
    std::cout << E.what() << std::endl;
    return 1;
  }
}
