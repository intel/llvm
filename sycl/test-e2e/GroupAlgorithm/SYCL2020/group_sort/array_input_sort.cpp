// REQUIRES: sg-8
// RUN: %{build} -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %{run} %t.out

// The test verifies sorting APIs for fixed-size array input from group_sort
// extension.
#include "common.hpp"
#include <sycl/ext/oneapi/experimental/group_sort.hpp>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

template <UseGroupT UseGroup, int Dims, size_t ElementsPerWorkItem,
          class Property = sycl::ext::oneapi::experimental::empty_properties_t,
          class T, class Compare>
void RunSortOverGroupArray(sycl::queue &Q, const std::vector<T> &DataToSort,
                           const Compare &Comp, Property Prop) {

  const size_t WorkSize = DataToSort.size() / ElementsPerWorkItem;
  const size_t NumSubGroups = WorkSize / ReqSubGroupSize + 1;

  const sycl::nd_range<Dims> NDRange = [&]() {
    if constexpr (Dims == 1)
      return sycl::nd_range<1>{{WorkSize}, {WorkSize}};
    else
      return sycl::nd_range<2>{{1, WorkSize}, {1, WorkSize}};
    static_assert(Dims < 3,
                  "Only one and two dimensional kernels are supported");
  }();

  using DefaultSorterT =
      oneapi_exp::default_sorters::group_sorter<T, Compare,
                                                ElementsPerWorkItem>;

  using RadixSorterT = oneapi_exp::radix_sorters::group_sorter<
      typename ConvertToSimpleType<T>::Type,
      ConvertToSortingOrder<Compare>::Type, ElementsPerWorkItem>;

  constexpr bool IsSG = (UseGroup == UseGroupT::SubGroup);
  auto Scope =
      IsSG ? sycl::memory_scope::sub_group : sycl::memory_scope::work_group;
  auto WGSize = NDRange.get_local_range().size();
  auto GroupSize = IsSG ? ReqSubGroupSize : WGSize;
  std::size_t LocalMemorySizeDefault =
      DefaultSorterT::memory_required(Scope, GroupSize);
  std::size_t LocalMemorySizeRadix =
      RadixSorterT::memory_required(Scope, GroupSize);
  std::array<std::vector<T>, 4> DataToSortCase = {DataToSort, DataToSort,
                                                  DataToSort, DataToSort};

  // Sort data using 3 different versions of sort_over_group API
  {
    std::array<std::shared_ptr<sycl::buffer<T>>, 4> BufToSort;
    for (int i = 0; i < 4; i++)
      BufToSort[i].reset(new sycl::buffer<T>(DataToSortCase[i].data(),
                                             DataToSortCase[i].size()));

    Q.submit([&](sycl::handler &CGH) {
       auto AccToSort0 = sycl::accessor(*BufToSort[0], CGH);
       auto AccToSort1 = sycl::accessor(*BufToSort[1], CGH);
       auto AccToSort2 = sycl::accessor(*BufToSort[2], CGH);
       auto AccToSort3 = sycl::accessor(*BufToSort[3], CGH);

       // Allocate local memory for all sub-groups in a work-group
       const size_t TotalLocalMemSizeDefault =
           IsSG ? LocalMemorySizeDefault * NumSubGroups
                : LocalMemorySizeDefault;
       sycl::local_accessor<std::byte, 1> ScratchDefault(
           {TotalLocalMemSizeDefault}, CGH);

       const size_t TotalLocalMemSizeRadix =
           IsSG ? LocalMemorySizeRadix * NumSubGroups : LocalMemorySizeRadix;

       sycl::local_accessor<std::byte, 1> ScratchRadix({TotalLocalMemSizeRadix},
                                                       CGH);

       CGH.parallel_for(
           NDRange, [=](sycl::nd_item<Dims> id) [[intel::reqd_sub_group_size(
                        ReqSubGroupSize)]] {
             const size_t GlobalLinearID = id.get_global_linear_id();
             using RadixSorterT = oneapi_exp::radix_sorters::group_sorter<
                 typename ConvertToSimpleType<T>::Type,
                 ConvertToSortingOrder<Compare>::Type, ElementsPerWorkItem>;

             auto Group = [&]() {
               if constexpr (IsSG)
                 return id.get_sub_group();
               else
                 return id.get_group();
             }();

             // Each sub-group should use its own part of the scratch pad
             const size_t ScratchShiftDefault =
                 IsSG ? id.get_sub_group().get_group_linear_id() *
                            LocalMemorySizeDefault
                      : 0;
             std::byte *ScratchPtrDefault =
                 &ScratchDefault[0] + ScratchShiftDefault;

             T ValsPrivate[ElementsPerWorkItem];

             auto ReadToPrivate = [&](auto Acc) {
               for (std::size_t I = 0; I < ElementsPerWorkItem; ++I)
                 ValsPrivate[I] = Acc[GlobalLinearID * ElementsPerWorkItem + I];
             };
             auto WriteToGlobal = [&](auto Acc) {
               for (std::size_t I = 0; I < ElementsPerWorkItem; ++I)
                 Acc[GlobalLinearID * ElementsPerWorkItem + I] = ValsPrivate[I];
             };

             auto Scratch =
                 sycl::span{ScratchPtrDefault, LocalMemorySizeDefault};
             auto PrivateArr = sycl::span<T, ElementsPerWorkItem>{
                 ValsPrivate, ValsPrivate + ElementsPerWorkItem};
             if constexpr (std::is_same_v<Compare, std::less<T>>) {
               ReadToPrivate(AccToSort0);
               oneapi_exp::sort_over_group(
                   oneapi_exp::group_with_scratchpad(Group, Scratch),
                   PrivateArr, Prop); // (4)
               WriteToGlobal(AccToSort0);
             }

             ReadToPrivate(AccToSort1);
             oneapi_exp::sort_over_group(
                 oneapi_exp::group_with_scratchpad(Group, Scratch), PrivateArr,
                 Comp, Prop); // (5)
             WriteToGlobal(AccToSort1);

             ReadToPrivate(AccToSort2);
             oneapi_exp::sort_over_group(Group, PrivateArr,
                                         DefaultSorterT(Scratch), Prop); // (6)
             WriteToGlobal(AccToSort2);

             // Each sub-group should use its own part of the scratch pad
             const size_t ScratchShiftRadix =
                 IsSG ? id.get_sub_group().get_group_linear_id() *
                            LocalMemorySizeRadix
                      : 0;
             std::byte *ScratchPtrRadix = &ScratchRadix[0] + ScratchShiftRadix;

             // Radix doesn't support custom types
             if constexpr (!std::is_same_v<CustomType, T>) {
               ReadToPrivate(AccToSort3);
               oneapi_exp::sort_over_group(
                   Group, PrivateArr,
                   RadixSorterT(
                       sycl::span{ScratchPtrRadix, LocalMemorySizeRadix}),
                   Prop); // (6)
               WriteToGlobal(AccToSort3);
             }
           });
     }).wait_and_throw();
  }

  // Verification
  {
    // Emulate independent sorting of each work-group/sub-group
    const size_t ChunkSize = GroupSize * ElementsPerWorkItem;
    std::vector<T> TempSorted = DataToSort;
    auto It = TempSorted.begin();
    for (; (It + ChunkSize) < TempSorted.end(); It += ChunkSize)
      std::sort(It, It + ChunkSize, Comp);

    // Sort reminder
    std::sort(It, TempSorted.end(), Comp);
    std::vector<T> DataSorted;
    DataSorted.resize(TempSorted.size());
    writeBlockedOrStriped<T>(/*In */ TempSorted, /* Out */ DataSorted,
                             GroupSize, ElementsPerWorkItem, Prop);

    if constexpr (std::is_same_v<Compare, std::less<T>>)
      assert(DataToSortCase[0] == DataSorted);

    assert(DataToSortCase[1] == DataSorted);
    assert(DataToSortCase[2] == DataSorted);
    // Radix doesn't support custom types
    if constexpr (!std::is_same_v<CustomType, T>)
      assert(DataToSortCase[3] == DataSorted);
  }
}

template <
    UseGroupT UseGroup, int Dim, size_t ElementsPerWorkItem, class T,
    typename Compare,
    typename Properties = sycl::ext::oneapi::experimental::empty_properties_t>
void RunOnData(sycl::queue &Q, const std::vector<T> &Data,
               const Compare &Comparator, Properties Prop = {}) {
  if constexpr (UseGroup == UseGroupT::SubGroup)
    if (Q.get_backend() == sycl::backend::ext_oneapi_cuda ||
        Q.get_backend() == sycl::backend::ext_oneapi_hip) {
      std::cout << "Note! Skipping sub group testing on CUDA BE" << std::endl;
      return;
    }

  RunSortOverGroupArray<UseGroup, Dim, ElementsPerWorkItem>(Q, Data, Comparator,
                                                            Prop);
};

template <class T> void RunOverType(sycl::queue &Q, size_t DataSize) {
  constexpr size_t PerWI = 4;
  std::vector<T> ArrayDataRandom(DataSize * PerWI);

  // Fill using random numbers
  std::default_random_engine generator;
  std::normal_distribution<float> distribution((10.0), (2.0));
  for (T &Elem : ArrayDataRandom)
    Elem = T(distribution(generator));

  auto blocked = oneapi_exp::properties{oneapi_exp::input_data_placement<
      oneapi_exp::group_algorithm_data_placement::blocked>};
  auto striped = oneapi_exp::properties{oneapi_exp::input_data_placement<
      oneapi_exp::group_algorithm_data_placement::striped>};
  RunOnData<UseGroupT::WorkGroup, 1, PerWI>(Q, ArrayDataRandom, std::less<T>{},
                                            blocked);
  RunOnData<UseGroupT::WorkGroup, 1, PerWI>(Q, ArrayDataRandom,
                                            std::greater<T>{}, striped);
  RunOnData<UseGroupT::WorkGroup, 2, PerWI>(Q, ArrayDataRandom, std::less<T>{},
                                            blocked);
  RunOnData<UseGroupT::WorkGroup, 2, PerWI>(Q, ArrayDataRandom,
                                            std::greater<T>{}, striped);
  RunOnData<UseGroupT::SubGroup, 1, PerWI>(Q, ArrayDataRandom,
                                           std::greater<T>{}, blocked);
  RunOnData<UseGroupT::SubGroup, 1, PerWI>(Q, ArrayDataRandom, std::less<T>{},
                                           striped);
  RunOnData<UseGroupT::SubGroup, 2, PerWI>(Q, ArrayDataRandom,
                                           std::greater<T>{}, blocked);
  RunOnData<UseGroupT::SubGroup, 2, PerWI>(Q, ArrayDataRandom, std::less<T>{},
                                           striped);
}

int main() {
  try {
    sycl::queue Q;

    static constexpr size_t Size = 18;

    RunOverType<std::int32_t>(Q, Size);
    RunOverType<CustomType>(Q, Size);

    std::cout << "Test passed." << std::endl;
    return 0;
  } catch (std::exception &E) {
    std::cout << "Test failed" << std::endl;
    std::cout << E.what() << std::endl;
    return 1;
  }
}
