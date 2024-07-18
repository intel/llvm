// REQUIRES: sg-8
// RUN: %{build} -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %{run} %t.out

// The test verifies key/value sorting APIs for fixed-size array input from
// group_sort extension.
#include "common.hpp"
#include <sycl/ext/oneapi/experimental/group_sort.hpp>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

template <UseGroupT UseGroup, int Dims, size_t ElementsPerWorkItem,
          class Property = sycl::ext::oneapi::experimental::empty_properties_t,
          class KeyTy, class ValueTy, class Compare>
void RunSortKeyValueOverGroupArray(sycl::queue &Q,
                                   const std::vector<KeyTy> &KeysToSort,
                                   const std::vector<ValueTy> &DataToSort,
                                   const Compare &Comp, Property Prop) {
  const size_t WorkSize = KeysToSort.size() / ElementsPerWorkItem;
  const size_t NumSubGroups = WorkSize / ReqSubGroupSize + 1;

  const sycl::nd_range<Dims> NDRange = [&]() {
    if constexpr (Dims == 1)
      return sycl::nd_range<1>{{WorkSize}, {WorkSize}};
    else
      return sycl::nd_range<2>{{1, WorkSize}, {1, WorkSize}};
    static_assert(Dims < 3,
                  "Only one and two dimensional kernels are supported");
  }();

  using DefaultSorterT = oneapi_exp::default_sorters::group_key_value_sorter<
      KeyTy, ValueTy, Compare, ElementsPerWorkItem>;

  using RadixSorterT = oneapi_exp::radix_sorters::group_key_value_sorter<
      typename ConvertToSimpleType<KeyTy>::Type, ValueTy,
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

  std::array<std::vector<KeyTy>, 4> KeysToSortCase = {KeysToSort, KeysToSort,
                                                      KeysToSort, KeysToSort};
  std::array<std::vector<ValueTy>, 4> DataToSortCase = {DataToSort, DataToSort,
                                                        DataToSort, DataToSort};

  // Sort data using 3 different versions of sort_over_group API
  {
    std::array<std::shared_ptr<sycl::buffer<KeyTy>>, 4> BufKeysToSort;
    std::array<std::shared_ptr<sycl::buffer<ValueTy>>, 4> BufDataToSort;
    for (int i = 0; i < 4; i++) {
      BufKeysToSort[i].reset(new sycl::buffer<KeyTy>(KeysToSortCase[i].data(),
                                                     KeysToSortCase[i].size()));
      BufDataToSort[i].reset(new sycl::buffer<ValueTy>(
          DataToSortCase[i].data(), DataToSortCase[i].size()));
    }

    Q.submit([&](sycl::handler &CGH) {
       auto AccDataToSort0 = sycl::accessor(*BufDataToSort[0], CGH);
       auto AccKeysToSort0 = sycl::accessor(*BufKeysToSort[0], CGH);

       auto AccDataToSort1 = sycl::accessor(*BufDataToSort[1], CGH);
       auto AccKeysToSort1 = sycl::accessor(*BufKeysToSort[1], CGH);

       auto AccDataToSort2 = sycl::accessor(*BufDataToSort[2], CGH);
       auto AccKeysToSort2 = sycl::accessor(*BufKeysToSort[2], CGH);

       auto AccDataToSort3 = sycl::accessor(*BufDataToSort[3], CGH);
       auto AccKeysToSort3 = sycl::accessor(*BufKeysToSort[3], CGH);

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

       CGH.parallel_for(NDRange, [=](sycl::nd_item<Dims>
                                         id) [[intel::reqd_sub_group_size(
                                     ReqSubGroupSize)]] {
         const size_t GlobalLinearID = id.get_global_linear_id();

         auto Group = [&]() {
           if constexpr (UseGroup == UseGroupT::SubGroup)
             return id.get_sub_group();
           else
             return id.get_group();
         }();

         // Each sub-group should use its own part of the scratch pad
         const size_t ScratchShiftDefault =
             UseGroup == UseGroupT::SubGroup
                 ? id.get_sub_group().get_group_linear_id() *
                       LocalMemorySizeDefault
                 : 0;
         std::byte *ScratchPtrDefault =
             &ScratchDefault[0] + ScratchShiftDefault;

         KeyTy KeysPrivate[ElementsPerWorkItem];
         ValueTy ValsPrivate[ElementsPerWorkItem];

         auto ReadToPrivate = [&](auto &KeyAcc, auto &ValAcc) {
           for (std::size_t I = 0; I < ElementsPerWorkItem; ++I) {
             KeysPrivate[I] = KeyAcc[GlobalLinearID * ElementsPerWorkItem + I];
             ValsPrivate[I] = ValAcc[GlobalLinearID * ElementsPerWorkItem + I];
           }
         };
         auto WriteToGlobal = [&](auto &KeyAcc, auto &ValAcc) {
           for (std::size_t I = 0; I < ElementsPerWorkItem; ++I) {
             KeyAcc[GlobalLinearID * ElementsPerWorkItem + I] = KeysPrivate[I];
             ValAcc[GlobalLinearID * ElementsPerWorkItem + I] = ValsPrivate[I];
           }
         };
         auto Scratch = sycl::span{ScratchPtrDefault, LocalMemorySizeDefault};
         auto KeysPrivateArr = sycl::span<KeyTy, ElementsPerWorkItem>{
             KeysPrivate, KeysPrivate + ElementsPerWorkItem};
         auto ValsPrivateArr = sycl::span<ValueTy, ElementsPerWorkItem>{
             ValsPrivate, ValsPrivate + ElementsPerWorkItem};
         if constexpr (std::is_same_v<Compare, std::less<KeyTy>>) {
           ReadToPrivate(AccKeysToSort0, AccDataToSort0);
           oneapi_exp::sort_key_value_over_group(
               oneapi_exp::group_with_scratchpad(Group, Scratch),
               KeysPrivateArr, ValsPrivateArr, Prop);
           WriteToGlobal(AccKeysToSort0, AccDataToSort0);
         }

         ReadToPrivate(AccKeysToSort1, AccDataToSort1);
         oneapi_exp::sort_key_value_over_group(
             oneapi_exp::group_with_scratchpad(Group, Scratch), KeysPrivateArr,
             ValsPrivateArr, Comp, Prop);
         WriteToGlobal(AccKeysToSort1, AccDataToSort1);

         ReadToPrivate(AccKeysToSort2, AccDataToSort2);
         oneapi_exp::sort_key_value_over_group(Group, KeysPrivateArr,
                                               ValsPrivateArr,
                                               DefaultSorterT(Scratch), Prop);
         WriteToGlobal(AccKeysToSort2, AccDataToSort2);

         // Each sub-group should use its own part of the scratch pad
         const size_t ScratchShiftRadix =
             UseGroup == UseGroupT::SubGroup
                 ? id.get_sub_group().get_group_linear_id() *
                       LocalMemorySizeRadix
                 : 0;
         std::byte *ScratchPtrRadix = &ScratchRadix[0] + ScratchShiftRadix;

         // Radix doesn't support custom types
         if constexpr (!std::is_same_v<CustomType, KeyTy>) {
           ReadToPrivate(AccKeysToSort3, AccDataToSort3);
           oneapi_exp::sort_key_value_over_group(
               Group, KeysPrivateArr, ValsPrivateArr,
               RadixSorterT(sycl::span{ScratchPtrRadix, LocalMemorySizeRadix}),
               Prop);
           WriteToGlobal(AccKeysToSort3, AccDataToSort3);
         }
       });
     }).wait_and_throw();
  }

  // Verification
  {

    std::vector<std::pair<KeyTy, ValueTy>> KeyDataToSort;
    KeyDataToSort.reserve(KeysToSort.size());
    std::transform(
        KeysToSort.begin(), KeysToSort.end(), DataToSort.begin(),
        std::back_inserter(KeyDataToSort),
        [](KeyTy Key, ValueTy Value) { return std::make_pair(Key, Value); });
    // Emulate independent sorting of each work-group/sub-group
    auto GroupSize = UseGroup == UseGroupT::SubGroup
                         ? ReqSubGroupSize
                         : NDRange.get_local_range().size();
    const size_t ChunkSize = GroupSize * ElementsPerWorkItem;
    auto Temp1 = KeyDataToSort;
    ReadWriteBlockedOrStriped(Temp1, KeyDataToSort, GroupSize,
                              ElementsPerWorkItem, Prop, /* Read */ true);
    auto It = KeyDataToSort.begin();
    auto KeyValueComp = [&](const std::pair<KeyTy, ValueTy> &A,
                            const std::pair<KeyTy, ValueTy> &B) -> bool {
      return Comp(A.first, B.first);
    };
    for (; (It + ChunkSize) < KeyDataToSort.end(); It += ChunkSize)
      std::stable_sort(It, It + ChunkSize, KeyValueComp);

    // Sort remainder
    std::stable_sort(It, KeyDataToSort.end(), KeyValueComp);

    auto Temp2 = KeyDataToSort;
    ReadWriteBlockedOrStriped(Temp2, KeyDataToSort, GroupSize,
                              ElementsPerWorkItem, Prop, /* Read */ false);

    std::vector<KeyTy> KeysSorted;
    std::vector<ValueTy> DataSorted;
    KeysSorted.reserve(KeyDataToSort.size());
    DataSorted.reserve(KeyDataToSort.size());
    std::transform(KeyDataToSort.begin(), KeyDataToSort.end(),
                   std::back_inserter(KeysSorted),
                   [](const std::pair<KeyTy, ValueTy> &KeyValue) {
                     return KeyValue.first;
                   });
    std::transform(KeyDataToSort.begin(), KeyDataToSort.end(),
                   std::back_inserter(DataSorted),
                   [](const std::pair<KeyTy, ValueTy> &KeyValue) {
                     return KeyValue.second;
                   });

    if constexpr (std::is_same_v<Compare, std::less<KeyTy>>) {
      assert(KeysToSortCase[0] == KeysSorted);
      assert(DataToSortCase[0] == DataSorted);
    }

    assert(KeysToSortCase[1] == KeysSorted);
    assert(DataToSortCase[1] == DataSorted);
    assert(KeysToSortCase[2] == KeysSorted);
    assert(DataToSortCase[2] == DataSorted);
    if constexpr (!std::is_same_v<CustomType, KeyTy>) {
      assert(KeysToSortCase[3] == KeysSorted);
      assert(DataToSortCase[3] == DataSorted);
    }
  }
}

template <
    UseGroupT UseGroup, int Dim, size_t ElementsPerWorkItem, class KeyTy,
    class ValueTy, typename Compare,
    typename Properties = sycl::ext::oneapi::experimental::empty_properties_t>
void RunOnData(sycl::queue &Q, const std::vector<KeyTy> &Keys,
               const std::vector<ValueTy> &Values, const Compare &Comparator,
               Properties Prop = {}) {
  if constexpr (UseGroup == UseGroupT::SubGroup)
    if (Q.get_backend() == sycl::backend::ext_oneapi_cuda ||
        Q.get_backend() == sycl::backend::ext_oneapi_hip) {
      std::cout << "Note! Skipping sub group testing on CUDA BE" << std::endl;
      return;
    }

  RunSortKeyValueOverGroupArray<UseGroup, Dim, ElementsPerWorkItem>(
      Q, Keys, Values, Comparator, Prop);
};

template <class KeyTy, class ValueTy>
void RunOverType(sycl::queue &Q, size_t DataSize) {
  constexpr size_t PerWI = 4;
  std::vector<KeyTy> KeysRandom(DataSize * PerWI);
  std::vector<ValueTy> DataRandom(DataSize * PerWI);

  // Fill using random numbers
  {
    std::default_random_engine generator;
    std::normal_distribution<float> distribution((10.0), (2.0));
    for (KeyTy &Elem : KeysRandom)
      Elem = KeyTy(distribution(generator));

    for (ValueTy &Elem : DataRandom)
      Elem = ValueTy(distribution(generator));
  }

  auto input_blocked = oneapi_exp::properties{oneapi_exp::input_data_placement<
      oneapi_exp::group_algorithm_data_placement::blocked>};
  auto input_striped = oneapi_exp::properties{oneapi_exp::input_data_placement<
      oneapi_exp::group_algorithm_data_placement::striped>};
  auto output_blocked =
      oneapi_exp::properties{oneapi_exp::output_data_placement<
          oneapi_exp::group_algorithm_data_placement::blocked>};
  auto output_striped =
      oneapi_exp::properties{oneapi_exp::output_data_placement<
          oneapi_exp::group_algorithm_data_placement::striped>};
  auto both_striped = oneapi_exp::properties{
      oneapi_exp::input_data_placement<
          oneapi_exp::group_algorithm_data_placement::striped>,
      oneapi_exp::output_data_placement<
          oneapi_exp::group_algorithm_data_placement::striped>};
  RunOnData<UseGroupT::WorkGroup, 1, PerWI>(Q, KeysRandom, DataRandom,
                                            std::less<KeyTy>{}, output_blocked);
  RunOnData<UseGroupT::WorkGroup, 1, PerWI>(Q, KeysRandom, DataRandom,
                                            std::less<KeyTy>{}, input_striped);
  RunOnData<UseGroupT::WorkGroup, 1, PerWI>(Q, KeysRandom, DataRandom,
                                            std::less<KeyTy>{}, output_striped);
  RunOnData<UseGroupT::WorkGroup, 2, PerWI>(Q, KeysRandom, DataRandom,
                                            std::less<KeyTy>{}, both_striped);
  RunOnData<UseGroupT::WorkGroup, 2, PerWI>(Q, KeysRandom, DataRandom,
                                            std::less<KeyTy>{}, output_striped);
  RunOnData<UseGroupT::SubGroup, 1, PerWI>(
      Q, KeysRandom, DataRandom, std::greater<KeyTy>{}, input_blocked);
  RunOnData<UseGroupT::SubGroup, 1, PerWI>(
      Q, KeysRandom, DataRandom, std::greater<KeyTy>{}, output_striped);
  RunOnData<UseGroupT::SubGroup, 2, PerWI>(Q, KeysRandom, DataRandom,
                                           std::greater<KeyTy>{}, both_striped);
  RunOnData<UseGroupT::SubGroup, 2, PerWI>(
      Q, KeysRandom, DataRandom, std::greater<KeyTy>{}, input_striped);
}

int main() {
  try {
    sycl::queue Q;

    static constexpr size_t Size = 18;

    RunOverType<std::int32_t, char>(Q, Size);
    RunOverType<CustomType, std::int32_t>(Q, Size);

    std::cout << "Test passed." << std::endl;
    return 0;
  } catch (std::exception &E) {
    std::cout << "Test failed" << std::endl;
    std::cout << E.what() << std::endl;
    return 1;
  }
}
