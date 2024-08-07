// REQUIRES: sg-8
// RUN: %{build} -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %{run} %t.out

// The test verifies key/value sorting from group_sort extension.
#include "common.hpp"
#include <sycl/ext/oneapi/experimental/group_sort.hpp>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

template <typename...> class KernelNameOverGroup;

template <UseGroupT UseGroup, int Dims, class KeyTy, class ValueTy,
          class Compare>
void RunSortKeyValueOverGroup(sycl::queue &Q,
                              const std::vector<KeyTy> &KeysToSort,
                              const std::vector<ValueTy> &DataToSort,
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

  using RadixSorterT = oneapi_exp::radix_sorters::group_key_value_sorter<
      typename ConvertToSimpleType<KeyTy>::Type, ValueTy,
      ConvertToSortingOrder<Compare>::Type>;

  std::size_t LocalMemorySizeDefault = 0;
  std::size_t LocalMemorySizeRadix = 0;
  if (UseGroup == UseGroupT::SubGroup) {
    // Each sub-group needs a piece of memory for sorting
    LocalMemorySizeDefault = oneapi_exp::default_sorters::
        group_key_value_sorter<KeyTy, ValueTy, Compare>::memory_required(
            sycl::memory_scope::sub_group, ReqSubGroupSize);

    LocalMemorySizeRadix = RadixSorterT::memory_required(
        sycl::memory_scope::sub_group, ReqSubGroupSize);
  } else {
    // A single chunk of memory for each work-group
    LocalMemorySizeDefault = oneapi_exp::default_sorters::
        group_key_value_sorter<KeyTy, ValueTy, Compare>::memory_required(
            sycl::memory_scope::work_group, NumOfElements);

    LocalMemorySizeRadix = RadixSorterT::memory_required(
        sycl::memory_scope::work_group, NumOfElements);
  }

  std::vector<KeyTy> KeysToSortCase0 = KeysToSort;
  std::vector<ValueTy> DataToSortCase0 = DataToSort;

  std::vector<KeyTy> KeysToSortCase1 = KeysToSort;
  std::vector<ValueTy> DataToSortCase1 = DataToSort;

  std::vector<KeyTy> KeysToSortCase2 = KeysToSort;
  std::vector<ValueTy> DataToSortCase2 = DataToSort;

  std::vector<KeyTy> KeysToSortCase3 = KeysToSort;
  std::vector<ValueTy> DataToSortCase3 = DataToSort;

  // Sort data using 3 different versions of sort_over_group API
  {
    sycl::buffer<KeyTy> BufKeysToSort0(KeysToSortCase0.data(),
                                       KeysToSortCase0.size());
    sycl::buffer<ValueTy> BufDataToSort0(DataToSortCase0.data(),
                                         DataToSortCase0.size());

    sycl::buffer<KeyTy> BufKeysToSort1(KeysToSortCase1.data(),
                                       KeysToSortCase1.size());
    sycl::buffer<ValueTy> BufDataToSort1(DataToSortCase1.data(),
                                         DataToSortCase1.size());

    sycl::buffer<KeyTy> BufKeysToSort2(KeysToSortCase2.data(),
                                       KeysToSortCase2.size());
    sycl::buffer<ValueTy> BufDataToSort2(DataToSortCase2.data(),
                                         DataToSortCase2.size());

    sycl::buffer<KeyTy> BufKeysToSort3(KeysToSortCase3.data(),
                                       KeysToSortCase3.size());
    sycl::buffer<ValueTy> BufDataToSort3(DataToSortCase3.data(),
                                         DataToSortCase3.size());

    Q.submit([&](sycl::handler &CGH) {
       auto AccKeysToSort0 = sycl::accessor(BufKeysToSort0, CGH);
       auto AccDataToSort0 = sycl::accessor(BufDataToSort0, CGH);

       auto AccKeysToSort1 = sycl::accessor(BufKeysToSort1, CGH);
       auto AccDataToSort1 = sycl::accessor(BufDataToSort1, CGH);

       auto AccKeysToSort2 = sycl::accessor(BufKeysToSort2, CGH);
       auto AccDataToSort2 = sycl::accessor(BufDataToSort2, CGH);

       auto AccKeysToSort3 = sycl::accessor(BufKeysToSort3, CGH);
       auto AccDataToSort3 = sycl::accessor(BufDataToSort3, CGH);

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

       auto KeyValueSortKernel =
           [=](sycl::nd_item<Dims> id) [[intel::reqd_sub_group_size(
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

             if constexpr (std::is_same_v<Compare, std::less<KeyTy>>)
               std::tie(AccKeysToSort0[GlobalLinearID],
                        AccDataToSort0[GlobalLinearID]) =
                   oneapi_exp::sort_key_value_over_group(
                       oneapi_exp::group_with_scratchpad(
                           Group, sycl::span{ScratchPtrDefault,
                                             LocalMemorySizeDefault}),
                       AccKeysToSort0[GlobalLinearID],
                       AccDataToSort0[GlobalLinearID]); // (4)

             std::tie(AccKeysToSort1[GlobalLinearID],
                      AccDataToSort1[GlobalLinearID]) =
                 oneapi_exp::sort_key_value_over_group(
                     oneapi_exp::group_with_scratchpad(
                         Group,
                         sycl::span{ScratchPtrDefault, LocalMemorySizeDefault}),
                     AccKeysToSort1[GlobalLinearID],
                     AccDataToSort1[GlobalLinearID], Comp); // (5)

             std::tie(AccKeysToSort2[GlobalLinearID],
                      AccDataToSort2[GlobalLinearID]) =
                 oneapi_exp::sort_key_value_over_group(
                     Group, AccKeysToSort2[GlobalLinearID],
                     AccDataToSort2[GlobalLinearID],
                     oneapi_exp::default_sorters::group_key_value_sorter<
                         KeyTy, ValueTy, Compare, /*ElementsPerWorkItem*/ 1>(
                         sycl::span{ScratchPtrDefault,
                                    LocalMemorySizeDefault})); // (6)

             // Each sub-group should use its own part of the scratch pad
             const size_t ScratchShiftRadix =
                 UseGroup == UseGroupT::SubGroup
                     ? id.get_sub_group().get_group_linear_id() *
                           LocalMemorySizeRadix
                     : 0;
             std::byte *ScratchPtrRadix = &ScratchRadix[0] + ScratchShiftRadix;

             // Radix doesn't support custom types
             if constexpr (!std::is_same_v<CustomType, KeyTy>)
               std::tie(AccKeysToSort3[GlobalLinearID],
                        AccDataToSort3[GlobalLinearID]) =
                   oneapi_exp::sort_key_value_over_group(
                       Group, AccKeysToSort3[GlobalLinearID],
                       AccDataToSort3[GlobalLinearID],
                       RadixSorterT(
                           sycl::span{ScratchPtrRadix,
                                      LocalMemorySizeRadix})); // (6) radix
           };

       CGH.parallel_for<
           KernelNameOverGroup<IntWrapper<Dims>, UseGroupWrapper<UseGroup>,
                               KeyTy, ValueTy, Compare>>(NDRange,
                                                         KeyValueSortKernel);
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
    const size_t ChunkSize = UseGroup == UseGroupT::SubGroup
                                 ? ReqSubGroupSize
                                 : NDRange.get_local_range().size();
    auto It = KeyDataToSort.begin();
    auto KeyValueComp = [&](const std::pair<KeyTy, ValueTy> &A,
                            const std::pair<KeyTy, ValueTy> &B) -> bool {
      return Comp(A.first, B.first);
    };
    for (; (It + ChunkSize) < KeyDataToSort.end(); It += ChunkSize)
      std::stable_sort(It, It + ChunkSize, KeyValueComp);

    // Sort remainder
    std::stable_sort(It, KeyDataToSort.end(), KeyValueComp);

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
      assert(KeysToSortCase0 == KeysSorted);
      assert(DataToSortCase0 == DataSorted);
    }

    assert(KeysToSortCase1 == KeysSorted);
    assert(DataToSortCase1 == DataSorted);
    assert(KeysToSortCase2 == KeysSorted);
    assert(DataToSortCase2 == DataSorted);
    if constexpr (!std::is_same_v<CustomType, KeyTy>) {
      assert(KeysToSortCase3 == KeysSorted);
      assert(DataToSortCase3 == DataSorted);
    }
  }
}

template <class KeyTy, class ValueTy>
void RunOverType(sycl::queue &Q, size_t DataSize) {
  std::vector<KeyTy> KeysRandom(DataSize);
  std::vector<ValueTy> DataRandom(DataSize);

  // Fill using random numbers
  {
    std::default_random_engine generator;
    std::normal_distribution<float> distribution((10.0), (2.0));
    for (KeyTy &Elem : KeysRandom)
      Elem = KeyTy(distribution(generator));

    for (ValueTy &Elem : DataRandom)
      Elem = ValueTy(distribution(generator));
  }

  auto RunOnDataAndComp = [&](const std::vector<KeyTy> &Keys,
                              const std::vector<ValueTy> &Data,
                              const auto &Comparator) {
    RunSortKeyValueOverGroup<UseGroupT::WorkGroup, 1>(Q, Keys, Data,
                                                      Comparator);
    RunSortKeyValueOverGroup<UseGroupT::WorkGroup, 2>(Q, Keys, Data,
                                                      Comparator);

    if (Q.get_backend() == sycl::backend::ext_oneapi_cuda ||
        Q.get_backend() == sycl::backend::ext_oneapi_hip) {
      std::cout << "Note! Skipping sub group testing on CUDA BE" << std::endl;
      return;
    }

    RunSortKeyValueOverGroup<UseGroupT::SubGroup, 1>(Q, Keys, Data, Comparator);
    RunSortKeyValueOverGroup<UseGroupT::SubGroup, 2>(Q, Keys, Data, Comparator);
  };

  RunOnDataAndComp(KeysRandom, DataRandom, std::less<KeyTy>{});
  RunOnDataAndComp(KeysRandom, DataRandom, std::greater<KeyTy>{});
}

int main() {
  try {
    sycl::queue Q;

    std::vector<size_t> Sizes{18, 64};

    for (size_t Size : Sizes) {
      RunOverType<std::int32_t, char>(Q, Size);
      RunOverType<char, std::int32_t>(Q, Size);
      if (Q.get_device().has(sycl::aspect::fp16))
        RunOverType<sycl::half, std::int32_t>(Q, Size);
      if (Q.get_device().has(sycl::aspect::fp64))
        RunOverType<double, char>(Q, Size);
      RunOverType<CustomType, std::int32_t>(Q, Size);
    }

    std::cout << "Test passed." << std::endl;
    return 0;
  } catch (std::exception &E) {
    std::cout << "Test failed" << std::endl;
    std::cout << E.what() << std::endl;
    return 1;
  }
}
