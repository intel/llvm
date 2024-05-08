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

#include <sycl/builtins.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/group_sort.hpp>
#include <sycl/feature_test.hpp>
#include <sycl/group_algorithm.hpp>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

namespace oneapi_exp = sycl::ext::oneapi::experimental;

template <typename...> class KernelNameOverGroup;
template <typename...> class KernelNameOverGroupArray;
template <typename...> class KernelNameKeyValueOverGroup;
template <typename...> class KernelNameKeyValueOverGroupArray;
template <typename...> class KernelNameJoint;

enum class UseGroupT { SubGroup = true, WorkGroup = false };

// these classes are needed to pass non-type template parameters to KernelName
template <int> class IntWrapper;
template <UseGroupT> class UseGroupWrapper;

class CustomType {
public:
  CustomType(size_t Val) : MVal(Val) {}
  CustomType() : MVal(0) {}

  bool operator<(const CustomType &RHS) const { return MVal < RHS.MVal; }
  bool operator>(const CustomType &RHS) const { return MVal > RHS.MVal; }
  bool operator==(const CustomType &RHS) const { return MVal == RHS.MVal; }

private:
  size_t MVal = 0;
};

template <class T> struct ConvertToSimpleType {
  using Type = T;
};

// Dummy overloads for CustomType which is not supported by radix sorter
template <> struct ConvertToSimpleType<CustomType> {
  using Type = int;
};

template <class SorterT> struct ConvertToSortingOrder;

template <class T> struct ConvertToSortingOrder<std::greater<T>> {
  static const auto Type = oneapi_exp::sorting_order::descending;
};

template <class T> struct ConvertToSortingOrder<std::less<T>> {
  static const auto Type = oneapi_exp::sorting_order::ascending;
};

constexpr size_t ReqSubGroupSize = 8;

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

             // This version of API always sorts in ascending order
             if constexpr (std::is_same_v<Compare, std::less<T>>)
               oneapi_exp::joint_sort(
                   oneapi_exp::group_with_scratchpad(
                       Group, sycl::span{&ScratchDefault[LocalPartID],
                                         LocalMemorySizeDefault}),
                   &AccToSort0[StartIdx], &AccToSort0[EndIdx]); // (1)

             oneapi_exp::joint_sort(
                 oneapi_exp::group_with_scratchpad(
                     Group, sycl::span{&ScratchDefault[LocalPartID],
                                       LocalMemorySizeDefault}),
                 &AccToSort1[StartIdx], &AccToSort1[EndIdx], Comp); // (2)

             oneapi_exp::joint_sort(
                 Group, &AccToSort2[StartIdx], &AccToSort2[EndIdx],
                 oneapi_exp::default_sorters::joint_sorter<Compare>(
                     sycl::span{&ScratchDefault[LocalPartID],
                                LocalMemorySizeDefault})); // (3) default

             const size_t LocalPartIDRadix =
                 UseGroup == UseGroupT::SubGroup
                     ? LocalMemorySizeRadix * Group.get_group_linear_id()
                     : 0;

             // Radix doesn't support custom types
             if constexpr (!std::is_same_v<CustomType, T>)
               oneapi_exp::joint_sort(
                   Group, &AccToSort3[StartIdx], &AccToSort3[EndIdx],
                   RadixSorterT(sycl::span{&ScratchRadix[LocalPartIDRadix],
                                           LocalMemorySizeRadix})); // (3) radix
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

template <UseGroupT UseGroup, int Dims, class T, class U, class Compare>
void RunSortKeyValueOverGroup(sycl::queue &Q, const std::vector<T> &DataToSort,
                              const std::vector<U> &KeysToSort,
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
      typename ConvertToSimpleType<T>::Type, U,
      ConvertToSortingOrder<Compare>::Type>;

  std::size_t LocalMemorySizeDefault = 0;
  std::size_t LocalMemorySizeRadix = 0;
  if (UseGroup == UseGroupT::SubGroup) {
    // Each sub-group needs a piece of memory for sorting
    LocalMemorySizeDefault =
        oneapi_exp::default_sorters::group_key_value_sorter<
            T, U, Compare>::memory_required(sycl::memory_scope::sub_group,
                                            ReqSubGroupSize);

    LocalMemorySizeRadix = RadixSorterT::memory_required(
        sycl::memory_scope::sub_group, ReqSubGroupSize);
  } else {
    // A single chunk of memory for each work-group
    LocalMemorySizeDefault =
        oneapi_exp::default_sorters::group_key_value_sorter<
            T, U, Compare>::memory_required(sycl::memory_scope::work_group,
                                            NumOfElements);

    LocalMemorySizeRadix = RadixSorterT::memory_required(
        sycl::memory_scope::work_group, NumOfElements);
  }

  std::vector<T> DataToSortCase0 = DataToSort;
  std::vector<U> KeysToSortCase0 = KeysToSort;

  std::vector<T> DataToSortCase1 = DataToSort;
  std::vector<T> KeysToSortCase1 = KeysToSort;

  std::vector<T> DataToSortCase2 = DataToSort;
  std::vector<T> KeysToSortCase2 = KeysToSort;

  std::vector<T> DataToSortCase3 = DataToSort;
  std::vector<T> KeysToSortCase3 = KeysToSort;

  // Sort data using 3 different versions of sort_over_group API
  {
    sycl::buffer<T> BufDataToSort0(DataToSortCase0.data(),
                                   DataToSortCase0.size());
    sycl::buffer<U> BufKeysToSort0(KeysToSortCase0.data(),
                                   KeysToSortCase0.size());

    sycl::buffer<T> BufDataToSort1(DataToSortCase1.data(),
                                   DataToSortCase1.size());
    sycl::buffer<T> BufKeysToSort1(KeysToSortCase1.data(),
                                   KeysToSortCase1.size());

    sycl::buffer<T> BufDataToSort2(DataToSortCase2.data(),
                                   DataToSortCase2.size());
    sycl::buffer<T> BufKeysToSort2(KeysToSortCase2.data(),
                                   KeysToSortCase2.size());

    sycl::buffer<T> BufDataToSort3(DataToSortCase3.data(),
                                   DataToSortCase3.size());
    sycl::buffer<T> BufKeysToSort3(KeysToSortCase3.data(),
                                   KeysToSortCase3.size());

    Q.submit([&](sycl::handler &CGH) {
       auto AccDataToSort0 = sycl::accessor(BufDataToSort0, CGH);
       auto AccKeysToSort0 = sycl::accessor(BufKeysToSort0, CGH);

       auto AccDataToSort1 = sycl::accessor(BufDataToSort1, CGH);
       auto AccKeysToSort1 = sycl::accessor(BufKeysToSort1, CGH);

       auto AccDataToSort2 = sycl::accessor(BufDataToSort2, CGH);
       auto AccKeysToSort2 = sycl::accessor(BufKeysToSort2, CGH);

       auto AccDataToSort3 = sycl::accessor(BufDataToSort3, CGH);
       auto AccKeysToSort3 = sycl::accessor(BufKeysToSort3, CGH);

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

       CGH.parallel_for<KernelNameKeyValueOverGroup<
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
                         T, U, Compare, /*ElementsPerWorkItem*/ 1>(sycl::span{
                         ScratchPtrDefault, LocalMemorySizeDefault})); // (6)

             // Each sub-group should use it's own part of the scratch pad
             const size_t ScratchShiftRadix =
                 UseGroup == UseGroupT::SubGroup
                     ? id.get_sub_group().get_group_linear_id() *
                           LocalMemorySizeRadix
                     : 0;
             std::byte *ScratchPtrRadix = &ScratchRadix[0] + ScratchShiftRadix;

             // Radix doesn't support custom types
             if constexpr (!std::is_same_v<CustomType, T>)
               std::tie(AccKeysToSort3[GlobalLinearID],
                        AccDataToSort3[GlobalLinearID]) =
                   oneapi_exp::sort_key_value_over_group(
                       Group, AccKeysToSort3[GlobalLinearID],
                       AccDataToSort3[GlobalLinearID],
                       RadixSorterT(
                           sycl::span{ScratchPtrRadix,
                                      LocalMemorySizeRadix})); // (6) radix
           });
     }).wait_and_throw();
  }

  // Verification
  {
    std::vector<std::pair<T, T>> KeyDataToSort;
    KeyDataToSort.reserve(KeysToSort.size());
    std::transform(KeysToSort.begin(), KeysToSort.end(), DataToSort.begin(),
                   std::back_inserter(KeyDataToSort),
                   [](T Key, T Value) { return std::make_pair(Key, Value); });
    // Emulate independent sorting of each work-group/sub-group
    const size_t ChunkSize = UseGroup == UseGroupT::SubGroup
                                 ? ReqSubGroupSize
                                 : NDRange.get_local_range().size();
    auto It = KeyDataToSort.begin();
    auto KeyValueComp = [&](const std::pair<T, T> &A,
                            const std::pair<T, T> &B) -> bool {
      return Comp(A.first, B.first);
    };
    for (; (It + ChunkSize) < KeyDataToSort.end(); It += ChunkSize)
      std::stable_sort(It, It + ChunkSize, KeyValueComp);

    // Sort remainder
    std::stable_sort(It, KeyDataToSort.end(), KeyValueComp);

    std::vector<T> KeysSorted, DataSorted;
    KeysSorted.reserve(KeyDataToSort.size());
    DataSorted.reserve(KeyDataToSort.size());
    std::transform(
        KeyDataToSort.begin(), KeyDataToSort.end(),
        std::back_inserter(KeysSorted),
        [](const std::pair<T, T> &KeyValue) { return KeyValue.first; });
    std::transform(
        KeyDataToSort.begin(), KeyDataToSort.end(),
        std::back_inserter(DataSorted),
        [](const std::pair<T, T> &KeyValue) { return KeyValue.second; });

    if constexpr (std::is_same_v<Compare, std::less<T>>) {
      assert(KeysToSortCase0 == KeysSorted);
      assert(DataToSortCase0 == DataSorted);
    }

    assert(KeysToSortCase1 == KeysSorted);
    assert(DataToSortCase1 == DataSorted);
    assert(KeysToSortCase2 == KeysSorted);
    assert(DataToSortCase2 == DataSorted);
    if constexpr (!std::is_same_v<CustomType, T>) {
      assert(KeysToSortCase3 == KeysSorted);
      assert(DataToSortCase3 == DataSorted);
    }
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

  using RadixSorterT = oneapi_exp::radix_sorters::group_sorter<
      typename ConvertToSimpleType<T>::Type,
      ConvertToSortingOrder<Compare>::Type>;

  std::size_t LocalMemorySizeDefault = 0;
  std::size_t LocalMemorySizeRadix = 0;
  if (UseGroup == UseGroupT::SubGroup) {
    // Each sub-group needs a piece of memory for sorting
    LocalMemorySizeDefault = oneapi_exp::default_sorters::group_sorter<
        T, 1, Compare>::memory_required(sycl::memory_scope::sub_group,
                                        ReqSubGroupSize);

    LocalMemorySizeRadix = RadixSorterT::memory_required(
        sycl::memory_scope::sub_group, ReqSubGroupSize);
  } else {
    // A single chunk of memory for each work-group
    LocalMemorySizeDefault = oneapi_exp::default_sorters::group_sorter<
        T, 1, Compare>::memory_required(sycl::memory_scope::work_group,
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
                   AccToSort0[GlobalLinearID]); // (4)

             AccToSort1[GlobalLinearID] = oneapi_exp::sort_over_group(
                 oneapi_exp::group_with_scratchpad(
                     Group,
                     sycl::span{ScratchPtrDefault, LocalMemorySizeDefault}),
                 AccToSort1[GlobalLinearID], Comp); // (5)

             AccToSort2[GlobalLinearID] = oneapi_exp::sort_over_group(
                 Group, AccToSort2[GlobalLinearID],
                 oneapi_exp::default_sorters::group_sorter<T, 1, Compare>(
                     sycl::span{ScratchPtrDefault,
                                LocalMemorySizeDefault})); // (6) default

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
                   RadixSorterT(sycl::span{ScratchPtrRadix,
                                           LocalMemorySizeRadix})); // (6) radix
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

template <UseGroupT UseGroup, int Dims, size_t ElementsPerWorkItem,
          class Property = oneapi_exp::detail::is_blocked, class T,
          class Compare>
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

  using RadixSorterT = oneapi_exp::radix_sorters::group_sorter<
      typename ConvertToSimpleType<T>::Type,
      ConvertToSortingOrder<Compare>::Type, ElementsPerWorkItem>;

  std::size_t LocalMemorySizeDefault = 0;
  std::size_t LocalMemorySizeRadix = 0;
  if (UseGroup == UseGroupT::SubGroup) {
    // Each sub-group needs a piece of memory for sorting
    LocalMemorySizeDefault = oneapi_exp::default_sorters::
        group_sorter<T, ElementsPerWorkItem, Compare>::memory_required(
            sycl::memory_scope::sub_group, ReqSubGroupSize);

    LocalMemorySizeRadix = RadixSorterT::memory_required(
        sycl::memory_scope::sub_group, ReqSubGroupSize);
  } else {
    // A single chunk of memory for each work-group
    LocalMemorySizeDefault = oneapi_exp::default_sorters::
        group_sorter<T, ElementsPerWorkItem, Compare>::memory_required(
            sycl::memory_scope::work_group, NDRange.get_local_range().size());

    LocalMemorySizeRadix = RadixSorterT::memory_required(
        sycl::memory_scope::work_group, NDRange.get_local_range().size());
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

       CGH.parallel_for<KernelNameOverGroupArray<
           IntWrapper<Dims>, UseGroupWrapper<UseGroup>, Property, T, Compare>>(
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

             T ValsPrivate[ElementsPerWorkItem];

             auto ReadToPrivate = [&](auto Acc) {
               for (std::size_t I = 0; I < ElementsPerWorkItem; ++I)
                 ValsPrivate[I] = Acc[GlobalLinearID * ElementsPerWorkItem + I];
             };
             auto WriteToGlobal = [&](auto Acc) {
               for (std::size_t I = 0; I < ElementsPerWorkItem; ++I)
                 Acc[GlobalLinearID * ElementsPerWorkItem + I] = ValsPrivate[I];
             };

             if constexpr (std::is_same_v<Compare, std::less<T>>) {
               ReadToPrivate(AccToSort0);
               oneapi_exp::sort_over_group(
                   oneapi_exp::group_with_scratchpad(
                       Group,
                       sycl::span{ScratchPtrDefault, LocalMemorySizeDefault}),
                   sycl::span<T, ElementsPerWorkItem>{
                       ValsPrivate, ValsPrivate + ElementsPerWorkItem},
                   Prop); // (4)
               WriteToGlobal(AccToSort0);
             }

             ReadToPrivate(AccToSort1);
             oneapi_exp::sort_over_group(
                 oneapi_exp::group_with_scratchpad(
                     Group,
                     sycl::span{ScratchPtrDefault, LocalMemorySizeDefault}),
                 sycl::span<T, ElementsPerWorkItem>{
                     ValsPrivate, ValsPrivate + ElementsPerWorkItem},
                 Comp, Prop); // (5)
             WriteToGlobal(AccToSort1);

             ReadToPrivate(AccToSort2);
             oneapi_exp::sort_over_group(
                 Group,
                 sycl::span<T, ElementsPerWorkItem>{
                     ValsPrivate, ValsPrivate + ElementsPerWorkItem},
                 oneapi_exp::default_sorters::group_sorter<
                     T, ElementsPerWorkItem, Compare>(
                     sycl::span{ScratchPtrDefault, LocalMemorySizeDefault}),
                 Prop); // (6)
             WriteToGlobal(AccToSort2);

             // Each sub-group should use it's own part of the scratch pad
             const size_t ScratchShiftRadix =
                 UseGroup == UseGroupT::SubGroup
                     ? id.get_sub_group().get_group_linear_id() *
                           LocalMemorySizeRadix
                     : 0;
             std::byte *ScratchPtrRadix = &ScratchRadix[0] + ScratchShiftRadix;

             /* // Radix doesn't support custom types */
             if constexpr (!std::is_same_v<CustomType, T>) {
               ReadToPrivate(AccToSort3);
               oneapi_exp::sort_over_group(
                   Group,
                   sycl::span<T, ElementsPerWorkItem>{
                       ValsPrivate, ValsPrivate + ElementsPerWorkItem},
                   RadixSorterT(
                       sycl::span{ScratchPtrRadix, LocalMemorySizeRadix}),
                   Prop); // (6) radix
               WriteToGlobal(AccToSort3);
             }
           });
     }).wait_and_throw();
  }

  // Verification
  {
    // Emulate independent sorting of each work-group/sub-group
    const size_t ChunkSize =
        (UseGroup == UseGroupT::SubGroup ? ReqSubGroupSize
                                         : NDRange.get_local_range().size()) *
        ElementsPerWorkItem;
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

template <UseGroupT UseGroup, int Dims, size_t ElementsPerWorkItem,
          class Property = oneapi_exp::detail::is_blocked, class T, class U,
          class Compare>
void RunSortKeyValueOverGroupArray(sycl::queue &Q,
                                   const std::vector<T> &DataToSort,
                                   const std::vector<U> &KeysToSort,
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

  using RadixSorterT = oneapi_exp::radix_sorters::group_key_value_sorter<
      typename ConvertToSimpleType<T>::Type, U,
      ConvertToSortingOrder<Compare>::Type, ElementsPerWorkItem>;

  std::size_t LocalMemorySizeDefault = 0;
  std::size_t LocalMemorySizeRadix = 0;
  if (UseGroup == UseGroupT::SubGroup) {
    // Each sub-group needs a piece of memory for sorting
    LocalMemorySizeDefault =
        oneapi_exp::default_sorters::group_key_value_sorter<
            T, U, Compare,
            ElementsPerWorkItem>::memory_required(sycl::memory_scope::sub_group,
                                                  ReqSubGroupSize);

    LocalMemorySizeRadix = RadixSorterT::memory_required(
        sycl::memory_scope::sub_group, ReqSubGroupSize);
  } else {
    // A single chunk of memory for each work-group
    LocalMemorySizeDefault =
        oneapi_exp::default_sorters::group_key_value_sorter<
            T, U, Compare, ElementsPerWorkItem>::
            memory_required(sycl::memory_scope::work_group,
                            NDRange.get_local_range().size());

    LocalMemorySizeRadix = RadixSorterT::memory_required(
        sycl::memory_scope::work_group, NDRange.get_local_range().size());
  }

  std::vector<T> DataToSortCase0 = DataToSort;
  std::vector<U> KeysToSortCase0 = KeysToSort;

  std::vector<T> DataToSortCase1 = DataToSort;
  std::vector<T> KeysToSortCase1 = KeysToSort;

  std::vector<T> DataToSortCase2 = DataToSort;
  std::vector<T> KeysToSortCase2 = KeysToSort;

  std::vector<T> DataToSortCase3 = DataToSort;
  std::vector<T> KeysToSortCase3 = KeysToSort;

  // Sort data using 3 different versions of sort_over_group API
  {
    sycl::buffer<T> BufDataToSort0(DataToSortCase0.data(),
                                   DataToSortCase0.size());
    sycl::buffer<U> BufKeysToSort0(KeysToSortCase0.data(),
                                   KeysToSortCase0.size());

    sycl::buffer<T> BufDataToSort1(DataToSortCase1.data(),
                                   DataToSortCase1.size());
    sycl::buffer<T> BufKeysToSort1(KeysToSortCase1.data(),
                                   KeysToSortCase1.size());

    sycl::buffer<T> BufDataToSort2(DataToSortCase2.data(),
                                   DataToSortCase2.size());
    sycl::buffer<T> BufKeysToSort2(KeysToSortCase2.data(),
                                   KeysToSortCase2.size());

    sycl::buffer<T> BufDataToSort3(DataToSortCase3.data(),
                                   DataToSortCase3.size());
    sycl::buffer<T> BufKeysToSort3(KeysToSortCase3.data(),
                                   KeysToSortCase3.size());

    Q.submit([&](sycl::handler &CGH) {
       auto AccDataToSort0 = sycl::accessor(BufDataToSort0, CGH);
       auto AccKeysToSort0 = sycl::accessor(BufKeysToSort0, CGH);

       auto AccDataToSort1 = sycl::accessor(BufDataToSort1, CGH);
       auto AccKeysToSort1 = sycl::accessor(BufKeysToSort1, CGH);

       auto AccDataToSort2 = sycl::accessor(BufDataToSort2, CGH);
       auto AccKeysToSort2 = sycl::accessor(BufKeysToSort2, CGH);

       auto AccDataToSort3 = sycl::accessor(BufDataToSort3, CGH);
       auto AccKeysToSort3 = sycl::accessor(BufKeysToSort3, CGH);

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

       CGH.parallel_for<KernelNameKeyValueOverGroupArray<
           IntWrapper<Dims>, UseGroupWrapper<UseGroup>, Property, T, Compare>>(
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

             T KeysPrivate[ElementsPerWorkItem];
             U ValsPrivate[ElementsPerWorkItem];

             auto ReadToPrivate = [&](auto &KeyAcc, auto &ValAcc) {
               for (std::size_t I = 0; I < ElementsPerWorkItem; ++I) {
                 KeysPrivate[I] =
                     KeyAcc[GlobalLinearID * ElementsPerWorkItem + I];
                 ValsPrivate[I] =
                     ValAcc[GlobalLinearID * ElementsPerWorkItem + I];
               }
             };
             auto WriteToGlobal = [&](auto &KeyAcc, auto &ValAcc) {
               for (std::size_t I = 0; I < ElementsPerWorkItem; ++I) {
                 KeyAcc[GlobalLinearID * ElementsPerWorkItem + I] =
                     KeysPrivate[I];
                 ValAcc[GlobalLinearID * ElementsPerWorkItem + I] =
                     ValsPrivate[I];
               }
             };

             if constexpr (std::is_same_v<Compare, std::less<T>>) {
               ReadToPrivate(AccKeysToSort0, AccDataToSort0);
               oneapi_exp::sort_key_value_over_group(
                   oneapi_exp::group_with_scratchpad(
                       Group,
                       sycl::span{ScratchPtrDefault, LocalMemorySizeDefault}),
                   sycl::span<T, ElementsPerWorkItem>{
                       KeysPrivate, KeysPrivate + ElementsPerWorkItem},
                   sycl::span<U, ElementsPerWorkItem>{
                       ValsPrivate, ValsPrivate + ElementsPerWorkItem},
                   Prop); // (4)
               WriteToGlobal(AccKeysToSort0, AccDataToSort0);
             }

             ReadToPrivate(AccKeysToSort1, AccDataToSort1);
             oneapi_exp::sort_key_value_over_group(
                 oneapi_exp::group_with_scratchpad(
                     Group,
                     sycl::span{ScratchPtrDefault, LocalMemorySizeDefault}),
                 sycl::span<T, ElementsPerWorkItem>{
                     KeysPrivate, KeysPrivate + ElementsPerWorkItem},
                 sycl::span<U, ElementsPerWorkItem>{
                     ValsPrivate, ValsPrivate + ElementsPerWorkItem},
                 Comp, Prop); // (5)
             WriteToGlobal(AccKeysToSort1, AccDataToSort1);

             ReadToPrivate(AccKeysToSort2, AccDataToSort2);
             oneapi_exp::sort_key_value_over_group(
                 Group,
                 sycl::span<T, ElementsPerWorkItem>{
                     KeysPrivate, KeysPrivate + ElementsPerWorkItem},
                 sycl::span<U, ElementsPerWorkItem>{
                     ValsPrivate, ValsPrivate + ElementsPerWorkItem},
                 oneapi_exp::default_sorters::group_key_value_sorter<
                     T, U, Compare, ElementsPerWorkItem>(
                     sycl::span{ScratchPtrDefault, LocalMemorySizeDefault}),
                 Prop); // (6)
             WriteToGlobal(AccKeysToSort2, AccDataToSort2);

             // Each sub-group should use it's own part of the scratch pad
             const size_t ScratchShiftRadix =
                 UseGroup == UseGroupT::SubGroup
                     ? id.get_sub_group().get_group_linear_id() *
                           LocalMemorySizeRadix
                     : 0;
             std::byte *ScratchPtrRadix = &ScratchRadix[0] + ScratchShiftRadix;

             // Radix doesn't support custom types
             if constexpr (!std::is_same_v<CustomType, T>) {
               ReadToPrivate(AccKeysToSort3, AccDataToSort3);
               oneapi_exp::sort_key_value_over_group(
                   Group,
                   sycl::span<T, ElementsPerWorkItem>{
                       KeysPrivate, KeysPrivate + ElementsPerWorkItem},
                   sycl::span<U, ElementsPerWorkItem>{
                       ValsPrivate, ValsPrivate + ElementsPerWorkItem},
                   RadixSorterT(
                       sycl::span{ScratchPtrRadix, LocalMemorySizeRadix}),
                   Prop); // (6) radix
               WriteToGlobal(AccKeysToSort3, AccDataToSort3);
             }
           });
     }).wait_and_throw();
  }

  // Verification
  {
    std::vector<std::pair<T, U>> KeyDataToSort;
    KeyDataToSort.reserve(KeysToSort.size());
    std::transform(KeysToSort.begin(), KeysToSort.end(), DataToSort.begin(),
                   std::back_inserter(KeyDataToSort),
                   [](T Key, U Value) { return std::make_pair(Key, Value); });
    // Emulate independent sorting of each work-group/sub-group
    const size_t ChunkSize =
        (UseGroup == UseGroupT::SubGroup ? ReqSubGroupSize
                                         : NDRange.get_local_range().size()) *
        ElementsPerWorkItem;
    auto It = KeyDataToSort.begin();
    auto KeyValueComp = [&](const std::pair<T, U> &A,
                            const std::pair<T, U> &B) -> bool {
      return Comp(A.first, B.first);
    };
    for (; (It + ChunkSize) < KeyDataToSort.end(); It += ChunkSize)
      std::stable_sort(It, It + ChunkSize, KeyValueComp);

    // Sort remainder
    std::stable_sort(It, KeyDataToSort.end(), KeyValueComp);

    std::vector<T> KeysSorted, DataSorted;
    KeysSorted.reserve(KeyDataToSort.size());
    DataSorted.reserve(KeyDataToSort.size());
    std::transform(
        KeyDataToSort.begin(), KeyDataToSort.end(),
        std::back_inserter(KeysSorted),
        [](const std::pair<T, U> &KeyValue) { return KeyValue.first; });
    std::transform(
        KeyDataToSort.begin(), KeyDataToSort.end(),
        std::back_inserter(DataSorted),
        [](const std::pair<T, U> &KeyValue) { return KeyValue.second; });

    if constexpr (std::is_same_v<Compare, std::less<T>>) {
      assert(KeysToSortCase0 == KeysSorted);
      assert(DataToSortCase0 == DataSorted);
    }

    assert(KeysToSortCase1 == KeysSorted);
    assert(DataToSortCase1 == DataSorted);
    assert(KeysToSortCase2 == KeysSorted);
    assert(DataToSortCase2 == DataSorted);
    if constexpr (!std::is_same_v<CustomType, T>) {
      assert(KeysToSortCase3 == KeysSorted);
      assert(DataToSortCase3 == DataSorted);
    }
  }
}

template <class T> void RunOverType(sycl::queue &Q, size_t DataSize) {
  std::vector<T> DataReversed(DataSize);
  std::vector<T> KeysReversed(DataSize);

  std::vector<T> DataRandom(DataSize);
  std::vector<T> KeysRandom(DataSize);

  std::iota(DataReversed.rbegin(), DataReversed.rend(), (size_t)0);
  KeysReversed = DataReversed;

  // Fill using random numbers
  {
    std::default_random_engine generator;
    std::normal_distribution<float> distribution((10.0), (2.0));
    for (T &Elem : DataRandom)
      Elem = T(distribution(generator));

    for (T &Elem : KeysRandom)
      Elem = T(distribution(generator));
  }

  auto RunOnDataAndComp = [&](const std::vector<T> &Data,
                              const std::vector<T> &Keys,
                              const auto &Comparator) {
    RunSortOVerGroup<UseGroupT::WorkGroup, 1>(Q, Data, Comparator);
    RunSortOVerGroup<UseGroupT::WorkGroup, 2>(Q, Data, Comparator);

    RunJointSort<UseGroupT::WorkGroup, 1>(Q, Data, Comparator);
    RunJointSort<UseGroupT::WorkGroup, 2>(Q, Data, Comparator);

    RunSortKeyValueOverGroup<UseGroupT::WorkGroup, 1>(Q, Data, Keys,
                                                      Comparator);
    RunSortKeyValueOverGroup<UseGroupT::WorkGroup, 2>(Q, Data, Keys,
                                                      Comparator);

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

  RunOnDataAndComp(DataReversed, KeysReversed, std::greater<T>{});
  RunOnDataAndComp(DataReversed, KeysReversed, std::less<T>{});
  RunOnDataAndComp(DataRandom, KeysRandom, std::less<T>{});
  RunOnDataAndComp(DataRandom, KeysRandom, std::greater<T>{});

  constexpr size_t ElementsPerWorkItem = 4;
  std::vector<T> ArrayDataReversed(DataSize * ElementsPerWorkItem);
  std::vector<T> ArrayKeysReversed(DataSize * ElementsPerWorkItem);

  std::vector<T> ArrayDataRandom(DataSize * ElementsPerWorkItem);
  std::vector<T> ArrayKeysRandom(DataSize * ElementsPerWorkItem);

  std::iota(ArrayDataReversed.rbegin(), ArrayDataReversed.rend(), (size_t)0);
  ArrayKeysReversed = ArrayDataReversed;

  // Fill using random numbers
  {
    std::default_random_engine generator;
    std::normal_distribution<float> distribution((10.0), (2.0));
    for (T &Elem : ArrayDataRandom)
      Elem = T(distribution(generator));

    for (T &Elem : ArrayKeysRandom)
      Elem = T(distribution(generator));
  }

  auto RunOnDataAndCompArray = [&](const std::vector<T> &Data,
                                   const std::vector<T> &Keys,
                                   const auto &Comparator) {
    RunSortOverGroupArray<UseGroupT::WorkGroup, 1, ElementsPerWorkItem>(
        Q, Data, Comparator, oneapi_exp::detail::is_blocked{});
    // TODO: enable testing of is_striped.
    // RunSortOverGroupArray<UseGroupT::WorkGroup, 1, ElementsPerWorkItem>(
    //     Q, Data, Comparator, oneapi_exp::detail::is_striped{});
    RunSortOverGroupArray<UseGroupT::WorkGroup, 2, ElementsPerWorkItem>(
        Q, Data, Comparator, oneapi_exp::detail::is_blocked{});

    if (Q.get_backend() == sycl::backend::ext_oneapi_cuda ||
        Q.get_backend() == sycl::backend::ext_oneapi_hip) {
      std::cout << "Note! Skipping sub group testing on CUDA BE" << std::endl;
      return;
    }

    RunSortOverGroupArray<UseGroupT::SubGroup, 1, ElementsPerWorkItem>(
        Q, Data, Comparator, oneapi_exp::detail::is_blocked{});
    RunSortOverGroupArray<UseGroupT::SubGroup, 2, ElementsPerWorkItem>(
        Q, Data, Comparator, oneapi_exp::detail::is_blocked{});
  };

  RunOnDataAndCompArray(ArrayDataReversed, ArrayKeysReversed,
                        std::greater<T>{});
  RunOnDataAndCompArray(ArrayDataReversed, ArrayKeysReversed, std::less<T>{});
  RunOnDataAndCompArray(ArrayDataRandom, ArrayKeysRandom, std::less<T>{});
  RunOnDataAndCompArray(ArrayDataRandom, ArrayKeysRandom, std::greater<T>{});

  auto RunKeyValueOnDataAndCompArray = [&](const std::vector<T> &Data,
                                           const std::vector<T> &Keys,
                                           const auto &Comparator) {
    RunSortKeyValueOverGroupArray<UseGroupT::WorkGroup, 1, ElementsPerWorkItem>(
        Q, Data, Keys, Comparator, oneapi_exp::detail::is_blocked{});
    RunSortKeyValueOverGroupArray<UseGroupT::WorkGroup, 2, ElementsPerWorkItem>(
        Q, Data, Keys, Comparator, oneapi_exp::detail::is_blocked{});

    if (Q.get_backend() == sycl::backend::ext_oneapi_cuda ||
        Q.get_backend() == sycl::backend::ext_oneapi_hip) {
      std::cout << "Note! Skipping sub group testing on CUDA BE" << std::endl;
      return;
    }

    RunSortKeyValueOverGroupArray<UseGroupT::SubGroup, 1, ElementsPerWorkItem>(
        Q, Data, Keys, Comparator, oneapi_exp::detail::is_blocked{});
    RunSortKeyValueOverGroupArray<UseGroupT::SubGroup, 2, ElementsPerWorkItem>(
        Q, Data, Keys, Comparator, oneapi_exp::detail::is_blocked{});
  };

  RunKeyValueOnDataAndCompArray(ArrayDataReversed, ArrayKeysReversed,
                                std::greater<T>{});
  RunKeyValueOnDataAndCompArray(ArrayDataReversed, ArrayKeysReversed,
                                std::less<T>{});

  // TODO: Currently there is an issue with key/value sorting for array input -
  // for some reasons sorting is not stable, order of values is not preserved.
  //  RunKeyValueOnDataAndCompArray(ArrayDataRandom, ArrayKeysRandom,
  //  std::less<T>{}); RunKeyValueOnDataAndCompArray(ArrayDataRandom,
  //  ArrayKeysRandom, std::greater<T>{});
}

int main() {
  static_assert(SYCL_EXT_ONEAPI_GROUP_SORT == 2,
                "Unexpected extension version");
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
