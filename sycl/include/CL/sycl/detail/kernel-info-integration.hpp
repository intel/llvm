
//==---------------------- spec_const_integration.hpp ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

// This header file must not be included to any DPC++ headers.
// This header file should only be included to integration footer.

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

template <char...> struct KernelInfoData {
  static constexpr unsigned getNumParams() { return 0; }
  static const kernel_param_desc_t &getParamDesc(int Idx) {
    static kernel_param_desc_t Dummy;
    return Dummy;
  }
  static constexpr const char *getName() { return ""; }
  static constexpr bool isESIMD() { return 0; }
  static constexpr bool callsThisItem() { return false; }
  static constexpr bool callsAnyThisFreeFunction() { return false; }
};

// C++14 like index_sequence and make_index_sequence
// not needed C++14 members (value_type, size) not implemented
template <class T, T...> struct integer_sequence {};
template <unsigned long long... I>
using index_sequence = integer_sequence<unsigned long long, I...>;
template <unsigned long long N>
using make_index_sequence =
    __make_integer_seq<integer_sequence, unsigned long long, N>;

template <typename T> struct KernelInfoImpl {
private:
  static constexpr auto n = __builtin_sycl_unique_stable_name(T);
  template <unsigned long long... I>
  static KernelInfoData<n[I]...> impl(index_sequence<I...>) {
    return {};
  }

public:
  using type = decltype(impl(make_index_sequence<__builtin_strlen(n)>{}));
};

// For named kernels, this structure is specialized in the integration header.
// For unnamed kernels, KernelInfoData is specialized in the integration header,
// and this picks it up via the KernelInfoImpl. For non-existent kernels, this
// will also pick up a KernelInfoData (as SubKernelInfo) via KernelInfoImpl, but
// it will instead get the unspecialized case, defined above.
template <class KernelNameType> struct KernelInfo {
  using SubKernelInfo = typename KernelInfoImpl<KernelNameType>::type;
  static constexpr unsigned getNumParams() {
    return SubKernelInfo::getNumParams();
  }
  static const kernel_param_desc_t &getParamDesc(int Idx) {
    return SubKernelInfo::getParamDesc(Idx);
  }
  static constexpr const char *getName() { return SubKernelInfo::getName(); }
  static constexpr bool isESIMD() { return SubKernelInfo::isESIMD(); }
  static constexpr bool callsThisItem() {
    return SubKernelInfo::callsThisItem();
  }
  static constexpr bool callsAnyThisFreeFunction() {
    return SubKernelInfo::callsAnyThisFreeFunction();
  }
};

template<class KernelNameType> KernelInfoStruct getKernelInfoStruct() {
  KernelInfoStruct ret;
  ret.Name = KernelInfo<KernelNameType>::getName();
  ret.NumParams = KernelInfo<KernelNameType>::getNumParams();
  ret.ESIMD = KernelInfo<KernelNameType>::isESIMD();
  ret.ThisItem = KernelInfo<KernelNameType>::callsThisItem();
  ret.ThisFreeFunction = KernelInfo<KernelNameType>::callsAnyThisFreeFunction();
  return ret;
}

template<class KernelNameType> const kernel_param_desc_t &getKernelParamDesc(int Idx) {
  return KernelInfo<KernelNameType>::getParamDesc(Idx);
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
