// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -fsyntax-only -verify %s

// Tests the diagnostics for kernel attributes that do not match the equivalent
// kernel property in add_ir_attributes_function on the kernel, regardless of
// how the property was specified.

#include "sycl.hpp"

template <const char *Name, const char *Value> struct Prop {
  static constexpr const char *name = Name;
  static constexpr const char *value = Value;
};
constexpr char WGSize[] = "sycl-work-group-size";
constexpr char WGSizeHint[] = "sycl-work-group-size-hint";
constexpr char SGSize[] = "sycl-sub-group-size";
constexpr char V8[] = "8";
constexpr char V16[] = "16";
constexpr char V8_4[] = "8,4";
constexpr char V2_2_2[] = "2,2,2";

template <typename... Props> struct Wrapper {
  template <typename KernelName, typename KernelType>
  [[__sycl_detail__::add_ir_attributes_function(
      Props::name..., Props::value...)]] __attribute__((sycl_kernel)) void
  kernel_single_task(const KernelType &KernelFunc) {
    KernelFunc();
  }
};

template <typename... Props> struct WrapperWithImplicit {
  template <typename KernelName, typename KernelType>
  [[__sycl_detail__::add_ir_attributes_function(
      "sycl-single-task", Props::name..., nullptr,
      Props::value...)]] __attribute__((sycl_kernel)) void
  kernel_single_task(const KernelType &KernelFunc) {
    KernelFunc();
  }
};

template <typename... Props> struct WrapperWithFilter {
  template <typename KernelName, typename KernelType>
  [[__sycl_detail__::add_ir_attributes_function(
      {"sycl-sub-group-size"}, Props::name...,
      Props::value...)]] __attribute__((sycl_kernel)) void
  kernel_single_task(const KernelType &KernelFunc) {
    KernelFunc();
  }
};

using AllProps = Wrapper<Prop<WGSize, V8_4>, Prop<WGSizeHint, V2_2_2>,
                         Prop<SGSize, V16>>;

int main() {
  // Matching values: no diagnostic.
  AllProps{}.kernel_single_task<class K1>(
      []() [[sycl::reqd_work_group_size(8, 4)]]
      [[sycl::work_group_size_hint(2, 2, 2)]] [[sycl::reqd_sub_group_size(16)]] {});

  AllProps{}.kernel_single_task<class K2>(
      // expected-error@+1 {{'sycl::reqd_work_group_size' attribute arguments '8' do not match the value '8,4' of kernel property 'work_group_size'}}
      []() [[sycl::reqd_work_group_size(8)]]
      // expected-error@+1 {{'sycl::work_group_size_hint' attribute arguments '2,2' do not match the value '2,2,2' of kernel property 'work_group_size_hint'}}
      [[sycl::work_group_size_hint(2, 2)]]
      // expected-error@+1 {{'sycl::reqd_sub_group_size' attribute arguments '32' do not match the value '16' of kernel property 'sub_group_size'}}
      [[sycl::reqd_sub_group_size(32)]] {});

  // An attribute without an equivalent property is not checked.
  Wrapper<Prop<SGSize, V16>>{}.kernel_single_task<class K3>(
      []() [[sycl::reqd_work_group_size(8)]] {});

  // The implicit "sycl-single-task" property does not affect the check.
  WrapperWithImplicit<Prop<WGSize, V8>>{}.kernel_single_task<class K4>(
      []() [[sycl::reqd_work_group_size(8)]] {});
  WrapperWithImplicit<Prop<WGSize, V8>>{}.kernel_single_task<class K5>(
      // expected-error@+1 {{'sycl::reqd_work_group_size' attribute arguments '16' do not match the value '8' of kernel property 'work_group_size'}}
      []() [[sycl::reqd_work_group_size(16)]] {});

  // Properties removed by the filter are not checked.
  WrapperWithFilter<Prop<WGSize, V8>, Prop<SGSize, V16>>{}
      .kernel_single_task<class K6>(
          // expected-error@+1 {{'sycl::reqd_sub_group_size' attribute arguments '8' do not match the value '16' of kernel property 'sub_group_size'}}
          []() [[sycl::reqd_work_group_size(16)]] [[sycl::reqd_sub_group_size(8)]] {});
}
