//==------------------ free_function_kernel_args.cpp -----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/free_function_kernel_args.hpp>

#include <detail/free_function_kernel_args_impl.hpp>

#include <sycl/context.hpp>
#include <sycl/detail/cl.h>
#include <sycl/ext/oneapi/experimental/work_group_memory.hpp>
#include <sycl/sampler.hpp>

#include <cstring>
#include <memory>

namespace sycl {
inline namespace _V1 {
namespace detail {

FreeFunctionArgCollector::FreeFunctionArgCollector()
    : MImpl(new FreeFunctionArgsStorage()) {}

FreeFunctionArgCollector::~FreeFunctionArgCollector() { delete MImpl; }

FreeFunctionArgCollector::FreeFunctionArgCollector(
    FreeFunctionArgCollector &&Other) noexcept
    : MImpl(Other.MImpl) {
  Other.MImpl = nullptr;
}

FreeFunctionArgCollector &
FreeFunctionArgCollector::operator=(FreeFunctionArgCollector &&Other) noexcept {
  if (this != &Other) {
    delete MImpl;
    MImpl = Other.MImpl;
    Other.MImpl = nullptr;
  }
  return *this;
}

void FreeFunctionArgCollector::addPlainArg(kernel_param_kind_t Kind,
                                           const void *Ptr, int Size,
                                           int ArgIndex) {
  MImpl->MArgsStorage.emplace_back(Size);
  void *Storage = static_cast<void *>(MImpl->MArgsStorage.back().data());
  std::memcpy(Storage, Ptr, Size);
  MImpl->MArgs.emplace_back(Kind, Storage, Size, ArgIndex);
}

void FreeFunctionArgCollector::addWorkGroupMemoryArg(const void *ImplPtr,
                                                     int ArgIndex) {
  auto Impl = std::make_shared<work_group_memory_impl>(
      *static_cast<const work_group_memory_impl *>(ImplPtr));
  MImpl->MArgs.emplace_back(kernel_param_kind_t::kind_work_group_memory,
                            Impl.get(), 0, ArgIndex);
  MImpl->MSharedPtrStorage.push_back(std::move(Impl));
}

void FreeFunctionArgCollector::addSamplerArg(const void *Ptr, int ArgIndex) {
  auto S = std::make_shared<sampler>(*static_cast<const sampler *>(Ptr));
  MImpl->MArgs.emplace_back(kernel_param_kind_t::kind_sampler, S.get(),
                            static_cast<int>(sizeof(sampler)), ArgIndex);
  MImpl->MSharedPtrStorage.push_back(std::move(S));
}

void FreeFunctionArgCollector::addRawArg(const void *Ptr, size_t Size,
                                         int ArgIndex) {
  MImpl->MArgsStorage.emplace_back(Size);
  void *Storage = static_cast<void *>(MImpl->MArgsStorage.back().data());
  std::memcpy(Storage, Ptr, Size);
  MImpl->MArgs.emplace_back(kernel_param_kind_t::kind_std_layout, Storage,
                            static_cast<int>(Size), ArgIndex);
}

FreeFunctionArgsStorage *FreeFunctionArgCollector::release() noexcept {
  FreeFunctionArgsStorage *Result = MImpl;
  MImpl = nullptr;
  return Result;
}

} // namespace detail
} // namespace _V1
} // namespace sycl
