//==----- program_impl.hpp --- SYCL program implementation -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/common_info.hpp>
#include <CL/sycl/detail/kernel_desc.hpp>
#include <CL/sycl/detail/kernel_impl.hpp>
#include <CL/sycl/detail/program_manager/program_manager.hpp>
#include <CL/sycl/device.hpp>
#include <CL/sycl/program.hpp>
#include <CL/sycl/stl.hpp>

#include <algorithm>
#include <cassert>
#include <fstream>
#include <memory>

__SYCL_INLINE namespace cl {
namespace sycl {

// Forward declarations
class kernel;

namespace detail {

using ContextImplPtr = std::shared_ptr<detail::context_impl>;

// Used to identify the module the user code, which included this header,
// belongs to. Incurs some marginal inefficiency - there will be one copy
// per '#include "program_impl.hpp"'
static void *AddressInThisModule = &AddressInThisModule;

class program_impl {
public:
  program_impl() = delete;

  explicit program_impl(ContextImplPtr Context)
      : program_impl(Context, Context->get_info<info::context::devices>()) {}

  program_impl(ContextImplPtr Context, vector_class<device> DeviceList)
      : Context(Context), Devices(DeviceList) {}

  // Kernels caching for linked programs won't be allowed due to only compiled
  // state of each and every program in the list and thus unknown state of
  // caching resolution
  program_impl(vector_class<std::shared_ptr<program_impl>> ProgramList,
               string_class LinkOptions = "");


  program_impl(ContextImplPtr Context, RT::PiKernel Kernel);

  program_impl(const context &Context, RT::PiProgram Program);

  ~program_impl();

  cl_program get() const;

  RT::PiProgram &getHandleRef() { return Program; }
  const RT::PiProgram &getHandleRef() const { return Program; }

  bool is_host() const { return Context->is_host(); }

  void compile_with_kernel_type(string_class KernelName,
                                string_class CompileOptions = "");

  void compile_with_source(string_class KernelSource,
                           string_class CompileOptions = "");

  void build_with_kernel_type(string_class KernelName,
                              string_class BuildOptions = "");

  void build_with_source(string_class KernelSource,
                         string_class BuildOptions = "");

  void link(string_class LinkOptions = "");

  bool has_kernel(string_class KernelName, bool IsCreatedFromSource) const;

  kernel get_kernel(string_class KernelName,
                    std::shared_ptr<program_impl> PtrToSelf,
                    bool IsCreatedFromSource) const;

  template <info::program param>
  typename info::param_traits<info::program, param>::return_type
  get_info() const;

  vector_class<vector_class<char>> get_binaries() const;

  context get_context() const {
    if (is_host())
      return context();
    return createSyclObjFromImpl<context>(Context);
  }

  vector_class<device> get_devices() const { return Devices; }

  string_class get_compile_options() const { return CompileOptions; }

  string_class get_link_options() const { return LinkOptions; }

  string_class get_build_options() const { return BuildOptions; }

  program_state get_state() const { return State; }

private:
  template <info::device param>
  void check_device_feature_support(const vector_class<device> &devices) {
    for (const auto &device : devices) {
      if (!device.get_info<param>()) {
        throw feature_not_supported(
            "Online compilation is not supported by this device");
      }
    }
  }

  void create_pi_program_with_kernel_name(OSModuleHandle M,
                                          const string_class &KernelName);

  void create_cl_program_with_il(OSModuleHandle M);

  void create_cl_program_with_source(const string_class &Source);

  void compile(const string_class &Options);

  void build(const string_class &Options);

  vector_class<RT::PiDevice> get_pi_devices() const;

  bool is_cacheable() const { return IsProgramAndKernelCachingAllowed; }

  static bool is_cacheable_with_options(const string_class &Options) {
    return Options.empty();
  }

  bool has_cl_kernel(const string_class &KernelName) const;

  RT::PiKernel get_pi_kernel(const string_class &KernelName) const;

  std::vector<device>
  sort_devices_by_cl_device_id(vector_class<device> Devices);

  void throw_if_state_is(program_state State) const;

  void throw_if_state_is_not(program_state State) const;

  RT::PiProgram Program = nullptr;
  program_state State = program_state::none;
  ContextImplPtr Context;
  bool IsLinkable = false;
  vector_class<device> Devices;
  string_class CompileOptions;
  string_class LinkOptions;
  string_class BuildOptions;

  // Only allow kernel caching for programs constructed with context only (or
  // device list and context) and built with build_with_kernel_type with
  // default build options
  bool IsProgramAndKernelCachingAllowed = false;
};

template <>
cl_uint program_impl::get_info<info::program::reference_count>() const;

template <> context program_impl::get_info<info::program::context>() const;

template <>
vector_class<device> program_impl::get_info<info::program::devices>() const;

} // namespace detail
} // namespace sycl
} // namespace cl
