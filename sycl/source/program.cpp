//==--------------- program.cpp --- SYCL program ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/program_impl.hpp>
#include <CL/sycl/program.hpp>

#include <vector>

__SYCL_INLINE namespace cl {
namespace sycl {

program::program(const context &context)
    : impl(std::make_shared<detail::program_impl>(
          detail::getSyclObjImpl(context))) {}
program::program(const context &context, vector_class<device> deviceList)
    : impl(std::make_shared<detail::program_impl>(
          detail::getSyclObjImpl(context), deviceList)) {}
program::program(vector_class<program> programList, string_class linkOptions) {
  std::vector<std::shared_ptr<detail::program_impl>> impls;
  for (auto &x : programList) {
    impls.push_back(detail::getSyclObjImpl(x));
  }
  impl = std::make_shared<detail::program_impl>(impls, linkOptions);
}
program::program(const context &context, cl_program clProgram)
    : impl(std::make_shared<detail::program_impl>(
          detail::getSyclObjImpl(context),
          detail::pi::cast<detail::RT::PiProgram>(clProgram))) {}
program::program(std::shared_ptr<detail::program_impl> impl) : impl(impl) {}

cl_program program::get() const { return impl->get(); }

bool program::is_host() const { return impl->is_host(); }

void program::compile_with_source(string_class kernelSource,
                                  string_class compileOptions) {
  impl->compile_with_source(kernelSource, compileOptions);
}

void program::build_with_source(string_class kernelSource,
                                string_class buildOptions) {
  impl->build_with_source(kernelSource, buildOptions);
}

void program::compile_with_kernel_name(string_class KernelName,
                                       string_class compileOptions,
                                       detail::OSModuleHandle M) {
  impl->compile_with_kernel_name(KernelName, compileOptions, M);
}

void program::build_with_kernel_name(string_class KernelName,
                                     string_class buildOptions,
                                     detail::OSModuleHandle M) {
  impl->build_with_kernel_name(KernelName, buildOptions, M);
}

void program::link(string_class linkOptions) { impl->link(linkOptions); }

bool program::has_kernel(string_class kernelName) const {
  return has_kernel(kernelName, /*IsCreatedFromSource*/ true);
}

bool program::has_kernel(string_class kernelName,
                         bool IsCreatedFromSource) const {
  return impl->has_kernel(kernelName, IsCreatedFromSource);
}

kernel program::get_kernel(string_class kernelName) const {
  return get_kernel(kernelName, /*IsCreatedFromSource*/ true);
}

kernel program::get_kernel(string_class kernelName,
                           bool IsCreatedFromSource) const {
  return impl->get_kernel(kernelName, impl, IsCreatedFromSource);
}

template <info::program param>
typename info::param_traits<info::program, param>::return_type
program::get_info() const {
  return impl->get_info<param>();
}

#define PARAM_TRAITS_SPEC(param_type, param, ret_type)                         \
  template ret_type program::get_info<info::param_type::param>() const;

#include <CL/sycl/info/program_traits.def>

#undef PARAM_TRAITS_SPEC

vector_class<vector_class<char>> program::get_binaries() const {
  return impl->get_binaries();
}

context program::get_context() const { return impl->get_context(); }

vector_class<device> program::get_devices() const {
  return impl->get_devices();
}

string_class program::get_compile_options() const {
  return impl->get_compile_options();
}

string_class program::get_link_options() const {
  return impl->get_link_options();
}

string_class program::get_build_options() const {
  return impl->get_build_options();
}

program_state program::get_state() const { return impl->get_state(); }
} // namespace sycl
} // namespace cl
