//==--------------- program.cpp --- SYCL program ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/program.hpp>
#include <CL/sycl/properties/all_properties.hpp>
#include <CL/sycl/property_list.hpp>
#include <detail/backend_impl.hpp>
#include <detail/program_impl.hpp>

#include <vector>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

program::program(const context &context, const property_list &PropList)
    : impl(std::make_shared<detail::program_impl>(
          detail::getSyclObjImpl(context), PropList)) {}

program::program(const context &context, std::vector<device> deviceList,
                 const property_list &PropList)
    : impl(std::make_shared<detail::program_impl>(
          detail::getSyclObjImpl(context), deviceList, PropList)) {}

program::program(std::vector<program> programList,
                 const property_list &PropList)
    : program(std::move(programList), /*linkOptions=*/"", PropList) {}

program::program(std::vector<program> programList, std::string linkOptions,
                 const property_list &PropList) {
  std::vector<std::shared_ptr<detail::program_impl>> impls;
  for (auto &x : programList) {
    impls.push_back(detail::getSyclObjImpl(x));
  }
  impl = std::make_shared<detail::program_impl>(impls, linkOptions, PropList);
}

program::program(const context &context, cl_program clProgram)
    : impl(std::make_shared<detail::program_impl>(
          detail::getSyclObjImpl(context),
          detail::pi::cast<pi_native_handle>(clProgram))) {
  // The implementation constructor takes ownership of the native handle so we
  // must retain it in order to adhere to SYCL 1.2.1 spec (Rev6, section 4.3.1.)
  clRetainProgram(clProgram);
}

backend program::get_backend() const noexcept { return getImplBackend(impl); }

pi_native_handle program::getNative() const { return impl->getNative(); }

program::program(std::shared_ptr<detail::program_impl> impl) : impl(impl) {}

cl_program program::get() const { return impl->get(); }

bool program::is_host() const { return impl->is_host(); }

void program::compile_with_source(std::string kernelSource,
                                  std::string compileOptions) {
  impl->compile_with_source(kernelSource, compileOptions);
}

void program::build_with_source(std::string kernelSource,
                                std::string buildOptions) {
  impl->build_with_source(kernelSource, buildOptions);
}

void program::compile_with_kernel_name(std::string KernelName,
                                       std::string compileOptions,
                                       detail::OSModuleHandle M) {
  impl->compile_with_kernel_name(KernelName, compileOptions, M);
}

void program::build_with_kernel_name(std::string KernelName,
                                     std::string buildOptions,
                                     detail::OSModuleHandle M) {
  impl->build_with_kernel_name(KernelName, buildOptions, M);
}

void program::link(std::string linkOptions) { impl->link(linkOptions); }

bool program::has_kernel(std::string kernelName) const {
  return has_kernel(kernelName, /*IsCreatedFromSource*/ true);
}

bool program::has_kernel(std::string kernelName,
                         bool IsCreatedFromSource) const {
  return impl->has_kernel(kernelName, IsCreatedFromSource);
}

kernel program::get_kernel(std::string kernelName) const {
  return get_kernel(kernelName, /*IsCreatedFromSource*/ true);
}

kernel program::get_kernel(std::string kernelName,
                           bool IsCreatedFromSource) const {
  return impl->get_kernel(kernelName, impl, IsCreatedFromSource);
}

template <info::program param>
typename info::param_traits<info::program, param>::return_type
program::get_info() const {
  return impl->get_info<param>();
}

#define __SYCL_PARAM_TRAITS_SPEC(param_type, param, ret_type)                  \
  template __SYCL_EXPORT ret_type program::get_info<info::param_type::param>() \
      const;

#include <CL/sycl/info/program_traits.def>

#undef __SYCL_PARAM_TRAITS_SPEC

#define __SYCL_PARAM_TRAITS_SPEC(param_type)                                   \
  template <> __SYCL_EXPORT bool program::has_property<param_type>() const {   \
    return impl->has_property<param_type>();                                   \
  }
#include <CL/sycl/detail/properties_traits.def>

#undef __SYCL_PARAM_TRAITS_SPEC

#define __SYCL_PARAM_TRAITS_SPEC(param_type)                                   \
  template <>                                                                  \
  __SYCL_EXPORT param_type program::get_property<param_type>() const {         \
    return impl->get_property<param_type>();                                   \
  }
#include <CL/sycl/detail/properties_traits.def>

#undef __SYCL_PARAM_TRAITS_SPEC

std::vector<std::vector<char>> program::get_binaries() const {
  return impl->get_binaries();
}

context program::get_context() const { return impl->get_context(); }

std::vector<device> program::get_devices() const { return impl->get_devices(); }

std::string program::get_compile_options() const {
  return impl->get_compile_options();
}

std::string program::get_link_options() const {
  return impl->get_link_options();
}

std::string program::get_build_options() const {
  return impl->get_build_options();
}

program_state program::get_state() const { return impl->get_state(); }

void program::set_spec_constant_impl(const char *Name, void *Data,
                                     size_t Size) {
  impl->set_spec_constant_impl(Name, Data, Size);
}

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
