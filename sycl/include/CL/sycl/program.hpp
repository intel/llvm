//==--------------- program.hpp --- SYCL program ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/program_impl.hpp>
#include <CL/sycl/detail/kernel_desc.hpp>
#include <CL/sycl/info/info_desc.hpp>
#include <CL/sycl/stl.hpp>

#include <memory>

__SYCL_INLINE namespace cl {
namespace sycl {

// Forward declarations
class context;
class device;
class kernel;
/*namespace detail {
class program_impl;
}*/

class program {
  template <class Obj>
  friend decltype(Obj::impl) detail::getSyclObjImpl(const Obj &SyclObject);
  template <class T>
  friend T detail::createSyclObjFromImpl(decltype(T::impl) ImplObj);

public:
  program() = delete;

  explicit program(const context &context);

  program(const context &context, vector_class<device> deviceList);

  program(vector_class<program> programList, string_class linkOptions = "");

  program(const context &context, cl_program clProgram);

  program(const program &rhs) = default;

  program(program &&rhs) = default;

  program &operator=(const program &rhs) = default;

  program &operator=(program &&rhs) = default;

  bool operator==(const program &rhs) const;

  bool operator!=(const program &rhs) const;

  cl_program get() const;

  bool is_host() const;

  template <typename kernelT>
  void compile_with_kernel_type(string_class compileOptions = "") {
    impl->compile_with_kernel_type(detail::KernelInfo<kernelT>::getName(), compileOptions);
  }

  void compile_with_source(string_class kernelSource,
                           string_class compileOptions = "");

  template <typename kernelT>
  void build_with_kernel_type(string_class buildOptions = "") {
    impl->build_with_kernel_type(detail::KernelInfo<kernelT>::getName(), buildOptions);
  }

  void build_with_source(string_class kernelSource,
                         string_class buildOptions = "");

  void link(string_class linkOptions = "");

  template <typename kernelT> bool has_kernel() const {
    return has_kernel(detail::KernelInfo<kernelT>::getName(), /*IsCreatedFromSource*/ false);
  }

  bool has_kernel(string_class kernelName) const;

  template <typename kernelT> kernel get_kernel() const {
    return get_kernel(detail::KernelInfo<kernelT>::getName(), /*IsCreatedFromSource*/false);
  }

  kernel get_kernel(string_class kernelName) const;

  template <info::program param>
  typename info::param_traits<info::program, param>::return_type
  get_info() const;

  vector_class<vector_class<char>> get_binaries() const;

  context get_context() const;

  vector_class<device> get_devices() const;

  string_class get_compile_options() const;

  string_class get_link_options() const;

  string_class get_build_options() const;

  program_state get_state() const;

private:
  program(std::shared_ptr<detail::program_impl> impl);

  kernel get_kernel(string_class kernelName, bool IsCreatedFromSource) const;

  bool has_kernel(string_class kernelName, bool IsCreatedFromSource) const;

  std::shared_ptr<detail::program_impl> impl;
};
} // namespace sycl
} // namespace cl

namespace std {
template <> struct hash<cl::sycl::program> {
  size_t operator()(const cl::sycl::program &prg) const {
    return hash<std::shared_ptr<cl::sycl::detail::program_impl>>()(
        cl::sycl::detail::getSyclObjImpl(prg));
  }
};
} // namespace std
