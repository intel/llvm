//==----- program_impl.hpp --- SYCL program implementation -----------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/program_manager/program_manager.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/common_info.hpp>
#include <CL/sycl/detail/kernel_desc.hpp>
#include <CL/sycl/context.hpp>
#include <CL/sycl/device.hpp>
#include <CL/sycl/kernel.hpp>
#include <CL/sycl/stl.hpp>

#include <fstream>
#include <memory>

namespace cl {
namespace sycl {

enum class program_state { none, compiled, linked };

namespace detail {

class program_impl {
public:
  program_impl() = delete;

  explicit program_impl(const context &Context)
      : program_impl(Context, Context.get_devices()) {}

  program_impl(const context &Context, vector_class<device> DeviceList)
      : Context(Context), Devices(DeviceList) {}

  program_impl(vector_class<std::shared_ptr<program_impl>> ProgramList,
               string_class LinkOptions = "")
      : State(program_state::linked), LinkOptions(LinkOptions) {
    // Verify arguments
    if (ProgramList.empty()) {
      throw runtime_error("Non-empty vector of programs expected");
    }
    Context = ProgramList[0]->Context;
    Devices = ProgramList[0]->Devices;
    for (const auto &Prg : ProgramList) {
      Prg->throw_if_state_is_not(program_state::compiled);
      if (Prg->Context != Context) {
        throw invalid_object_error(
            "Not all programs are associated with the same context");
      }
      if (Prg->Devices != Devices) {
        throw invalid_object_error(
            "Not all programs are associated with the same devices");
      }
    }

    if (!is_host()) {
      vector_class<cl_device_id> ClDevices(get_cl_devices());
      vector_class<cl_program> ClPrograms;
      for (const auto &Prg : ProgramList) {
        ClPrograms.push_back(Prg->ClProgram);
      }
      cl_int Err;
      ClProgram =
          clLinkProgram(Context.get(), ClDevices.size(), ClDevices.data(),
                        LinkOptions.c_str(), ProgramList.size(),
                        ClPrograms.data(), nullptr, nullptr, &Err);
      CHECK_OCL_CODE_THROW(Err, compile_program_error);
    }
  }

  program_impl(const context &Context, cl_program ClProgram)
      : ClProgram(ClProgram), Context(Context) {
    // TODO it's unclear how to handle getting compile, link and build options
    // in this case
    // TODO handle the case when cl_program build is in progress
    cl_uint NumDevices;
    CHECK_OCL_CODE(clGetProgramInfo(ClProgram, CL_PROGRAM_NUM_DEVICES,
                                    sizeof(cl_uint), &NumDevices, nullptr));
    vector_class<cl_device_id> ClDevices(NumDevices);
    CHECK_OCL_CODE(clGetProgramInfo(ClProgram, CL_PROGRAM_DEVICES,
                                    sizeof(cl_device_id) * NumDevices,
                                    ClDevices.data(), nullptr));
    Devices = vector_class<device>(ClDevices.begin(), ClDevices.end());
    // TODO check build for each device instead
    cl_program_binary_type BinaryType;
    CHECK_OCL_CODE(clGetProgramBuildInfo(
        ClProgram, Devices[0].get(), CL_PROGRAM_BINARY_TYPE,
        sizeof(cl_program_binary_type), &BinaryType, nullptr));
    switch (BinaryType) {
    case CL_PROGRAM_BINARY_TYPE_NONE:
      State = program_state::none;
      break;
    case CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT:
      State = program_state::compiled;
      break;
    case CL_PROGRAM_BINARY_TYPE_LIBRARY:
    case CL_PROGRAM_BINARY_TYPE_EXECUTABLE:
      State = program_state::linked;
    }
    CHECK_OCL_CODE(clRetainProgram(ClProgram));
  }

  program_impl(const context &Context, cl_kernel ClKernel)
      : program_impl(
            Context,
            ProgramManager::getInstance().getClProgramFromClKernel(ClKernel)) {}

  ~program_impl() {
    // TODO replace CHECK_OCL_CODE_NO_EXC to CHECK_OCL_CODE and
    // catch an exception and put it to list of asynchronous exceptions
    if (!is_host() && ClProgram != nullptr) {
      CHECK_OCL_CODE_NO_EXC(clReleaseProgram(ClProgram));
    }
  }

  cl_program get() const {
    throw_if_state_is(program_state::none);
    if (is_host()) {
      throw invalid_object_error("This instance of program is a host instance");
    }
    CHECK_OCL_CODE(clRetainProgram(ClProgram));
    return ClProgram;
  }

  bool is_host() const { return Context.is_host(); }

  template <typename KernelT>
  void compile_with_kernel_type(string_class CompileOptions = "") {
    throw_if_state_is_not(program_state::none);
    // TODO Check for existence of kernel
    if (!is_host()) {
      create_cl_program_with_il();
      compile(CompileOptions);
    }
    State = program_state::compiled;
  }

  void compile_with_source(string_class KernelSource,
                           string_class CompileOptions = "") {
    throw_if_state_is_not(program_state::none);
    // TODO should it throw if it's host?
    if (!is_host()) {
      create_cl_program_with_source(KernelSource);
      compile(CompileOptions);
    }
    State = program_state::compiled;
  }

  template <typename KernelT>
  void build_with_kernel_type(string_class BuildOptions = "") {
    throw_if_state_is_not(program_state::none);
    // TODO Check for existence of kernel
    if (!is_host()) {
      create_cl_program_with_il();
      build(BuildOptions);
    }
    State = program_state::linked;
  }

  void build_with_source(string_class KernelSource,
                         string_class BuildOptions = "") {
    throw_if_state_is_not(program_state::none);
    // TODO should it throw if it's host?
    if (!is_host()) {
      create_cl_program_with_source(KernelSource);
      build(BuildOptions);
    }
    State = program_state::linked;
  }

  void link(string_class LinkOptions = "") {
    throw_if_state_is_not(program_state::compiled);
    if (!is_host()) {
      vector_class<cl_device_id> ClDevices(get_cl_devices());
      cl_int Err;
      ClProgram = clLinkProgram(Context.get(), ClDevices.size(),
                                ClDevices.data(), LinkOptions.c_str(), 1,
                                &ClProgram, nullptr, nullptr, &Err);
      CHECK_OCL_CODE_THROW(Err, compile_program_error);
      LinkOptions = LinkOptions;
    }
    State = program_state::linked;
  }

  template <typename KernelT>
  bool has_kernel() const
#ifdef __SYCL_DEVICE_ONLY__
      ;
#else
  {
    throw_if_state_is(program_state::none);
    if (is_host()) {
      return true;
    }
    return has_cl_kernel(KernelInfo<KernelT>::getName());
  }
#endif

  bool has_kernel(string_class KernelName) const {
    throw_if_state_is(program_state::none);
    if (is_host()) {
      return false;
    }
    return has_cl_kernel(KernelName);
  }

  template <typename KernelT>
  kernel get_kernel(std::shared_ptr<program_impl> PtrToSelf) const
#ifdef __SYCL_DEVICE_ONLY__
      ;
#else
  {
    throw_if_state_is(program_state::none);
    if (is_host()) {
      return createSyclObjFromImpl<kernel>(
          std::make_shared<kernel_impl>(Context, PtrToSelf));
    }
    return createSyclObjFromImpl<kernel>(std::make_shared<kernel_impl>(
        get_cl_kernel(KernelInfo<KernelT>::getName()), Context, PtrToSelf));
  }
#endif

  kernel get_kernel(string_class KernelName,
                    std::shared_ptr<program_impl> PtrToSelf) const {
    throw_if_state_is(program_state::none);
    if (is_host()) {
      throw invalid_object_error("This instance of program is a host instance");
    }
    return createSyclObjFromImpl<kernel>(std::make_shared<kernel_impl>(
        get_cl_kernel(KernelName), Context, PtrToSelf));
  }

  template <info::program param>
  typename info::param_traits<info::program, param>::return_type
  get_info() const;

  vector_class<vector_class<char>> get_binaries() const {
    throw_if_state_is(program_state::none);
    vector_class<size_t> BinarySizes(Devices.size());
    CHECK_OCL_CODE(clGetProgramInfo(ClProgram, CL_PROGRAM_BINARY_SIZES,
                                    sizeof(size_t) * BinarySizes.size(),
                                    BinarySizes.data(), nullptr));

    vector_class<vector_class<char>> Result;
    vector_class<char *> Pointers;
    for (size_t I = 0; I < BinarySizes.size(); ++I) {
      Result.emplace_back(BinarySizes[I]);
      Pointers.push_back(Result[I].data());
    }
    CHECK_OCL_CODE(clGetProgramInfo(ClProgram, CL_PROGRAM_BINARIES,
                                    sizeof(char *) * Pointers.size(),
                                    Pointers.data(), nullptr));
    return Result;
  }

  context get_context() const { return Context; }

  vector_class<device> get_devices() const { return Devices; }

  string_class get_compile_options() const { return CompileOptions; }

  string_class get_link_options() const { return LinkOptions; }

  string_class get_build_options() const { return BuildOptions; }

  program_state get_state() const { return State; }

private:
  void create_cl_program_with_il() {
    assert(!ClProgram && "This program already has an encapsulated cl_program");
    ClProgram = ProgramManager::getInstance().createOpenCLProgram(Context);
  }

  void create_cl_program_with_source(const string_class &Source) {
    assert(!ClProgram && "This program already has an encapsulated cl_program");
    cl_int Err;
    const char *Src = Source.c_str();
    size_t Size = Source.size();
    ClProgram = clCreateProgramWithSource(Context.get(), 1, &Src, &Size, &Err);
    CHECK_OCL_CODE(Err);
  }

  void compile(const string_class &Options) {
    vector_class<cl_device_id> ClDevices(get_cl_devices());
    // TODO make the exception message more descriptive
    if (clCompileProgram(ClProgram, ClDevices.size(), ClDevices.data(),
                         Options.c_str(), 0, nullptr, nullptr, nullptr,
                         nullptr) != CL_SUCCESS) {
      throw compile_program_error("Program compilation error");
    }
    CompileOptions = Options;
  }

  void build(const string_class &Options) {
    vector_class<cl_device_id> ClDevices(get_cl_devices());
    // TODO make the exception message more descriptive
    if (clBuildProgram(ClProgram, ClDevices.size(), ClDevices.data(),
                       Options.c_str(), nullptr, nullptr) != CL_SUCCESS) {
      throw compile_program_error("Program build error");
    }
    BuildOptions = Options;
  }

  vector_class<cl_device_id> get_cl_devices() const {
    vector_class<cl_device_id> ClDevices;
    for (const auto &Device : Devices) {
      ClDevices.push_back(Device.get());
    }
    return ClDevices;
  }

  bool has_cl_kernel(const string_class &KernelName) const {
    size_t Size;
    CHECK_OCL_CODE(clGetProgramInfo(ClProgram, CL_PROGRAM_KERNEL_NAMES, 0,
                                    nullptr, &Size));
    string_class ClResult(Size, ' ');
    CHECK_OCL_CODE(clGetProgramInfo(ClProgram, CL_PROGRAM_KERNEL_NAMES,
                                    ClResult.size(), &ClResult[0], nullptr));
    // Get rid of the null terminator
    ClResult.pop_back();
    vector_class<string_class> KernelNames(split_string(ClResult, ';'));
    for (const auto &Name : KernelNames) {
      if (Name == KernelName) {
        return true;
      }
    }
    return false;
  }

  cl_kernel get_cl_kernel(const string_class &KernelName) const {
    cl_int Err;
    cl_kernel ClKernel = clCreateKernel(ClProgram, KernelName.c_str(), &Err);
    if (Err == CL_INVALID_KERNEL_NAME) {
      throw invalid_object_error(
          "This instance of program does not contain the kernel requested");
    }
    CHECK_OCL_CODE(Err);
    return ClKernel;
  }

  void throw_if_state_is(program_state State) const {
    if (this->State == State) {
      throw invalid_object_error("Invalid program state");
    }
  }

  void throw_if_state_is_not(program_state State) const {
    if (this->State != State) {
      throw invalid_object_error("Invalid program state");
    }
  }

  cl_program ClProgram = nullptr;
  program_state State = program_state::none;
  context Context;
  vector_class<device> Devices;
  string_class CompileOptions;
  string_class LinkOptions;
  string_class BuildOptions;
};

template <>
cl_uint program_impl::get_info<info::program::reference_count>() const;

template <> context program_impl::get_info<info::program::context>() const;

template <>
vector_class<device> program_impl::get_info<info::program::devices>() const;

} // namespace detail
} // namespace sycl
} // namespace cl
