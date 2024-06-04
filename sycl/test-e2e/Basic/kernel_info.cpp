// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
// Fail is flaky for level_zero, enable when fixed.
// UNSUPPORTED: level_zero
//
// Consistently fails with opencl gpu, enable when fixed.
// XFAIL: opencl && gpu

//==--- kernel_info.cpp - SYCL kernel info test ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <sycl/detail/core.hpp>

using namespace sycl;

int main() {
  queue q;

  buffer<int, 1> buf(range<1>(1));
  auto KernelID = sycl::get_kernel_id<class SingleTask>();
  auto KB =
      get_kernel_bundle<bundle_state::executable>(q.get_context(), {KernelID});
  kernel krn = KB.get_kernel(KernelID);

  q.submit([&](handler &cgh) {
    auto acc = buf.get_access<access::mode::read_write>(cgh);
    cgh.single_task<class SingleTask>(krn, [=]() { acc[0] = acc[0] + 1; });
  });

  const std::string krnName = krn.get_info<info::kernel::function_name>();
  assert(!krnName.empty());

  std::string ErrMsg = "";
  std::error_code Errc;
  bool ExceptionWasThrown = false;
  try {
    const cl_uint krnArgCount = krn.get_info<info::kernel::num_args>();
  } catch (exception &e) {
    ErrMsg = e.what();
    Errc = e.code();
    ExceptionWasThrown = true;
  }
  assert(ExceptionWasThrown && "Invalid using of \"info::kernel::num_args\" "
                               "query should throw an exception.");
  assert(ErrMsg ==
         "info::kernel::num_args descriptor may only be used to query a kernel "
         "that resides in a kernel bundle constructed using a backend specific"
         "interoperability function or to query a device built-in kernel");
  assert(Errc == errc::invalid);

  const context krnCtx = krn.get_info<info::kernel::context>();
  assert(krnCtx == q.get_context());
  const cl_uint krnRefCount = krn.get_info<info::kernel::reference_count>();
  assert(krnRefCount > 0);
  const std::string krnAttr = krn.get_info<info::kernel::attributes>();
  assert(krnAttr.empty());

  device dev = q.get_device();
  const size_t wgSize =
      krn.get_info<info::kernel_device_specific::work_group_size>(dev);
  assert(wgSize > 0);
  const size_t prefWGSizeMult = krn.get_info<
      info::kernel_device_specific::preferred_work_group_size_multiple>(dev);
  assert(prefWGSizeMult > 0);
  const cl_uint maxSgSize =
      krn.get_info<info::kernel_device_specific::max_sub_group_size>(dev);
  assert(0 < maxSgSize && maxSgSize <= wgSize);
  const cl_uint compileSgSize =
      krn.get_info<info::kernel_device_specific::compile_sub_group_size>(dev);
  assert(compileSgSize <= maxSgSize);
  const cl_uint maxNumSg =
      krn.get_info<info::kernel_device_specific::max_num_sub_groups>(dev);
  assert(0 < maxNumSg);
  const cl_uint compileNumSg =
      krn.get_info<info::kernel_device_specific::compile_num_sub_groups>(dev);
  assert(compileNumSg <= maxNumSg);

  {
    std::error_code Errc;
    std::string ErrMsg = "";
    bool IsExceptionThrown = false;
    try {
      krn.get_info<sycl::info::kernel_device_specific::global_work_size>(dev);
      auto BuiltInIds = dev.get_info<info::device::built_in_kernel_ids>();
      bool isBuiltInKernel = std::find(BuiltInIds.begin(), BuiltInIds.end(),
                                       KernelID) != BuiltInIds.end();
      bool isCustomDevice = dev.get_info<sycl::info::device::device_type>() ==
                            sycl::info::device_type::custom;
      assert((isCustomDevice || isBuiltInKernel) &&
             "info::kernel_device_specific::global_work_size descriptor can "
             "only be used with custom device "
             "or built-in kernel.");

    } catch (sycl::exception &e) {
      IsExceptionThrown = true;
      Errc = e.code();
      ErrMsg = e.what();
    }
    assert(IsExceptionThrown &&
           "Invalid using of info::kernel_device_specific::global_work_size "
           "query should throw an exception.");
    assert(Errc == errc::invalid);
    assert(ErrMsg ==
           "info::kernel_device_specific::global_work_size descriptor may only "
           "be used if the device type is device_type::custom or if the "
           "kernel is a built-in kernel.");
  }
}
