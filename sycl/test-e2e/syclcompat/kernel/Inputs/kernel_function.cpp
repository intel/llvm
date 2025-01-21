/***************************************************************************
 *
 *  Copyright (C) Codeplay Software Ltd.
 *
 *  Part of the LLVM Project, under the Apache License v2.0 with LLVM
 *  Exceptions. See https://llvm.org/LICENSE.txt for license information.
 *  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  SYCLcompat API
 *
 *  kernel_function.cpp
 *
 *  Description:
 *    kernel_function header API tests
 **************************************************************************/

// The original source was under the license below:
// ====------ kernel_function_lin.cpp---------- -*- C++ -* ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#undef NOMINMAX
#else
#include <dlfcn.h>
#endif

#include <iostream>
#include <string>

#include <sycl/detail/core.hpp>

#include <syclcompat/defs.hpp>
#include <syclcompat/device.hpp>
#include <syclcompat/kernel.hpp>
#include <syclcompat/memory.hpp>

template <class T> void testTemplateKernel(T *data) {}

void testKernel(void *data) {}

template <class T> int getTemplateFuncAttrs() {
  syclcompat::kernel_function_info attrs;
  syclcompat::get_kernel_function_info(&attrs,
                                       (const void *)testTemplateKernel<T>);
  int threadPerBlock = attrs.max_work_group_size;
  return threadPerBlock;
}

int getFuncAttrs() {
  syclcompat::kernel_function_info attrs;
  syclcompat::get_kernel_function_info(&attrs, (const void *)testKernel);
  int threadPerBlock = attrs.max_work_group_size;
  return threadPerBlock;
}

void test_get_func_attrs() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  syclcompat::device_ext &dev_ct1 = syclcompat::get_current_device();

  int size = dev_ct1.get_info<sycl::info::device::max_work_group_size>();
  assert(getTemplateFuncAttrs<int>() == size);
  assert(getFuncAttrs() == size);
}

void call_library_func(syclcompat::kernel_library kernel_lib) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();

  std::string FunctionName = "foo";
  syclcompat::kernel_function func;
  SYCLCOMPAT_CHECK_ERROR(
      func = syclcompat::get_kernel_function(kernel_lib, FunctionName.c_str()));

  if (func == nullptr) {
    std::cout << "Could not load function pointer" << std::endl << std::flush;
    syclcompat::unload_kernel_library(kernel_lib);
    assert(false); // FAIL
  }

  int sharedSize = 10;
  void **param = nullptr, **extra = nullptr;

  constexpr size_t NUM_ELEMENTS = 16;
  int *dev = syclcompat::malloc<int>(NUM_ELEMENTS);
  syclcompat::fill<int>(dev, 0, NUM_ELEMENTS);

  param = (void **)(&dev);
  SYCLCOMPAT_CHECK_ERROR(syclcompat::invoke_kernel_function(
      func, q_ct1, sycl::range<3>(1, 1, 2), sycl::range<3>(1, 1, 8), sharedSize,
      param, extra));
  syclcompat::wait_and_throw();

  int *host_mem = syclcompat::malloc_host<int>(NUM_ELEMENTS);
  syclcompat::memcpy<int>(host_mem, dev, NUM_ELEMENTS);
  for (int i = 0; i < NUM_ELEMENTS; i++) {
    assert(host_mem[i] == i);
  }

  SYCLCOMPAT_CHECK_ERROR(syclcompat::unload_kernel_library(kernel_lib));

  syclcompat::free(dev);
  syclcompat::free(host_mem);
}

void test_kernel_functor_ptr() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  syclcompat::kernel_library kernel_lib;
  SYCLCOMPAT_CHECK_ERROR(kernel_lib =
                             syclcompat::load_kernel_library(TEST_SHARED_LIB));

  if (kernel_lib == nullptr) {
    std::cout << "Could not load the library" << std::endl;
    std::cout << "  " << TEST_SHARED_LIB << std::endl << std::flush;
    assert(false); // FAIL
  }

  call_library_func(kernel_lib);
}

void test_kernel_functor_ptr_memory() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  sycl::queue q_ct1 = syclcompat::get_default_queue();

  std::ifstream ifs;
  ifs.open(TEST_SHARED_LIB, std::ios::in | std::ios::binary);

  std::stringstream buffer;
  buffer << ifs.rdbuf();

  syclcompat::kernel_library kernel_lib;
  SYCLCOMPAT_CHECK_ERROR(
      kernel_lib = syclcompat::load_kernel_library_mem(buffer.str().c_str()));

  if (kernel_lib == nullptr) {
    std::cout << "Could not load the library" << std::endl;
    std::cout << "  " << TEST_SHARED_LIB << std::endl << std::flush;
    assert(false);
  }

  call_library_func(kernel_lib);
}

int main() {
  test_get_func_attrs();
  test_kernel_functor_ptr();
  test_kernel_functor_ptr_memory();

  return 0;
}
