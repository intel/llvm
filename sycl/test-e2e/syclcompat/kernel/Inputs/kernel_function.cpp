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
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <iostream>
#include <string>

#include <sycl/sycl.hpp>

#include <syclcompat/device.hpp>
#include <syclcompat/kernel.hpp>

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

#ifdef WIN32

#define DECLARE_MODULE_VAR(var) HMODULE var
#define LOAD_LIB(lib) LoadLibraryA(lib)
#define LOAD_FUNCTOR(module, name) GetProcAddress(module, name)
#define FREE_LIB(module) FreeLibrary(module)

#else // LINUX

#define DECLARE_MODULE_VAR(var) void *var
#define LOAD_LIB(lib) dlopen(lib, RTLD_LAZY)
#define LOAD_FUNCTOR(module, name) dlsym(module, name)
#define FREE_LIB(module) dlclose(module)

#endif

void test_kernel_functor_ptr() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  syclcompat::device_ext &dev_ct1 = syclcompat::get_current_device();
  sycl::queue *q_ct1 = dev_ct1.default_queue();

  int Size = dev_ct1.get_info<sycl::info::device::max_work_group_size>();
  assert(getTemplateFuncAttrs<int>() == Size);
  assert(getFuncAttrs() == Size);

  DECLARE_MODULE_VAR(M);
  M = LOAD_LIB(TEST_SHARED_LIB);

  if (M == NULL) {
    std::cout << "Could not load the library" << std::endl;
    std::cout << "  " << TEST_SHARED_LIB << std::endl << std::flush;
    assert(false); // FAIL
  }

  std::string FunctionName = "foo_wrapper";
  syclcompat::kernel_functor F;
  F = (syclcompat::kernel_functor)LOAD_FUNCTOR(M, FunctionName.c_str());

  if (F == NULL) {
    std::cout << "Could not load function pointer" << std::endl << std::flush;
    FREE_LIB(M);
    assert(false); // FAIL
  }

  int sharedSize = 10;
  void **param = nullptr, **extra = nullptr;
  if (!q_ct1->get_device().has(sycl::aspect::usm_shared_allocations)) return;
  int *dev = sycl::malloc_shared<int>(16, *q_ct1);
  for (int i = 0; i < 16; i++) {
    dev[i] = 0;
  }
  param = (void **)(&dev);
  F(*q_ct1,
    sycl::nd_range<3>(sycl::range<3>(1, 1, 2) * sycl::range<3>(1, 1, 8),
                      sycl::range<3>(1, 1, 8)),
    sharedSize, param, extra);
  q_ct1->wait_and_throw();

  for (int i = 0; i < 16; i++) {
    assert(dev[i] == i);
  }

  sycl::free(dev, *q_ct1);
  FREE_LIB(M);
}

int main() {
  test_kernel_functor_ptr();

  return 0;
}
