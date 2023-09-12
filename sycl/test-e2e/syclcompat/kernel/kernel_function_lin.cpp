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
 *  kernel_function_lin.cpp
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

// REQUIRES: linux

// RUN: %clangxx -fPIC -shared -fsycl -fsycl-targets=%{sycl_triple} %S/Inputs/kernel_module_lin.cpp -o %t.so
// RUN: %clangxx -DTEST_SHARED_LIB='"%t.so"' -ldl -fsycl -fsycl-targets=%{sycl_triple} %t.so %s -o %t.out
// RUN: %{run} %t.out

#include <dlfcn.h>
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

void kernel_functor_ptr() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  syclcompat::device_ext &dev_ct1 = syclcompat::get_current_device();
  sycl::queue *q_ct1 = dev_ct1.default_queue();

  int Size = dev_ct1.get_info<sycl::info::device::max_work_group_size>();
  assert(getTemplateFuncAttrs<int>() == Size);
  assert(getFuncAttrs() == Size);

  void *M;
  syclcompat::kernel_functor F;

  M = dlopen(TEST_SHARED_LIB, RTLD_LAZY);
  if (M == NULL) {
    std::cout << "Could not load the library" << std::endl;
    std::cout << "  " << TEST_SHARED_LIB << std::endl << std::flush;
    assert(false); // FAIL
  }

  std::string FunctionName = "foo_wrapper";
  F = (syclcompat::kernel_functor)dlsym(M, FunctionName.c_str());
  if (F == NULL) {
    std::cout << "Could not load function pointer" << std::endl << std::flush;
    dlclose(M);
    assert(false); // FAIL
  }

  int sharedSize = 10;
  void **param = nullptr, **extra = nullptr;

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
  dlclose(M);
}

int main() {
  kernel_functor_ptr();

  return 0;
}
