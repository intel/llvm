//==-------------- cg.cpp --------------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CL/sycl/detail/cg.hpp"
#include <CL/sycl/detail/scheduler/scheduler.hpp>
#include <CL/sycl/detail/scheduler/commands.hpp>
#include <CL/sycl/detail/memory_manager.hpp>
#include <CL/sycl/detail/queue_impl.hpp>


#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace cl {
namespace sycl {

cl_mem interop_handler::getMemImpl(detail::Requirement* Req) const {
    auto Iter = std::find_if(std::begin(MMemObjs), std::end(MMemObjs),
      [=](ReqToMem Elem) {
        return (Elem.first == Req);
    });

    if (Iter == std::end(MMemObjs)) {
        throw("Invalid memory object used inside interop");
    }
    return detail::pi::cast<cl_mem>(Iter->second);
  }

}  // sycl
}  // sycl
