//==--------------- function_class.hpp - SYCL context -----------*- C++-*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/stl.hpp>

namespace cl {
namespace sycl {
namespace detail {

template <typename>
class function_class;

template <typename RetType, typename... Args>
class function_class<RetType(Args...)> {
    public:
    template<typename Functor>
    function_class(Functor Func) : MFuncHolder(std::make_shared<holder<Functor>>(Func)) {}

    template<typename Functor>
    function_class& operator=(const Functor &Func) {
        MFuncHolder = std::make_shared<holder<Functor>>(Func);
        return *this;
    }

    RetType operator()(Args&&... Arguments) const {
        return MFuncHolder->invoke(std::forward<Args>(Arguments)...);
    }

    private:
    class holder_base {
        public:
            virtual ~holder_base() = default;
            virtual RetType invoke(Args...) = 0;
    };

    template <typename Functor>
    class holder final : public holder_base {
        public:
            holder(const Functor& Func) : MFunc(Func) {}
            ~holder() override = default;
            RetType invoke(Args&&... Arguments) override {
                return MFunc(std::forward<Args>(Arguments)...);
            }
        private:
            Functor MFunc;
    };

    shared_ptr_class<holder_base> MFuncHolder;
};

}
}
}
