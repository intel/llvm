//==--- barrier.hpp - SYCL_ONEAPI_BARRIER  ---------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once


#include <atomic>
#include <cstddef>
#include <CL/sycl/atomic.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl::ext::oneapi {

struct arrival_token {
    int phase;
};

template <sycl::memory_scope scope, typename CompletionFunction>
class barrier {
    sycl::atomic_ref<std::ptrdiff_t, sycl::memory_order::relaxed, scope, sycl::access::address_space:::local_space> counter;
    sycl::atomic_ref<std::ptrdiff_t, sycl::memory_order::relaxed, scope, sycl::access::address_space:::local_space> expected_count;
    int phase;

    public:
        // PTX expects an unsigned 32 bit int https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-init
        // the valid range of count is [1,2^20 -1] - https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-contents
        // TODO should this have a completion function??
        // expected to sync after this so all threads see the initialized barrier (maybe the sync should be part of the constructor)
        // TODO only one thread needs to initialize, but many init is fine
        barrier(sycl::local_accessor<std::ptrdiff_t, 0> expected_count, sycl::local_accessor<std::ptrdiff_t, 0> count) : counter(&count), expected_count(expected_count){ }
        ~barrier();

        // barriers cannot be moved or copied
        barrier(const barrier& other) = delete;
        barrier(barrier&& other) noexcept = delete;
        barrier& operator=(const barrier& other) = delete;
        barrier& operator=(barrier&& other) noexcept = delete;

        const arrival_token arrive() {
            counter.fetch_sub(1);
            return arrival_token{phase};
        }
        void wait(arrival_token&& arrival) const {
            while (counter != )
        }
        // equivalent to wait(arrive());
        void arrive_and_wait();
        // arrive and also drop the expected count by one
        void arrive_and_drop();


        // returns the maximum value of expected count that is supported by the implementation.
        static constexpr std::uint32_t max() noexcept();
};

} // namespace sycl::ext::oneapi
}