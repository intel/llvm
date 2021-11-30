// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note -Xclang -verify-ignore-unexpected=error -o - %s
#include <CL/sycl.hpp>
#include <iostream>
int main(){
    {
        int varA = 42;
        int varB = 42;
        int sum  = 0;
        sycl::queue myQueue{};
        {
            auto bufA   = sycl::buffer(&varA, sycl::range{1});
            auto bufB   = sycl::buffer(&varB, sycl::range{1});
            auto bufSum = sycl::buffer(&sum, sycl::range{1});
            myQueue.single_task([&](sycl::handler &cgh) {
                // expected-error-re@CL/sycl/queue.hpp:691 {{static_assert failed due to requirement 'detail::check_fn_signature<(lambda at {{.*}}single_task_error_message.cpp:14:33), void ()>::value || detail::check_fn_signature<(lambda at {{.*}}single_task_error_message.cpp:14:33), void (sycl::kernel_handler)>::value' "sycl::queue.single_task() requires a kernel instead of command group. Use queue.submit() instead"}}
                    auto accessorA      = sycl::accessor(bufA, cgh, sycl::read_only);
                    // expected-error@-1 {{'sycl::buffer<int, 1, sycl::detail::aligned_allocator<char>, void> &' cannot be used as the type of a kernel parameter}}
                    auto accessorB      = sycl::accessor(bufB, cgh, sycl::read_only);
                    // expected-error@-1 {{'sycl::buffer<int, 1, sycl::detail::aligned_allocator<char>, void> &' cannot be used as the type of a kernel parameter}}
                    auto accessorResult = sycl::accessor(bufSum, cgh, sycl::write_only);
                    // expected-error@-1 {{'sycl::buffer<int, 1, sycl::detail::aligned_allocator<char>, void> &' cannot be used as the type of a kernel parameter}}
                    cgh.single_task([=] {
                        accessorResult[0] = accessorA[0] + accessorB[0];
                    });
                })
            .wait();
        }
    }
    {
        int varA = 42;
        int varB = 42;
        int sum  = 0;
        sycl::queue myQueue{};
        {
            auto bufA   = sycl::buffer(&varA, sycl::range{1});
            auto bufB   = sycl::buffer(&varB, sycl::range{1});
            auto bufSum = sycl::buffer(&sum, sycl::range{1});
            sycl::event e {};
            myQueue.single_task(e, [&](sycl::handler &cgh) {
                // expected-error-re@CL/sycl/queue.hpp:710 {{static_assert failed due to requirement 'detail::check_fn_signature<(lambda at {{.*}}single_task_error_message.cpp:39:36), void ()>::value || detail::check_fn_signature<(lambda at {{.*}}single_task_error_message.cpp:39:36), void (sycl::kernel_handler)>::value' "sycl::queue.single_task() requires a kernel instead of command group. Use queue.submit() instead"}}
                    auto accessorA      = sycl::accessor(bufA, cgh, sycl::read_only);
                    // expected-error@-1 {{'sycl::buffer<int, 1, sycl::detail::aligned_allocator<char>, void> &' cannot be used as the type of a kernel parameter}}
                    auto accessorB      = sycl::accessor(bufB, cgh, sycl::read_only);
                    // expected-error@-1 {{'sycl::buffer<int, 1, sycl::detail::aligned_allocator<char>, void> &' cannot be used as the type of a kernel parameter}}
                    auto accessorResult = sycl::accessor(bufSum, cgh, sycl::write_only);
                    // expected-error@-1 {{'sycl::buffer<int, 1, sycl::detail::aligned_allocator<char>, void> &' cannot be used as the type of a kernel parameter}}
                    cgh.single_task([=] {
                        accessorResult[0] = accessorA[0] + accessorB[0];
                    });
                })
            .wait();
        }
    }
    {
        int varA = 42;
        int varB = 42;
        int sum  = 0;
        sycl::queue myQueue{};
        {
            auto bufA   = sycl::buffer(&varA, sycl::range{1});
            auto bufB   = sycl::buffer(&varB, sycl::range{1});
            auto bufSum = sycl::buffer(&sum, sycl::range{1});
            std::vector<sycl::event> vector_event;
            myQueue.single_task(vector_event, [&](sycl::handler &cgh) {
                // expected-error-re@CL/sycl/queue.hpp:731 {{static_assert failed due to requirement 'detail::check_fn_signature<(lambda at {{.*}}single_task_error_message.cpp:64:47), void ()>::value || detail::check_fn_signature<(lambda at {{.*}}single_task_error_message.cpp:64:47), void (sycl::kernel_handler)>::value' "sycl::queue.single_task() requires a kernel instead of command group. Use queue.submit() instead"}}
                    auto accessorA      = sycl::accessor(bufA, cgh, sycl::read_only);
                    // expected-error@-1 {{'sycl::buffer<int, 1, sycl::detail::aligned_allocator<char>, void> &' cannot be used as the type of a kernel parameter}}
                    auto accessorB      = sycl::accessor(bufB, cgh, sycl::read_only);
                    // expected-error@-1 {{'sycl::buffer<int, 1, sycl::detail::aligned_allocator<char>, void> &' cannot be used as the type of a kernel parameter}}
                    auto accessorResult = sycl::accessor(bufSum, cgh, sycl::write_only);
                    // expected-error@-1 {{'sycl::buffer<int, 1, sycl::detail::aligned_allocator<char>, void> &' cannot be used as the type of a kernel parameter}}
                    cgh.single_task([=] {
                        accessorResult[0] = accessorA[0] + accessorB[0];
                    });
                })
            .wait();
        }
    }
}
