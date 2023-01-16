// RUN: %clangxx -fsycl %s -o %t.out
// Test for group load/store functionality
// TODO: Add stripped case

#include <algorithm>
#include <cassert>
#include <limits>
#include <numeric>
#include <sycl/sycl.hpp>

namespace sycl_exp = sycl::ext::oneapi::experimental;

constexpr std::size_t block_size = 32;
constexpr std::size_t items_per_thread = 3;
constexpr std::size_t block_count = 2;
constexpr std::size_t size = block_count * block_size * items_per_thread;

template <typename InputContainer, typename OutputContainer>
void test_no_mem(sycl::queue q, InputContainer& input, OutputContainer& output) {
    typedef typename InputContainer::value_type InputT;
    typedef typename OutputContainer::value_type OutputT;
    {
        sycl::buffer<InputT> in_buf(input.data(), input.size());
        sycl::buffer<OutputT> out_buf(output.data(), output.size());

        q.submit([&](sycl::handler& cgh) {
            sycl::accessor in(in_buf, cgh, sycl::read_only);
            sycl::accessor out(out_buf, cgh, sycl::write_only);

            cgh.parallel_for(
                sycl::nd_range<1>(block_count * block_size, block_size),
                [=](sycl::nd_item<1> item) {
                    InputT data[items_per_thread];

                    sycl_exp::joint_load(item.get_group(), in.get_pointer(), sycl::span{ data });

                    for (int i = 0; i < items_per_thread; ++i) {
                        data[i] += item.get_global_linear_id() * 100000 + i * 1000;
                    }

                    sycl_exp::joint_store(item.get_group(), out.get_pointer(), sycl::span{ data });
                });
        });
    }
}

template <typename InputContainer, typename OutputContainer>
void test_local_acc(sycl::queue q, InputContainer& input, OutputContainer& output) {
    typedef typename InputContainer::value_type InputT;
    typedef typename OutputContainer::value_type OutputT;
    {
        sycl::buffer<InputT> in_buf(input.data(), input.size());
        sycl::buffer<OutputT> out_buf(output.data(), output.size());

        q.submit([&](sycl::handler& cgh) {
            sycl::accessor in(in_buf, cgh, sycl::read_only);
            sycl::accessor out(out_buf, cgh, sycl::write_only);
            constexpr auto temp_memory_size = sycl_exp::memory_required<InputT, items_per_thread>(
                sycl::memory_scope::work_group, block_size);
            sycl::local_accessor<std::byte> buf(temp_memory_size, cgh);
            cgh.parallel_for(
                sycl::nd_range<1>(block_count * block_size, block_size),
                [=](sycl::nd_item<1> item) {
                    InputT data[items_per_thread];
                    std::byte* buf_ptr = buf.get_pointer().get();
                    sycl_exp::group_with_scratchpad gh{ item.get_group(),
                                                        sycl::span{ buf_ptr, temp_memory_size } };

                    sycl_exp::joint_load(gh, in.get_pointer(), sycl::span{ data });

                    for (int i = 0; i < items_per_thread; ++i) {
                        data[i] += item.get_global_linear_id() * 100000 + i * 1000;
                    }

                    sycl_exp::joint_store(gh, out.get_pointer(), sycl::span{ data });
                });
        });
    }
}

template <typename InputContainer, typename OutputContainer>
void test_group_local_memory(sycl::queue q, InputContainer& input, OutputContainer& output) {
    typedef typename InputContainer::value_type InputT;
    typedef typename OutputContainer::value_type OutputT;
    {
        sycl::buffer<InputT> in_buf(input.data(), input.size());
        sycl::buffer<OutputT> out_buf(output.data(), output.size());

        q.submit([&](sycl::handler& cgh) {
            sycl::accessor in(in_buf, cgh, sycl::read_only);
            sycl::accessor out(out_buf, cgh, sycl::write_only);
            constexpr auto temp_memory_size = sycl_exp::memory_required<InputT, items_per_thread>(
                sycl::memory_scope::work_group, block_size);
            cgh.parallel_for(
                sycl::nd_range<1>(block_count * block_size, block_size),
                [=](sycl::nd_item<1> item) {
                    InputT data[items_per_thread];
                    auto scratch =
                        sycl::ext::oneapi::group_local_memory<std::byte[temp_memory_size]>(
                            item.get_group());
                    std::byte* buf_ptr = (std::byte*)(scratch.get());

                    sycl_exp::group_with_scratchpad gh{ item.get_group(),
                                                        sycl::span{ buf_ptr, temp_memory_size } };

                    sycl_exp::joint_load(gh, in.get_pointer(), sycl::span{ data });

                    for (int i = 0; i < items_per_thread; ++i) {
                        data[i] += item.get_global_linear_id() * 100000 + i * 1000;
                    }

                    sycl_exp::joint_store(gh, out.get_pointer(), sycl::span{ data });
                });
        });
    }
}

template <typename InputContainer, typename OutputContainer>
void test_marray(sycl::queue q, InputContainer& input, OutputContainer& output) {
    typedef typename InputContainer::value_type InputT;
    typedef typename OutputContainer::value_type OutputT;
    {
        sycl::buffer<InputT> in_buf(input.data(), input.size());
        sycl::buffer<OutputT> out_buf(output.data(), output.size());

        q.submit([&](sycl::handler& cgh) {
            sycl::accessor in(in_buf, cgh, sycl::read_only);
            sycl::accessor out(out_buf, cgh, sycl::write_only);
            cgh.parallel_for(
                sycl::nd_range<1>(block_count * block_size, block_size),
                [=](sycl::nd_item<1> item) {
                    sycl::marray<InputT, items_per_thread> data;

                    sycl_exp::joint_load(
                        item.get_group(), in.get_pointer(),
                        sycl::span<InputT, items_per_thread>{ data.begin(), data.end() });

                    for (int i = 0; i < items_per_thread; ++i) {
                        data[i] += item.get_global_linear_id() * 100000 + i * 1000;
                    }

                    sycl_exp::joint_store(
                        item.get_group(), out.get_pointer(),
                        sycl::span<InputT, items_per_thread>{ data.begin(), data.end() });
                });
        });
    }
}

template <typename InputContainer, typename OutputContainer>
void test_vec(sycl::queue q, InputContainer& input, OutputContainer& output) {
    typedef typename InputContainer::value_type InputT;
    typedef typename OutputContainer::value_type OutputT;
    {
        sycl::buffer<InputT> in_buf(input.data(), input.size());
        sycl::buffer<OutputT> out_buf(output.data(), output.size());

        q.submit([&](sycl::handler& cgh) {
            sycl::accessor in(in_buf, cgh, sycl::read_only);
            sycl::accessor out(out_buf, cgh, sycl::write_only);
            cgh.parallel_for(sycl::nd_range<1>(block_count * block_size, block_size),
                             [=](sycl::nd_item<1> item) {
                                 sycl::vec<InputT, items_per_thread> data;

                                 sycl_exp::joint_load(item.get_group(), in.get_pointer(),
                                                      sycl::span<InputT, items_per_thread>{
                                                          &data[0], &data[0] + items_per_thread });

                                 for (int i = 0; i < items_per_thread; ++i) {
                                     data[i] += item.get_global_linear_id() * 100000 + i * 1000;
                                 }

                                 sycl_exp::joint_store(item.get_group(), out.get_pointer(),
                                                       sycl::span<InputT, items_per_thread>{
                                                           &data[0], &data[0] + items_per_thread });
                             });
        });
    }
}

template <typename InputContainer, typename OutputContainer>
int check_correctness(InputContainer& input, OutputContainer& output, std::string test_name) {
    for (int i = 0; i < input.size() / items_per_thread; i++) {
        for (int j = 0; j < items_per_thread; j++) {
            int idx = i * items_per_thread + j;
            if ((input[idx] + i * 100000 + j * 1000) != output[idx]) {
                std::cout << i << " " << input[idx] << " " << output[idx] << std::endl;
                std::cout << test_name << " test failed" << std::endl;
                return 1;
            }
        }
    }
    std::cout << test_name << " test passed" << std::endl;
    return 0;
}

int main() {
    // TODO: turn test on when functionality will be implemented
    /*
    sycl::queue q;

    std::vector<int> input(size);
    std::vector<int> output(size);
    std::iota(input.begin(), input.end(), 0);
    std::fill(output.begin(), output.end(), 0);

    test_no_mem(q, input, output);
    assert(check_correctness(input, output, "No local memory") == 0);

    std::iota(input.begin(), input.end(), 0);
    std::fill(output.begin(), output.end(), 0);

    test_local_acc(q, input, output);
    assert(check_correctness(input, output, "Local accessor") == 0);

    std::iota(input.begin(), input.end(), 0);
    std::fill(output.begin(), output.end(), 0);

    test_group_local_memory(q, input, output);
    assert(check_correctness(input, output, "Group local memory") == 0);

    std::iota(input.begin(), input.end(), 0);
    std::fill(output.begin(), output.end(), 0);

    test_marray(q, input, output);
    assert(check_correctness(input, output, "sycl::marray") == 0);

    std::iota(input.begin(), input.end(), 0);
    std::fill(output.begin(), output.end(), 0);

    test_vec(q, input, output);
    assert(check_correctness(input, output, "sycl::vec") == 0);

    std::cout << "All tests passed" << std::endl;
    */
}
