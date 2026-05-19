HEADERS=(
    sycl/detail/defines_elementary.hpp
    sycl/detail/export.hpp

    sycl/buffer.hpp
    sycl/accessor.hpp

    sycl/detail/core.hpp

    sycl/khr/split_headers/accessor.hpp
    sycl/khr/split_headers/atomic.hpp
    # backend header depends on how the project was configured and as such it
    # is not exactly portable, so it is excluded
    # sycl/khr/split_headers/backend
    sycl/khr/split_headers/bit.hpp
    sycl/khr/split_headers/buffer.hpp
    sycl/khr/split_headers/byte.hpp
    sycl/khr/split_headers/builtins_common.hpp
    sycl/khr/split_headers/builtins_geometric.hpp
    sycl/khr/split_headers/builtins_integer.hpp
    sycl/khr/split_headers/builtins_math.hpp
    sycl/khr/split_headers/builtins_relational.hpp
    sycl/khr/split_headers/context.hpp
    sycl/khr/split_headers/device.hpp
    sycl/khr/split_headers/event.hpp
    sycl/khr/split_headers/exception.hpp
    sycl/khr/split_headers/functional.hpp
    sycl/khr/split_headers/group_algorithms.hpp
    sycl/khr/split_headers/groups.hpp
    sycl/khr/split_headers/half.hpp
    sycl/khr/split_headers/handler.hpp
    sycl/khr/split_headers/hierarchical_parallelism.hpp
    sycl/khr/split_headers/images.hpp
    sycl/khr/split_headers/index_space.hpp
    sycl/khr/split_headers/interop_handle.hpp
    sycl/khr/split_headers/kernel_bundle.hpp
    sycl/khr/split_headers/kernel_handler.hpp
    sycl/khr/split_headers/marray.hpp
    sycl/khr/split_headers/math.hpp
    sycl/khr/split_headers/multi_ptr.hpp
    sycl/khr/split_headers/platform.hpp
    sycl/khr/split_headers/property_list.hpp
    sycl/khr/split_headers/queue.hpp
    sycl/khr/split_headers/reduction.hpp
    sycl/khr/split_headers/span.hpp
    sycl/khr/split_headers/stream.hpp
    sycl/khr/split_headers/type_traits.hpp
    sycl/khr/split_headers/usm.hpp
    sycl/khr/split_headers/vec.hpp
    sycl/khr/split_headers/version.hpp
)

for x in ${HEADERS[@]} ; do
    name="$(echo $x | sed 's@/@_@g').cpp"
    echo -e "// Use update_test.sh to (re-)generate the checks" > $name
    echo -e "// REQUIRES: linux" >> $name
    echo -e "// RUN: bash %S/deps_known.sh $x | FileCheck %s\n" >> $name
    bash deps_known.sh $x | \
        sed 's@^@// CHECK-NEXT: @' | \
        sed 's@CHECK-NEXT: Dependencies@CHECK-LABEL: Dependencies@' | \
        sed 's@CHECK-NEXT: $@CHECK-EMPTY:@' >> $name
done
