HEADERS=(
    sycl/detail/defines_elementary.hpp
    sycl/detail/export.hpp

    sycl/buffer.hpp
    sycl/accessor.hpp

    sycl/detail/core.hpp

    sycl/khr/includes/accessor.hpp
    sycl/khr/includes/atomic.hpp
    # backend header depends on how the project was configured and as such it
    # is not exactly portable, so it is excluded
    # sycl/khr/includes/backend
    sycl/khr/includes/bit.hpp
    sycl/khr/includes/buffer.hpp
    sycl/khr/includes/byte.hpp
    sycl/khr/includes/context.hpp
    sycl/khr/includes/device.hpp
    sycl/khr/includes/event.hpp
    sycl/khr/includes/exception.hpp
    sycl/khr/includes/functional.hpp
    sycl/khr/includes/group_algorithms.hpp
    sycl/khr/includes/groups.hpp
    sycl/khr/includes/half.hpp
    sycl/khr/includes/handler.hpp
    sycl/khr/includes/hierarchical_parallelism.hpp
    sycl/khr/includes/images.hpp
    sycl/khr/includes/index_space.hpp
    sycl/khr/includes/interop_handle.hpp
    sycl/khr/includes/kernel_bundle.hpp
    sycl/khr/includes/kernel_handler.hpp
    sycl/khr/includes/marray.hpp
    sycl/khr/includes/math.hpp
    sycl/khr/includes/multi_ptr.hpp
    sycl/khr/includes/platform.hpp
    sycl/khr/includes/property_list.hpp
    sycl/khr/includes/queue.hpp
    sycl/khr/includes/reduction.hpp
    sycl/khr/includes/span.hpp
    sycl/khr/includes/stream.hpp
    sycl/khr/includes/type_traits.hpp
    sycl/khr/includes/usm.hpp
    sycl/khr/includes/vec.hpp
    sycl/khr/includes/version.hpp
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
