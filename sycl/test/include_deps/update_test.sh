HEADERS=(
    sycl/detail/defines_elementary.hpp
    sycl/detail/export.hpp

    sycl/buffer.hpp
    sycl/accessor.hpp

    sycl/detail/core.hpp

    sycl/khr/includes/accessor
    sycl/khr/includes/atomic
    sycl/khr/includes/backend
    sycl/khr/includes/bit
    sycl/khr/includes/buffer
    sycl/khr/includes/byte
    sycl/khr/includes/context
    sycl/khr/includes/device
    sycl/khr/includes/event
    sycl/khr/includes/exception
    sycl/khr/includes/functional
    sycl/khr/includes/groups
    sycl/khr/includes/half
    sycl/khr/includes/handler
    sycl/khr/includes/hierarchical_parallelism
    sycl/khr/includes/image
    sycl/khr/includes/index_space
    sycl/khr/includes/interop_handle
    sycl/khr/includes/kernel_bundle
    sycl/khr/includes/marray
    sycl/khr/includes/math
    sycl/khr/includes/multi_ptr
    sycl/khr/includes/platform
    sycl/khr/includes/property_list
    sycl/khr/includes/queue
    sycl/khr/includes/reduction
    sycl/khr/includes/span
    sycl/khr/includes/stream
    sycl/khr/includes/type_traits
    sycl/khr/includes/usm
    sycl/khr/includes/vec
    sycl/khr/includes/version
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
