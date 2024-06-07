HEADERS=(
    sycl/detail/defines_elementary.hpp
    sycl/detail/export.hpp

    sycl/buffer.hpp
    sycl/accessor.hpp

    sycl/detail/core.hpp
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
