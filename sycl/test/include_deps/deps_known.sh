function deps() {
    HEADER=$1
    echo "Dependencies for <$HEADER>:"

    # Format is something like this:
    #
    # $ clang++ -fsycl -fsycl-device-only -include "detail/defines_elementary.hpp" -c -x c++ /dev/null -MD -MF -
    # /dev/null: /dev/null \
    #       /localdisk2/aeloviko/sycl/build/bin/../include/sycl/detail/defines_elementary.hpp
    #
    # However, sometimes first header is on the same line with
    # "null.o: /dev/null <header>", so add an explicit line break there.

    clang++ -fsycl -fsycl-device-only -include "$HEADER" -c -x c++ /dev/null -o /dev/null  -MD -MF - \
        | sed 's@: /dev/null@: /dev/null\n@' \
        | grep 'include/sycl\|/dev/null\|CL/\|ur_\|:' \
        | sed 's@.*/include/sycl/@@' \
        | sed 's@.*/include/CL/@CL/@' \
        | sed 's@.*/include/ur_@ur_@' \
        | sed 's/ \\//'
}

deps $1
echo ""
