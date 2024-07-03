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
    #
    # Also, <detail/plugins/.*/features.hpp> is dependent on what plugins were
    # enabled, so ignore them.

    clang++ -fsycl -fsycl-device-only -include "$HEADER" -c -x c++ /dev/null -o /dev/null  -MD -MF - \
        | sed 's@: /dev/null@: /dev/null\n@' \
        | grep 'include/sycl\|/dev/null\|:' \
        | grep -v 'detail/plugins/.*/features.hpp' \
        | sed 's@.*/include/sycl/@@' \
        | sed 's/ \\//'
}

deps $1
echo ""
