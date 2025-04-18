#!/bin/sh
if [ -z $1 ]; then
    echo "Usage: $0 builddir" >&2;
    exit;
fi

which hardening-check >> /dev/null;
if [ $? != "0" ]; then
    echo "hardening-check not found - on Ubuntu it is from the 'devscripts' package." >&2;
    exit;
fi

RET=0;

for file in $1/bin/*; do
    case "$file" in
        */urtrace)
            # This is a python script
            true;;
        *)
            hardening-check -q --nocfprotection --nofortify $file;;
    esac
    RET=$(($RET + $?))
done;

for file in $1/lib/*.so; do
    case "$file" in
        */libOpenCL*)
            # This is not built as part of UR
            true;;
        */libzeCallMap.so | */libur_mock_headers.so)
            # Only used in testing, and are too simple for many of the hardening flags to have an effect.
            true;;
        *)
            hardening-check -q --nocfprotection --nofortify $file;;
    esac
    RET=$(($RET + $?))
done;

if [ $RET != "0" ]; then
    exit 1;
fi
