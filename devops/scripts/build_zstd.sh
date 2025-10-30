#!/bin/bash

# Script to build and install zstd on Ubuntu 24, with -fPIC flag.
# The default installation of zstd on Ubuntu 24 does not have -fPIC flag
# enabled, which is required for building DPC++ in shared libraries mode.

# OR on Rocky Linux 8.10 (used for nightly release builds). There is no static
# library (libzstd.a) in available packages, therefore it is necessary to build
# it from source.

# Function to check OS
check_os() {
    local expected_name="$1"
    local expected_version="$2"
    . /etc/os-release
    if [[ "$NAME" == "$expected_name" && "$VERSION_ID" == "$expected_version" ]]; then
        return 0
    else
        return 1
    fi
}

# Function to install packages with or without sudo
install_packages() {
    if [ "$USE_SUDO" = true ]; then
        sudo apt-get update
        sudo apt-get install -y build-essential wget
    else
        apt-get update
        apt-get install -y build-essential wget
    fi
}

# Function to uninstall libzstd-dev if installed
uninstall_libzstd_dev() {
    if dpkg -l | grep -q libzstd-dev; then
        if [ "$USE_SUDO" = true ]; then
            sudo apt-get remove -y libzstd-dev
        else
            apt-get remove -y libzstd-dev
        fi
    fi
}

# Function to build a shared library by linking zstd static lib.
# This is used to verify that zstd is built correctly, with -fPIC flag.
build_test_program() {
    cat <<EOF > test_zstd.c
      #include <zstd.h>
      int main() {
        ZSTD_CCtx* cctx = ZSTD_createCCtx();
        ZSTD_freeCCtx(cctx);
        return 0;
      }
EOF

    # Try to use zstd's static library with -fPIC
    gcc test_zstd.c -lzstd -fPIC -shared
    if [ $? -ne 0 ]; then
        echo "zstd installation verification failed."
    else
        echo "zstd installation verification passed."
    fi

    # There won't be a.out file if verification failed.
    rm test_zstd.c a.out || true
}

# Check the OS
if ! check_os "Ubuntu" "24.04" && ! check_os "Rocky Linux" "8.10"; then
    echo "Warning: This script has only been tested with Ubuntu 24.04 and Rocky Linux 8.10."
fi

# Set USE_SUDO to true or false based on your preference
USE_SUDO=true

# Install necessary build tools & uninstall libzstd-dev package if installed
if check_os "Ubuntu" "24.04"; then
    install_packages
    uninstall_libzstd_dev
fi

# Define the version and URL for zstd
ZSTD_VERSION="1.5.7"
ZSTD_URL="https://github.com/facebook/zstd/releases/download/v$ZSTD_VERSION/zstd-$ZSTD_VERSION.tar.gz"

# Create a directory for the source code
mkdir -p zstd_build
cd zstd_build

# Download and extract zstd source code
wget $ZSTD_URL
tar -xzf zstd-$ZSTD_VERSION.tar.gz
cd zstd-$ZSTD_VERSION

# Build zstd with -fPIC flag.
CFLAGS="-fPIC" CXXFLAGS="-fPIC" make
if [ $? -ne 0 ]; then
    echo "Error: make failed."
    exit 1
fi

# Install zstd.
if [ "$USE_SUDO" = true ]; then
    sudo make install
else
    make install
fi
if [ $? -ne 0 ]; then
    echo "Error: make install failed."
    exit 1
fi

# Verify zstd installation.
build_test_program

# Clean up
rm -rf zstd_build
