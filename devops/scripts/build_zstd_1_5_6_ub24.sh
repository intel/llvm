#!/bin/bash

# Function to check if the OS is Ubuntu 24
check_os() {
    . /etc/os-release
    if [[ "$NAME" != "Ubuntu" || "$VERSION_ID" != "24.04" ]]; then
        echo "Warning: This script has only been tested with Ubuntu 24."
        exit 1
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

# Function to check if zstd is built correctly
check_zstd() {
    if command -v zstd &> /dev/null; then
        echo "zstd has been built and installed successfully!"
    else
        echo "Error: zstd was not built correctly."
        exit 1
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

# Function to build and run a simple test program
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
check_os

# Set USE_SUDO to true or false based on your preference
USE_SUDO=false

# Install necessary build tools
install_packages

# Uninstall libzstd-dev if installed
uninstall_libzstd_dev

# Define the version and URL for zstd
ZSTD_VERSION="1.5.6"
ZSTD_URL="https://github.com/facebook/zstd/releases/download/v$ZSTD_VERSION/zstd-$ZSTD_VERSION.tar.gz"

# Create a directory for the source code
mkdir -p zstd_build
cd zstd_build

# Download the source code
wget $ZSTD_URL

# Extract the tarball
tar -xzf zstd-$ZSTD_VERSION.tar.gz
cd zstd-$ZSTD_VERSION

# Build zstd with -fPIC flag for both static and dynamic libraries
CFLAGS="-fPIC" make
if [ $? -ne 0 ]; then
    echo "Error: make failed."
    exit 1
fi

# Optionally, install zstd
if [ "$USE_SUDO" = true ]; then
    sudo make install
else
    make install
fi
if [ $? -ne 0 ]; then
    echo "Error: make install failed."
    exit 1
fi

# Check if zstd is built correctly
check_zstd

# Build and run a simple test program
build_test_program

# Clean up
cd ~
rm -rf zstd_build
