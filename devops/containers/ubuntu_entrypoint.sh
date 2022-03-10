#!/bin/bash

if [ -d "$GITHUB_WORKSPACE" ]; then
  chown -R sycl:sycl $GITHUB_WORKSPACE
  su sycl
fi

exec "$@"
