#!/bin/bash

if [ -d "$GITHUB_WORKSPACE" ]; then
  sudo chown -R sycl:sycl $GITHUB_WORKSPACE
fi

exec "$@"
