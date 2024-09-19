# Content

Dockerfiles and scripts placed in this directory are intended to be used as
development process vehicles and part of continuous integration process.

Images built out of those recipes may be used with Docker or podman as
development environment.

# How to build docker image

To build docker image on local machine, enter the root dir of the repository and execute:

```sh
docker build -t ur:ubuntu-22.04 -f .github/docker/ubuntu-22.04.Dockerfile .
```

To set any build time variable (e.g., an optional ARG from docker recipe), add to the command (after `build`), e.g.:

```sh
 --build-arg SKIP_DPCPP_BUILD=1
```

One other example of using these extra build arguments are proxy settings. They are required for accessing network
(e.g., to download dependencies within docker), if a host is using a proxy server. Example usage:

```sh
 --build-arg https_proxy=http://proxy.com:port --build-arg http_proxy=http://proxy.com:port 
```

# How to use docker image

To run docker container (using the previously built image) execute:

```sh
docker run --shm-size=4G -v /your/workspace/path/:/opt/workspace:z -w /opt/workspace/ -it ur:ubuntu-22.04 /bin/bash
```

To set (or override) any docker environment variable, add to the command (after `run`):

```sh
 -e ENV_VARIABLE=VALUE
```
