# Docker Containers BKMs

## Docker vs Podman

Docker and Podman are very similar tools, that allow you to manage and run
container images. Unlike Docker, Podman runs without a daemon, allows you to run
containers without root permissions, but does not let you build a container from
source. The command line syntax is mostly identical for Docker and Podman.
Choose whatever is available on your system.

## SYCL Containers overview

The following containers are publicly available for DPC++ compiler development:

- `ghcr.io/intel/llvm/ubuntu2004_base`: contains basic environment setup for
   building DPC++ compiler from source.
- `ghcr.io/intel/llvm/ubuntu2004_intel_drivers`: contains everything from base
   container + pre-installed Intel drivers. This image provides two main tags:
   `latest`, that uses latest available drivers, and `stable`, that uses
   recommended drivers.
- `ghcr.io/intel/llvm/ubuntu2004_nightly`: contains latest successfully built
   nightly build of DPC++ compiler, as well as pre-installed Intel drivers.

## Building a Docker Container from scratch

Docker containers can be build with the following command:

```
docker build -f path/to/devops/containers/file.Dockerfile path/to/devops/
```

The `ubuntu2004_preinstalled.Dockerfile` script expects `llvm_sycl.tar.gz` file
to be present in `devops/` directory.

Containers other than base provide several configurable arguments, most commonly
used are`base_image` and `base_tag`, that specify base Docker image and its tag.
You can set additional arguments with `--build-arg ARG=value` argument.

## Running Docker container interactively

The main application of Docker is containerizing services. But it also allows
you to run containers interactively, so that you can use it as you would use a
terminal or SSH session. The following command allows you to do that:

```
docker run --name <friendly_name> -it --entrypoint /bin/bash <image_name>[:<tag>]
```

This command will download an image, if it does not exist locally. If you've
downloaded an image previously, and you want to update it, use
`docker pull <image_name>` command.

## Passthrough an Intel GPU to container

Add `--device=/dev/dri` argument to `run` command to passthrough you Intel GPU.

## Passthrough a directory to container

Use `-v path/on/host:path/in/container` argument for `run` command to
passthrough a host directory or a file.

## Managing downloaded Docker image

List local images:
```
docker image ls
```

Remove local image:
```
docker image rm <image_name_or_id>
```
