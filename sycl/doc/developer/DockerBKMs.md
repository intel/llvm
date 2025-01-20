# Docker Containers BKMs

## Installation

Follow the [official guide](https://docs.docker.com/engine/install/) for your
OS or distro.

### Change location of the images on the system

By default Docker stores images in `/var/lib/docker`, but that can be changed.

Create a new file called `/etc/docker/daemon.json` and put the following
content:

```
{
  "data-root": "/path/to/data/root",
  "exec-root": "/path/to/exec/root"
}
```

### Running Docker without sudo

Add your local user to `docker` group to be able to run docker commands without
sudo.


### Docker vs Podman

Docker and Podman are very similar tools, that allow you to manage and run
container images. Unlike Docker, Podman runs without a daemon and allows you
to run containers without root permissions. The command line syntax is mostly
identical for Docker and Podman. Choose whatever is available on your system.

## SYCL Dockerfiles overview

The following Dockerfiles are publicly available for creating DPC++ compiler
development containers:

### Ubuntu 22.04-based Dockerfiles

- `devops/containers/ubuntu2204_base`: contains basic environment
   setup for building DPC++ compiler from source.
- `devops/containers/ubuntu2204_intel_drivers`: contains everything from the
  base Dockerfile + pre-installed Intel drivers.
   The Dockerfile comes in two flavors/tags:
   * `latest`: Intel drivers are downloaded from release/tag and saved in
    dependencies.json. The drivers are tested/validated everytime we upgrade
    the driver.
   * `alldeps`: Includes the same Intel drivers as `latest`, as well as the
   development kits for NVidia/AMD from the `ubuntu2204_build` Dockerfile.
- `devops/containers/ubuntu2204_build`: has development kits installed for
   NVidia/AMD and can be used for building DPC++
   compiler from source with all backends enabled or for end-to-end testing
   with HIP/CUDA on machines with corresponding GPUs available.

### Ubuntu 24.04-based Dockerfiles

- `devops/containers/ubuntu2404_base`: contains basic environment
   setup for building DPC++ compiler from source.
- `devops/containers/ubuntu2404_intel_drivers`: contains everything from the
   base Dockerfile + pre-installed Intel drivers.
   The Dockerfile comes in four flavors/tags:
   * `latest`: Intel drivers are downloaded from release/tag and saved in
    dependencies.json. The drivers are tested/validated everytime we upgrade
    the driver.
   * `devigc`: Intel Graphics Compiler driver from github actions artifacts,
   other drivers are downloaded from release/tag and saved in dependencies.json.
   * `unstable`: Intel drivers are downloaded from release/latest.
   * `alldeps`: Includes the same Intel drivers as `latest`, as well as the
   development kits for NVidia/AMD from the `ubuntu2404_build` Dockerfile.
   The drivers are installed as it is, not tested or validated.
- `devops/containers/ubuntu2404_build`: has development kits installed for
   NVidia/AMD and can be used for building DPC++
   compiler from source with all backends enabled or for end-to-end testing
   with HIP/CUDA on machines with corresponding GPUs available.  
 - `devops/containers/nightly`: contains the latest successfully
   built nightly build of DPC++ compiler.

## Running Docker container interactively

The main application of Docker is containerizing services. But it also allows
you to run containers interactively, so that you can use them as you would use a
terminal or SSH session. The following command allows you to do that:

```
docker run --name <friendly_name> -it <image_name>[:<tag>] /bin/bash
```

This command will download an image if it does not exist locally. If you've
downloaded an image previously, and you want to update it, use
`docker pull <image_name>` command.

## Persisting data

### Persisting data with volumes

Docker container images are read-only. When container is destroyed, all its data
is lost. To persist data when working with containers (i.e. when upgrading
container version) one can use Docker volumes.

Creating a volume:

```
docker volume create <volume name>
```

Listing all volumes:

```
docker volume list
```

Mounting volume to the container:

```
docker run <options> -v <volume_name>:/path/inside/container <image_name> bash
```

Deleting a volume:

```
docker volume rm <image_name>
```

See [official documentation](https://docs.docker.com/storage/volumes/) for more
info.

### Passthrough a directory to a container

Use `-v path/on/host:path/in/container` argument for `run` command to
passthrough a host directory or a file.

## GPU passthrough

### Intel

Add `--device=/dev/dri` argument to `run` command to passthrough you Intel GPU.
Make sure you're a member of `video` group to be able to access GPU.

In case the container is running under WSL, add `--device=/dev/dxg -v /usr/lib/wsl:/usr/lib/wsl` 
argument to `run` command. See [official guide](https://github.com/microsoft/wslg/blob/main/samples/container/Containers.md#containerized-applications-access-to-the-vgpu) 
for more information.

### AMD

Follow the [official guide](https://rocmdocs.amd.com/en/latest/ROCm_Virtualization_Containers/ROCm-Virtualization-&-Containers.html).

### Nvidia

Follow [these](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/user-guide.html)
instructions.

## Changing Docker user

By default all processes within our containers are run as the `sycl` user which
has password-less `sudo` access.

If you want to change the user, you can do that by specifying the
`-u <username or uid>` option when running the container.

## Managing downloaded Docker images

List local images:
```
docker image ls
```

Remove local image:
```
docker image rm <image_name_or_id>
```

## Managing disk usage

See how much space is taken by docker:

```
docker system df
```

Cleaning unused data:

```
docker system prune
```

See [official documentation](https://docs.docker.com/engine/reference/commandline/system_prune/)
for more info.

## Building a Docker Container from scratch

Docker containers can be built with the following command:

```
docker build -f path/to/devops/containers/file.Dockerfile path/to/devops/
```

The `nightly.Dockerfile` script expects `llvm_sycl.tar.xz` file
to be present in `devops/` directory.

Containers other than base provide several configurable arguments, the most
commonly used are `base_image` and `base_tag`, which specify the base Docker
image and its tag. You can set additional arguments with `--build-arg ARG=value`
argument.

