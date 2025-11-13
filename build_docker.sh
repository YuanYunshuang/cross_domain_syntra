#!/usr/bin/env bash
# 1) Set your image name
export DH_USER="opheliayuan"
export IMAGE_NAME="syntra"
export IMAGE_TAG="0.1.0"
export IMAGE="${DH_USER}/${IMAGE_NAME}:${IMAGE_TAG}"

# 2) Build the image
DOCKER_BUILDKIT=1 docker build -t "${IMAGE}" -f Dockerfile .

# (optional) also tag as latest
docker tag "${IMAGE}" "${DH_USER}/${IMAGE_NAME}:latest"

# 3) Login to Docker Hub
docker login -u "${DH_USER}"
# enter your password or token

# 4) Push
docker push "${IMAGE}"
docker push "${DH_USER}/${IMAGE_NAME}:latest"

# 5) Verify
docker pull "${IMAGE}"
docker pull "${DH_USER}/${IMAGE_NAME}:latest"