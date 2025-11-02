# Milvus Stack (Docker)

This folder provides a Docker Compose configuration for running a local Milvus
stack (standalone Milvus, MinIO, etcd and Attu). Use it to develop against the
vector store, experiment with the Attu UI, or populate/inspect sample data.

## Prerequisites

- Docker Desktop or Docker Engine with Compose v2 (`docker compose`) available.
- A dedicated Docker network named `milvus` (the Compose file expects it to
  exist because it is marked as `external: true`):

  ```bash
  docker network create milvus
  ```

- (Optional) Set `DOCKER_VOLUME_DIRECTORY` to an absolute path if you do not
  want volumes to live under this repository’s `docker/volumes` directory.

## Directory layout

```
docker/
├─ docker-compose.yaml   # Service definitions
└─ volumes/              # Default bind-mount location for etcd, MinIO, Milvus
```

If Docker creates the `volumes/` folders as `root`, fix permissions so the repo
remains readable:

```bash
sudo chown -R "$USER":"$USER" docker/volumes
```

## Services

| Service     | Image                     | Purpose                                  | Exposed ports |
|-------------|---------------------------|------------------------------------------|---------------|
| `etcd`      | `quay.io/coreos/etcd`     | Metadata store required by Milvus        | —             |
| `minio`     | `minio/minio`             | Object storage backend for Milvus        | 9000 (API), 9001 (console) |
| `standalone`| `milvusdb/milvus`         | Milvus server (standalone mode)          | 19530 (gRPC), 9091 (metrics) |
| `attu`      | `zilliz/attu`             | Web UI for browsing Milvus collections   | 8000 (mapped to 3000) |

## Running the stack

```bash
cd docker
docker compose up -d     # Start or update the stack
docker compose ps        # Check container status
```

- Milvus server: `localhost:19530`
- Milvus metrics: `http://localhost:9091/metrics`
- MinIO console:  `http://localhost:9001` (user/pass `minioadmin`)
- Attu UI:        `http://localhost:8000`

To stop the stack but keep data:

```bash
docker compose down
```

To stop and remove persisted data, also delete the bind mount directory:

```bash
docker compose down
rm -rf docker/volumes/*
```

> **Note:** Removing volumes erases all collections and MinIO buckets.

