# Hướng dẫn chạy Milvus và Neo4j

Có 2 cách để chạy Milvus và Neo4j: **Docker Compose** (khuyến nghị) hoặc **cài đặt trực tiếp**.

## Cách 1: Docker Compose (Khuyến nghị)

### Bước 1: Tạo Docker network

Milvus cần một network tên `milvus`:

```bash
docker network create milvus
```

### Bước 2: Chạy tất cả services

Vào thư mục `docker` và chạy docker-compose:

```bash
cd docker
docker compose up -d
```

Lệnh này sẽ chạy:
- **Neo4j**: Port 7474 (Web UI), 7687 (Bolt)
- **Milvus**: Port 19530 (gRPC), 9091 (Metrics)
- **etcd**: Metadata store cho Milvus
- **MinIO**: Object storage cho Milvus (Port 9000, 9001)
- **Attu**: Milvus Web UI (Port 8000)

### Bước 3: Kiểm tra services đã chạy

```bash
docker compose ps
```

Hoặc kiểm tra từng service:

```bash
# Neo4j
curl http://localhost:7474

# Milvus
curl http://localhost:9091/healthz
```

### Bước 4: Truy cập Web UIs

- **Neo4j Browser**: http://localhost:7474
  - Username: `neo4j`
  - Password: `password123`

- **Milvus Attu UI**: http://localhost:8000
  - Kết nối đến Milvus tự động

- **MinIO Console**: http://localhost:9001
  - Username: `minioadmin`
  - Password: `minioadmin`

### Dừng services

```bash
docker compose down
```

**Lưu ý**: Lệnh này sẽ dừng containers nhưng giữ lại data.

### Xóa tất cả data (nếu cần)

```bash
docker compose down -v
```

## Cách 2: Chạy riêng từng service

### Chạy Neo4j

```bash
docker run -d \
  --name neo4j-local \
  -p 7474:7474 \
  -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password123 \
  neo4j:5.21
```

### Chạy Milvus (Standalone)

Milvus cần etcd và MinIO. Khuyến nghị dùng docker-compose.

Hoặc dùng Milvus Lite (standalone, không cần etcd/MinIO):

```bash
pip install milvus
milvus start
```

## Kiểm tra kết nối

### Test Neo4j

```bash
# Test connection
docker exec -it neo4j-local cypher-shell -u neo4j -p password123

# Hoặc từ Python
python -c "from neo4j import GraphDatabase; driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password123')); driver.verify_connectivity(); print('Connected!')"
```

### Test Milvus

```bash
# Test connection
python -c "from pymilvus import connections; connections.connect('default', host='localhost', port='19530'); print('Connected!')"
```

## Environment Variables

Nếu muốn đổi ports hoặc credentials, set các biến môi trường:

```bash
# Neo4j
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=password123

# Milvus
export MILVUS_URI=http://localhost:19530
export MILVUS_ALIAS=default
export MILVUS_DB_NAME=default
```

## Troubleshooting

### Lỗi: Network "milvus" not found

```bash
docker network create milvus
```

### Lỗi: Port đã được sử dụng

Kiểm tra port nào đang được dùng:

```bash
# Windows
netstat -ano | findstr :19530
netstat -ano | findstr :7687

# Linux/Mac
lsof -i :19530
lsof -i :7687
```

### Lỗi: Neo4j không kết nối được

1. Kiểm tra container đang chạy:
   ```bash
   docker ps | grep neo4j
   ```

2. Kiểm tra logs:
   ```bash
   docker logs neo4j-local
   ```

3. Reset password nếu quên:
   ```bash
   docker exec -it neo4j-local cypher-shell -u neo4j -p neo4j
   # Sau đó đổi password trong cypher-shell
   ```

### Lỗi: Milvus không kết nối được

1. Kiểm tra tất cả containers đang chạy:
   ```bash
   docker compose ps
   ```

2. Kiểm tra logs:
   ```bash
   docker logs milvus-standalone
   docker logs milvus-etcd
   docker logs milvus-minio
   ```

3. Đảm bảo network `milvus` đã được tạo

## Sau khi chạy services

1. **Tạo mock data**:
   ```bash
   python examples/create_mock_data.py
   ```

2. **Test RAG Agent**:
   ```bash
   python examples/test_rag_simple.py
   ```

## Ports được sử dụng

| Service | Port | Mô tả |
|---------|------|-------|
| Neo4j Browser | 7474 | Web UI cho Neo4j |
| Neo4j Bolt | 7687 | Connection port cho drivers |
| Milvus | 19530 | gRPC port |
| Milvus Metrics | 9091 | Health check port |
| Milvus Attu | 8000 | Web UI cho Milvus |
| MinIO API | 9000 | Object storage API |
| MinIO Console | 9001 | MinIO Web UI |



