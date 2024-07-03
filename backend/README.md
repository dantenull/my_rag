# backend

## docker

``` shell
cd /app/src
docker build -f Dockerfile.base -t python-3.10-base .
docker build -t my_rag_backend .
cd /app/frontend
docker build -t my_rag_frontend .
cd /app
docker-compose up -d 
```

### 问题

```txt
elastic_transport.ConnectionError: Connection error caused by: ConnectionError(Connection error caused by: NewConnectionError(<elastic_transport._node._urllib3_chain_certs.HTTPSConnection object at 0x7f5aa870a740>: Failed to establish a new connection: [Errno 111] Connection refused))
```

连接elasticsearch时使用`https://es01:9200`而不是`https://localhost:9200`，`https://es01:9200`是在docker compose文件中设置好的。
