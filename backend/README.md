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

### 遗留问题

#### elasticsearch

1. 设置ELASTIC_PASSWORD环境变量失败，不得不直接进入容器查看密码。
2. 没有安装所需插件。
