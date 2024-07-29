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
