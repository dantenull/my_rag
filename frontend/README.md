# frontend

This template should help get you started developing with Vue 3 in Vite.

## Recommended IDE Setup

[VSCode](https://code.visualstudio.com/) + [Volar](https://marketplace.visualstudio.com/items?itemName=Vue.volar) (and disable Vetur).

## Customize configuration

See [Vite Configuration Reference](https://vitejs.dev/config/).

## Project Setup

```sh
npm install
```

### Compile and Hot-Reload for Development

```sh
npm run dev
```

### Compile and Minify for Production

```sh
npm run build
```

## docker

```powershell
docker run --name frontend_test -dp 5173:5173 -v ${pwd}/src:/app/src my_rag_frontend
```

### vite config

添加以下配置以保证可以连接到容器中的程序：

```js
  server: {
    host: '0.0.0.0'
  }
```
