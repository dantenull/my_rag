
# 说明

这是个个人的学习的AI项目，很不成熟，未经过完整的测试。
主要是学习了下llama index和langchain等框架。
运用的优化方法包括query改写、embedding模型、rerank模型、混合检索等，适配了openai和智谱ai的api接口，也可使用本地LLM。
使用的 fastapi、celery、mongodb、chromadb、elasticsearch 等技术。

# 遗留问题

## elasticsearch

1. 设置ELASTIC_PASSWORD环境变量失败，不得不直接进入容器查看密码。
2. 没有安装所需插件。
