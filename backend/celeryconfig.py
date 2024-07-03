from kombu import Exchange, Queue

broker_url = 'redis://localhost:6379/0'
result_backend = 'redis://localhost:6379/0'

task_queues = (
    Queue('default', Exchange('default'), routing_key='default'),
    Queue('failed', Exchange('failed'), routing_key='failed'),  # 配置失败队列
)

task_routes = {
    '*': {'queue': 'default'},
}

task_default_queue = 'default'
task_default_exchange = 'default'
task_default_routing_key = 'default'

# 全局重试策略
task_annotations = {
    '*': {'max_retries': 2},
}
