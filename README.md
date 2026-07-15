# 客服投诉标签系统

基于 DeepSeek API 的智能投诉标签工具，自动从客服聊天记录中提取投诉关键词、评估用户情绪分数并生成解释。

## 项目背景

客服聊天记录中蕴含着大量高价值信息，传统人工标注方式成本高、效率低。本项目采用大模型 + Prompt 的方法，自动化完成文本标注，覆盖投诉风险识别、情绪评估等场景。

支持 42 万条文本在数小时内批量处理完毕，关键词识别准确率约 90%，情绪解读准确率约 80%。

## 功能特性

- **投诉关键词提取** — 自动识别 10 类投诉类型（退款、物流、服务态度等）
- **情绪评分（0~10）** — 从满意到极端愤怒的分级评估
- **情绪解读** — 对每条文本生成一句解释说明
- **批量处理** — 支持大规模 CSV 文件批处理
- **断点续跑** — 处理中断后可从中断点继续运行，不丢数据
- **自动重试** — API 调用失败自动指数退避重试
- **速率控制** — 可配置的请求频率限制
- **日志记录** — 详细的运行日志，便于排查问题

## 项目结构

```
customer-complaint-labeling/
├── config/
│   ├── config.yaml       # 全局配置文件
│   └── prompt.txt         # Prompt 模板
├── data/
│   ├── raw/               # 原始数据目录
│   ├── processed/         # 处理后数据目录
│   └── sample_chat.csv    # 测试数据集 (1000条)
├── logs/                  # 运行日志
├── output/                # 输出结果
├── src/
│   ├── __init__.py
│   ├── main.py            # 主入口
│   ├── config_loader.py   # 配置加载
│   ├── prompt_manager.py  # Prompt 管理
│   ├── api_client.py      # DeepSeek API 客户端
│   ├── batch_processor.py # 批量处理 (断点续跑)
│   └── utils.py           # 工具函数
├── tests/
│   ├── __init__.py
│   ├── test_config_loader.py
│   ├── test_api_client.py
│   └── test_batch_processor.py
├── .env                   # API Key (需自行配置)
├── .gitignore
├── requirements.txt
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

编辑 `.env` 文件，填入你的 DeepSeek API Key：

```
DEEPSEEK_API_KEY=sk-your-deepseek-api-key-here
```

### 3. 准备数据

将客服聊天记录保存为 CSV 格式，放在 `data/` 目录下。默认测试数据为 `data/sample_chat.csv`。

输入格式：

| id | text |
|----|------|
| 1  | 为什么一直没人处理我的退款？ |
| 2  | 谢谢客服，很满意。 |

### 4. 运行

```bash
# 使用默认配置
python -m src.main

# 指定输入输出文件
python -m src.main --input data/custom.csv --output output/custom_results.csv

# 调整批大小
python -m src.main --batch-size 10

# 指定日志级别
python -m src.main --log-level DEBUG
```

### 5. 查看结果

输出文件位于 `output/results.csv`，格式如下：

| id | text | keywords | sentiment_score | sentiment_reason |
|----|------|----------|----------------|-----------------|
| 1  | 为什么一直没人处理我的退款？ | 退款;投诉 | 8 | 用户对于退款速度非常不满 |
| 2  | 谢谢客服，很满意。 | 满意;感谢 | 0 | 用户表达满意 |

## 业务规则

### 投诉类型

| 类别 | 说明 |
|------|------|
| 退款/退货 | 退款审核慢、金额不对、流程复杂 |
| 物流/配送 | 配送慢、包裹破损、快递员态度差 |
| 服务态度 | 客服恶劣、排队久、机器人敷衍 |
| 产品质量 | 商品破损、色差、描述不符 |
| 虚假宣传 | 图片不符、夸大功能 |
| 霸王条款 | 退款门槛高、不退不换 |
| 欺诈/诈骗 | 重复扣款、虚假发货 |
| 价格问题 | 价格变动、多收费 |
| 会员/优惠 | 折扣不生效、优惠券不能用 |
| 订单问题 | 订单取消、发错货 |

### 情绪分定义

| 分数 | 含义 |
|------|------|
| 0 | 满意 - 表达感谢或满意 |
| 2 | 一般 - 中性咨询或普通反馈 |
| 5 | 有点不满 - 轻微抱怨 |
| 8 | 投诉 - 明显投诉或愤怒 |
| 10 | 极端愤怒 - 强烈投诉或威胁性言论 |

## 自定义配置

- **修改 Prompt**：编辑 `config/prompt.txt`，可调整分析规则和输出格式
- **修改业务规则**：编辑 `config/config.yaml` 中的 `rules` 部分
- **调整 API 参数**：编辑 `config/config.yaml` 中的 `api` 部分

## 运行测试

```bash
# 运行全部测试
pytest tests/ -v

# 带覆盖率报告
pytest tests/ --cov=src -v
```

## 技术栈

- Python 3.11+
- DeepSeek API (OpenAI 兼容接口)
- Pandas / PyYAML / python-dotenv
- Tenacity (重试) / tqdm (进度条) / pytest (测试)

## 许可

MIT
