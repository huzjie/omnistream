# AI Daily Agent

<div align="center">

**🤖 自动化 AI 热点项目生成与发布系统**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Automation](https://img.shields.io/badge/Automation-Daily-green.svg)](https://github.com/yourusername/ai-daily-agent)

*每天一个 AI 项目，建立你的技术影响力 🚀*

</div>

---

## 📖 简介

**AI Daily Agent** 是一个全自动的 AI 热点追踪和项目生成系统。它每天自动：

1. 🔍 **发现热点** - 从多个来源（GitHub Trending、Hugging Face、AI 新闻等）发现最新的 AI 热点
2. 🛠️ **生成项目** - 基于热点自动生成完整的、生产可用的项目代码
3. 📦 **发布到 GitHub** - 自动创建公开仓库，包含完整的文档、测试、CI/CD
4. 📊 **监控数据** - 追踪每个项目的 stars、forks 等指标，生成增长报告

## 🎯 核心价值

### 对于个人 IP 建设

- ✅ **持续输出** - 每天一个高质量项目，展示你的技术视野
- ✅ **紧跟前沿** - 基于最新 AI 趋势，展示你对行业的敏锐度
- ✅ **代码质量** - 每个项目都有完整的文档、测试、CI/CD，展示专业素养
- ✅ **数据驱动** - 通过 GitHub 数据展示你的影响力增长

### 对于技术学习

- ✅ **实战项目** - 不是纸上谈兵，每个项目都是可运行的代码
- ✅ **多技术栈** - 支持 Python、TypeScript、全栈应用等多种类型
- ✅ **最佳实践** - 遵循行业标准，学习如何构建生产级应用

## 🏗️ 系统架构

```
ai-daily-agent/
├── agent.py                    # 主编排器
├── config.yaml                 # 配置文件
├── requirements.txt            # 依赖
│
├── src/
│   ├── discoverer/             # 热点发现模块
│   │   └── hot_topic_discoverer.py
│   ├── generator/              # 项目生成模块
│   │   └── project_generator.py
│   ├── publisher/              # GitHub 发布模块
│   │   └── github_publisher.py
│   └── monitor/                # 数据监控模块
│       └── repo_monitor.py
│
├── data/
│   ├── projects/               # 生成的项目
│   ├── metrics/                # 指标数据
│   ├── logs/                   # 运行日志
│   └── reports/                # 报告
│
├── templates/                  # 项目模板
├── docs/                       # 文档
└── tests/                      # 测试
```

## 🚀 快速开始

### 1. 安装依赖

```bash
cd ai-daily-agent
pip install -r requirements.txt
```

### 2. 配置环境

复制 `.env.example` 为 `.env` 并填写你的配置：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```env
GITHUB_USERNAME=your_github_username
GITHUB_TOKEN=your_github_personal_access_token
```

**获取 GitHub Token：**
1. 访问 https://github.com/settings/tokens
2. 点击 "Generate new token (classic)"
3. 勾选 `repo` 权限
4. 生成并复制 token

### 3. 配置 agent

编辑 `config.yaml`：

```yaml
github:
  username: "your_github_username"  # 你的 GitHub 用户名
  repo_prefix: "ai-daily-"          # 仓库名前缀
  branch: "main"
  visibility: "public"

sources:
  - name: "web_search"
    enabled: true
    keywords:
      - "AI breakthrough 2026"
      - "LLM new release"
      - "AI agent framework"
      # ... 更多关键词
```

### 4. 运行

#### 完整流程（发现 + 生成 + 发布 + 监控）

```bash
python agent.py --config config.yaml
```

#### 自定义主题

```bash
python agent.py --topic "Multi-Agent Collaboration Framework"
```

#### 仅监控模式（更新指标和生成报告）

```bash
python agent.py --monitor-only
```

## ⏰ 自动化运行

### 使用 WorkBuddy 自动化

在 WorkBuddy 中创建自动化任务：

- **时间**: 每天早上 9:00
- **Prompt**: "运行 AI Daily Agent 生成今日项目并发布到 GitHub"
- **工作目录**: `C:\Users\Administrator\WorkBuddy\2026-07-13-13-59-03\ai-daily-agent`

### 使用 Cron (Linux/Mac)

```bash
# 每天早上 9:00 运行
0 9 * * * cd /path/to/ai-daily-agent && python agent.py
```

### 使用 Task Scheduler (Windows)

1. 打开任务计划程序
2. 创建基本任务
3. 设置触发器：每天 9:00
4. 操作：启动程序 `python agent.py`
5. 起始位置：`C:\Users\Administrator\WorkBuddy\2026-07-13-13-59-03\ai-daily-agent`

## 📊 监控和报告

### 查看指标报告

```bash
python agent.py --report
```

报告将保存在 `data/reports/` 目录下，包含：

- 总项目数
- 总 stars 和 forks
- 每日增长趋势
- 最受欢迎的项目排名

### 查看日志

```bash
tail -f data/logs/agent.log
```

## 🎨 项目类型

系统支持多种项目类型，会根据热点自动选择最合适的：

### 1. Python 工具 (`python_tool`)
- CLI 工具或库
- 包含完整的命令行参数处理
- 适合快速原型和实用工具

### 2. Web 应用 (`web_app`)
- 前端 + 后端分离
- React + TypeScript 前端
- Express/Node.js 后端
- 适合展示类应用

### 3. 全栈应用 (`full_stack`)
- FastAPI 后端
- Next.js 前端
- PostgreSQL 数据库
- 适合完整的企业级应用

### 4. AI Agent (`ai_agent`)
- 基于 LLM 的智能代理
- 工具集成
- 适合 AI 应用展示

### 5. 数据管道 (`data_pipeline`)
- 数据处理和分析
- Pandas + 可视化
- 适合数据科学项目

## 🔧 自定义和扩展

### 添加新的热点来源

编辑 `src/discoverer/hot_topic_discoverer.py`：

```python
def _discover_from_custom_source(self, source: Dict) -> List[HotTopic]:
    # 实现你的自定义来源
    pass
```

### 添加新的项目模板

编辑 `src/generator/project_generator.py`：

```python
def _generate_custom_project(self, project_dir: Path, topic: Dict):
    # 实现你的自定义项目生成逻辑
    pass
```

### 自定义 README 模板

编辑 `src/generator/project_generator.py` 中的 `_generate_readme` 方法。

## 📝 配置说明

### config.yaml 关键字段

```yaml
github:
  username: "your_username"     # GitHub 用户名
  token: "your_token"            # GitHub Personal Access Token
  repo_prefix: "ai-daily-"       # 仓库名前缀
  branch: "main"                 # 默认分支
  visibility: "public"           # 仓库可见性

sources:
  - name: "web_search"           # 网络搜索
    enabled: true
    keywords: [...]              # 搜索关键词
    max_topics: 5                # 最大主题数
  
  - name: "github_trending"      # GitHub Trending
    enabled: true
    languages: ["python", "typescript"]
    since: "daily"
  
  - name: "huggingface"          # Hugging Face
    enabled: true
    model_types: ["text-generation"]

generator:
  tech_stacks:                   # 支持的技术栈
    - name: "python_tool"
    - name: "web_app"
    - name: "full_stack"
    - name: "ai_agent"
    - name: "data_pipeline"
  
  readme:                        # README 配置
    include_badges: true
    include_demo: true
    include_architecture: true
  
  quality:                       # 质量门控
    must_have_tests: true
    must_have_dockerfile: true
    must_have_ci: true

monitor:
  metrics:                       # 追踪的指标
    - "stars"
    - "forks"
    - "watchers"
  
  notifications:                 # 通知设置
    enabled: true
    star_milestones: [1, 5, 10, 50, 100]
```

## 🔐 安全注意事项

1. **不要提交 `.env` 文件** - 已在 `.gitignore` 中排除
2. **使用最小权限的 GitHub Token** - 只需要 `repo` 权限
3. **定期轮换 Token** - 建议每 90 天更换一次
4. **审查生成的代码** - 虽然自动生成，但发布前建议快速审查

## 🐛 故障排除

### 问题：GitHub 认证失败

```bash
# 检查 gh CLI 是否安装
gh --version

# 登录 GitHub
gh auth login
```

### 问题：项目生成失败

检查日志：
```bash
cat data/logs/agent.log
```

常见问题：
- 磁盘空间不足
- 依赖未安装
- 配置文件格式错误

### 问题：无法推送代码

```bash
# 检查 Git 配置
git config --global user.name
git config --global user.email

# 检查远程仓库
git remote -v
```

## 📚 进阶用法

### 批量生成

```python
from agent import AIDailyAgent

agent = AIDailyAgent('config.yaml')

# 生成多个项目
topics = ["AI Agent Framework", "LLM Optimization", "Computer Vision"]
for topic in topics:
    agent.run_daily_pipeline(custom_topic=topic)
```

### 自定义监控指标

编辑 `config.yaml`：

```yaml
monitor:
  metrics:
    - "stars"
    - "forks"
    - "watchers"
    - "issues"
    - "pull_requests"  # 添加自定义指标
```

### 集成通知

添加 Slack/Discord 通知：

```python
# 在 agent.py 的 run_daily_pipeline 方法末尾添加
if result['status'] == 'success':
    # 发送通知
    send_notification(f"✅ 今日项目已发布: {result['steps']['publish']['repo_url']}")
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License

## 🙏 致谢

- 感谢所有开源项目的贡献者
- 基于最新的 AI 技术趋势
- 使用 WorkBuddy 自动化平台

---

<div align="center">

**⭐ 如果这个项目对你有帮助，请给它一个 Star！⭐**

*每天一个 AI 项目，让你的 GitHub 成为技术前沿的代名词 🚀*

</div>
