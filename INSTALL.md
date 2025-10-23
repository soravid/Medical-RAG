# 安装指南

## 快速安装

### 1. 克隆项目

```bash
git clone https://github.com/soravid/medical-recommendation-system.git
cd medical-recommendation-system
2. 创建虚拟环境

# 使用 venv
python -m venv venv
# 激活虚拟环境
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
3. 安装依赖
标准安装（推荐）

pip install -r requirements.txt
最小安装

pip install -r requirements-minimal.txt
完整安装

pip install -r requirements-full.txt
开发安装

pip install -r requirements-dev.txt
4. 配置环境变量
创建 .env 文件：


cp .env.example .env
# 编辑 .env 文件，填入你的 API 密钥
5. 验证安装

python scripts/test_basic_functions.py
使用 pip 直接安装（将来支持）

pip install medical-recommendation-system
使用 setup.py 安装

# 开发模式安装
pip install -e .
# 包含完整功能
pip install -e ".[full]"
# 开发模式
pip install -e ".[dev]"
```
