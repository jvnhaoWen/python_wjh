#!/bin/bash

# 获取脚本所在目录
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "=== 开始配置 Python 虚拟环境 (macOS/Linux) ==="

# 检查 Python3 是否安装
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 python3，请先安装 Python。"
    exit 1
fi

# 创建虚拟环境 (venv)
if [ ! -d "venv" ]; then
    echo "正在创建虚拟环境..."
    python3 -m venv venv
else
    echo "虚拟环境已存在。"
fi

# 激活虚拟环境并安装依赖
echo "正在激活虚拟环境并安装依赖..."
source venv/bin/activate

# 升级 pip
pip install --upgrade pip

# 安装 requirements.txt 中的依赖
if [ -f "requirements.txt" ]; then
    echo "正在安装核心依赖，这可能需要一些时间..."
    pip install -r requirements.txt
else
    echo "警告: 未找到 requirements.txt，跳过依赖安装。"
fi

echo "=== 虚拟环境配置完成 ==="
echo "请运行以下命令激活环境并启动项目:"
echo "source venv/bin/activate"
echo "python main.py"
