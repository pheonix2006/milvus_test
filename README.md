# Milvus Test Project

这个项目是使用 `uv` 初始化的 Milvus 测试项目。

## 目录结构

- `src/milvus_test/`: 源代码目录
- `tests/`: 测试目录

## 环境要求

- Python >= 3.11
- [uv](https://github.com/astral-sh/uv)

## 安装依赖

```bash
uv sync
```

## 运行测试

```bash
uv run pytest
```

## 运行主程序

```bash
uv run python -m milvus_test.main
```
