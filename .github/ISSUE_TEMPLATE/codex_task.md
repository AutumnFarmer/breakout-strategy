---
name: Codex task
about: Task for server-side Codex implementation
title: "[Codex] "
labels: codex-task
---

## 目标

说明这次要解决什么问题。

## 背景

说明当前策略、代码或回测中发现的问题。

## 修改范围

建议修改文件：

- `src/a_breakout_screener/scoring.py`
- `src/a_breakout_screener/screener.py`
- `src/a_breakout_screener/data.py`
- `tests/...`

## 具体要求

1.
2.
3.

## 不要做

- 不要提交 `.env`
- 不要提交行情缓存
- 不要提交输出目录
- 不要大改无关模块

## 验收标准

- `uv run pytest` 通过
- 新增或更新相关测试
- 报告输出字段符合预期
- 不破坏现有 CLI

## Review 重点

- 策略逻辑是否正确
- 是否引入伪信号
- 回测假设是否合理
- 数据源异常是否处理
