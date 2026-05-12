from __future__ import annotations

from collections import Counter
from datetime import datetime
import json
import subprocess
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from .config import AIAnalysisConfig
from .models import Candidate


def generate_ai_analysis(
    config: AIAnalysisConfig,
    candidates: list[Candidate],
    scanned_count: int,
    failed_count: int,
    latest_trade_date: str,
) -> str:
    if not config.enabled or not candidates:
        return ""
    try:
        api_key = config.api_key or _discover_local_sub2api_key(config)
        if not api_key:
            raise RuntimeError("缺少 sub2api API key")
        prompt = _build_prompt(candidates, scanned_count, failed_count, latest_trade_date, config.max_candidates)
        return _call_sub2api(config=config, api_key=api_key, prompt=prompt, max_tokens=1800).strip()
    except Exception as exc:  # pragma: no cover - external AI gateway variance
        return f"AI分析暂不可用：{_safe_error(exc)}"


def _build_prompt(
    candidates: list[Candidate],
    scanned_count: int,
    failed_count: int,
    latest_trade_date: str,
    max_candidates: int,
) -> str:
    limited = candidates[: max(1, max_candidates)]
    tag_counts = Counter(tag for item in limited for tag in item.tags)
    payload = {
        "trade_date": latest_trade_date,
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "scanned_count": scanned_count,
        "failed_count": failed_count,
        "candidate_count": len(candidates),
        "tag_distribution": tag_counts.most_common(20),
        "candidates": [_candidate_summary(idx, item) for idx, item in enumerate(limited, start=1)],
    }
    return (
        "你是一个A股突破策略的AI选股分析员。请基于给定的程序筛选结果做复核分析，"
        "输出中文 Markdown。\n\n"
        "分析要求：\n"
        "1. 结合候选股的行业/题材tag聚类，判断当前结果更偏向哪些市场热点或主线。\n"
        "2. 从宏观环境角度讨论可能影响这些主线的因素，例如政策、流动性、汇率、出口、地产、"
        "利率、业绩兑现和风险偏好；不要编造具体未提供的新闻事实。\n"
        "3. 评估未来是否有预期：只能用“需要验证/值得跟踪/风险较高”等审慎表述，不能承诺收益。\n"
        "4. 挑 5-8 只最需要复核的候选，说明关注理由、需要确认的催化/业绩/成交信号和主要风险。\n"
        "5. 最后给出“明日/下次扫描观察清单”：需要关注的tag、量价确认、止损纪律。\n"
        "6. 这是研究辅助，不是投资建议；不得使用确定性买卖指令。\n\n"
        "请控制在 1200-1800 中文字以内，不要输出大表格。\n\n"
        "输出结构固定为：\n"
        "## 总体判断\n"
        "## 题材主线\n"
        "## 重点候选复核\n"
        "## 风险与反证\n"
        "## 下次扫描观察\n\n"
        "输入数据如下：\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )


def _candidate_summary(rank: int, item: Candidate) -> dict[str, Any]:
    return {
        "rank": rank,
        "code": item.code,
        "name": item.name,
        "tags": list(item.tags),
        "latest_close": round(item.latest_close, 2),
        "circ_mv_yi": round(item.circ_mv, 1) if item.circ_mv > 0 else None,
        "resistance": round(item.resistance, 2),
        "breakout_pct": round(item.breakout_pct * 100, 2),
        "volume_ratio": round(item.volume_ratio, 2),
        "volume_trend": round(item.volume_trend, 2),
        "atr_pct": round(item.atr_pct * 100, 2),
        "score": round(item.score, 1),
        "buy_zone": [round(item.buy_zone_low, 2), round(item.buy_zone_high, 2)],
        "stop_loss": round(item.stop_loss, 2),
        "position_hint": item.position_hint,
    }


def _call_sub2api(config: AIAnalysisConfig, api_key: str, prompt: str, max_tokens: int) -> str:
    url = config.base_url.rstrip("/") + "/v1/messages"
    body = json.dumps(
        {
            "model": config.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(request, timeout=max(10, config.timeout)) as response:
        payload = json.loads(response.read().decode("utf-8"))
    text = _extract_text(payload)
    if not text:
        raise RuntimeError("AI网关返回为空")
    return text


def _extract_text(payload: dict[str, Any]) -> str:
    content = payload.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        if parts:
            return "\n".join(parts)
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        if isinstance(message.get("content"), str):
            return message["content"]
    return ""


def _discover_local_sub2api_key(config: AIAnalysisConfig) -> str:
    if not config.allow_local_key_discovery or "127.0.0.1" not in config.base_url:
        return ""
    command = [
        "docker",
        "exec",
        "sub2api-postgres",
        "psql",
        "-U",
        "sub2api",
        "-d",
        "sub2api",
        "-At",
        "-c",
        (
            "select key from api_keys "
            "where status='active' and deleted_at is null "
            "order by case when name='public-openai-admin' then 0 else 1 end, id "
            "limit 1"
        ),
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def _safe_error(exc: Exception) -> str:
    if isinstance(exc, HTTPError):
        try:
            detail = exc.read().decode("utf-8")[:300]
        except Exception:
            detail = ""
        return f"HTTP {exc.code} {detail}".strip()
    if isinstance(exc, URLError):
        return str(exc.reason)
    return str(exc)
