"""性能基准测试脚本"""
import sys
sys.path.insert(0, ".")

import time
import statistics
import httpx

BASE_URL = "http://localhost:8000"


def test_health_latency(rounds: int = 50) -> dict:
    """测试健康检查接口延迟"""
    latencies = []
    with httpx.Client() as client:
        for _ in range(rounds):
            start = time.time()
            resp = client.get(f"{BASE_URL}/health")
            elapsed = (time.time() - start) * 1000
            if resp.status_code == 200:
                latencies.append(elapsed)

    if not latencies:
        return {"error": "All requests failed"}

    return {
        "endpoint": "/health",
        "rounds": rounds,
        "avg_ms": round(statistics.mean(latencies), 2),
        "p50_ms": round(statistics.median(latencies), 2),
        "p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2),
        "p99_ms": round(sorted(latencies)[int(len(latencies) * 0.99)], 2),
        "min_ms": round(min(latencies), 2),
        "max_ms": round(max(latencies), 2),
    }


def test_concurrent_requests(concurrency: int = 10, total: int = 50) -> dict:
    """并发压力测试"""
    import concurrent.futures

    results = {"success": 0, "failed": 0, "latencies": []}

    def single_request():
        with httpx.Client(timeout=10) as client:
            start = time.time()
            try:
                resp = client.get(f"{BASE_URL}/health")
                elapsed = (time.time() - start) * 1000
                if resp.status_code == 200:
                    results["success"] += 1
                else:
                    results["failed"] += 1
                results["latencies"].append(elapsed)
            except Exception:
                results["failed"] += 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(single_request) for _ in range(total)]
        concurrent.futures.wait(futures)

    latencies = results["latencies"]
    return {
        "concurrency": concurrency,
        "total_requests": total,
        "success": results["success"],
        "failed": results["failed"],
        "avg_ms": round(statistics.mean(latencies), 2) if latencies else 0,
        "p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2) if latencies else 0,
    }


def test_vector_search_latency() -> dict:
    """测试向量检索延迟（需要有已索引的数据）"""
    with httpx.Client(timeout=30) as client:
        # 先获取知识库列表
        resp = client.get(f"{BASE_URL}/api/knowledge/list")
        if resp.status_code != 200:
            return {"error": "Cannot access API"}
        kbs = resp.json()
        if not kbs:
            return {"error": "No knowledge bases found"}

        kb_id = kbs[0]["id"]
        latencies = []
        for _ in range(10):
            start = time.time()
            resp = client.post(f"{BASE_URL}/api/chat/send", json={
                "question": "测试查询",
                "knowledge_base_id": kb_id,
                "mode": "rag",
            })
            elapsed = (time.time() - start) * 1000
            if resp.status_code == 200:
                latencies.append(elapsed)

    if not latencies:
        return {"error": "All chat requests failed"}

    return {
        "endpoint": "/api/chat/send (RAG)",
        "rounds": len(latencies),
        "avg_ms": round(statistics.mean(latencies), 2),
        "p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2),
    }


if __name__ == "__main__":
    print("=" * 50)
    print("MultiModal KB Agent - 性能基准测试")
    print("=" * 50)

    print("\n[1] 健康检查延迟测试")
    r = test_health_latency()
    for k, v in r.items():
        print(f"  {k}: {v}")

    print("\n[2] 并发压力测试 (10并发, 50请求)")
    r = test_concurrent_requests()
    for k, v in r.items():
        print(f"  {k}: {v}")

    print("\n[3] RAG问答延迟测试")
    r = test_vector_search_latency()
    for k, v in r.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 50)
    print("测试完成")
