import sys
sys.path.insert(0, ".")

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestHealthEndpoints:
    def test_health(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_root(self):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "version" in resp.json()


class TestKnowledgeAPI:
    def test_create_and_list(self):
        # 创建
        resp = client.post("/api/knowledge/create", json={"name": "test_kb", "description": "测试"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "test_kb"
        kb_id = data["id"]

        # 列表
        resp = client.get("/api/knowledge/list")
        assert resp.status_code == 200
        assert any(kb["id"] == kb_id for kb in resp.json())

        # 详情
        resp = client.get(f"/api/knowledge/{kb_id}")
        assert resp.status_code == 200
        assert resp.json()["name"] == "test_kb"

        # 删除
        resp = client.delete(f"/api/knowledge/{kb_id}")
        assert resp.status_code == 200

    def test_not_found(self):
        resp = client.get("/api/knowledge/nonexistent")
        assert resp.status_code == 404


class TestChatAPI:
    def test_list_tools(self):
        resp = client.get("/api/chat/tools")
        assert resp.status_code == 200
        tools = resp.json()
        assert len(tools) > 0
        assert any(t["name"] == "calculator" for t in tools)

    def test_list_sessions(self):
        resp = client.get("/api/chat/sessions")
        assert resp.status_code == 200


class TestAdminAPI:
    def test_stats(self):
        resp = client.get("/api/admin/stats")
        assert resp.status_code == 200
        assert "knowledge_bases" in resp.json()
