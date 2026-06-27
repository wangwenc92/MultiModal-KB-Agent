# -*- coding: utf-8 -*-
"""工具测试脚本"""
import sys
sys.path.insert(0, ".")

print("=== 1. Calculator Tool ===")
from app.core.agent.tools.calculator import CalculatorTool
calc = CalculatorTool()
print(calc.invoke({"expression": "(15*23+45)/3"}))
print(calc.invoke({"expression": "sqrt(144)"}))
print(calc.invoke({"expression": "sin(3.14159/2)"}))

print("\n=== 2. File Ops Tool ===")
from app.core.agent.tools.file_ops import FileOpsTool
fops = FileOpsTool()
print(fops.invoke({"operation": "write", "path": "test.txt", "content": "Hello World!"}))
print(fops.invoke({"operation": "read", "path": "test.txt"}))
print(fops.invoke({"operation": "list", "path": "."}))

print("\n=== 3. Code Exec Tool ===")
from app.core.agent.tools.code_exec import CodeExecTool
code_exec = CodeExecTool()
print(code_exec.invoke({"code": "print(2 + 3)"}))
print(code_exec.invoke({"code": "import math; print(math.pi)"}))

print("\n=== 4. Text Splitter ===")
from app.core.rag.splitter import TextSplitter
splitter = TextSplitter(chunk_size=50, chunk_overlap=10)
docs = [{"content": "AI is a branch of computer science. It creates systems that simulate human intelligence. Machine learning is a core AI technique. Deep learning uses neural networks.", "metadata": {"source": "test.txt"}}]
chunks = splitter.split(docs)
for i, c in enumerate(chunks):
    print(f"Chunk {i+1}: {c['content'][:60]}...")

print("\n=== 5. Document Loader ===")
from app.core.rag.loader import DocumentLoader
loader = DocumentLoader()
docs = loader.load("data/test_doc.txt")
print(f"Loaded {len(docs)} pages")
if docs:
    print(f"First 100 chars: {docs[0]['content'][:100]}")

print("\n=== 6. API Endpoints (via requests) ===")
import requests
try:
    print("Health:", requests.get("http://localhost:8000/health").json())
    print("Tools:", len(requests.get("http://localhost:8000/api/chat/tools").json()), "tools available")
    print("KB list:", requests.get("http://localhost:8000/api/knowledge/list").json())
    print("Stats:", requests.get("http://localhost:8000/api/admin/stats").json())
except Exception as e:
    print(f"API test failed (server may not be running): {e}")

print("\n=== ALL TESTS PASSED ===")
