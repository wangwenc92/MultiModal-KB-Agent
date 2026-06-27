import sys
sys.path.insert(0, ".")

import pytest
from app.core.rag.loader import DocumentLoader
from app.core.rag.splitter import TextSplitter


class TestDocumentLoader:
    def test_load_text(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello World\n这是测试内容", encoding="utf-8")
        loader = DocumentLoader()
        docs = loader.load(str(f))
        assert len(docs) == 1
        assert "Hello World" in docs[0]["content"]
        assert docs[0]["metadata"]["type"] == "txt"

    def test_load_markdown(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("# 标题\n\n正文内容", encoding="utf-8")
        loader = DocumentLoader()
        docs = loader.load(str(f))
        assert len(docs) == 1
        assert "标题" in docs[0]["content"]

    def test_unsupported_type(self, tmp_path):
        f = tmp_path / "test.xyz"
        f.write_text("content")
        loader = DocumentLoader()
        with pytest.raises(ValueError, match="Unsupported"):
            loader.load(str(f))


class TestTextSplitter:
    def test_recursive_split(self):
        splitter = TextSplitter(chunk_size=20, chunk_overlap=5)
        docs = [{"content": "这是一段很长的文本，需要被分割成多个小块。每个小块应该不超过指定的大小限制。", "metadata": {"source": "test.txt"}}]
        chunks = splitter.split(docs, strategy="recursive")
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk["content"]) <= 25  # 允许少量溢出

    def test_sentence_split(self):
        splitter = TextSplitter(chunk_size=50, chunk_overlap=10)
        docs = [{"content": "第一句话。第二句话。第三句话。第四句话。", "metadata": {"source": "test.txt"}}]
        chunks = splitter.split(docs, strategy="sentence")
        assert len(chunks) >= 1

    def test_empty_doc(self):
        splitter = TextSplitter()
        chunks = splitter.split([{"content": "", "metadata": {}}])
        assert len(chunks) == 0

    def test_small_doc_single_chunk(self):
        splitter = TextSplitter(chunk_size=500)
        docs = [{"content": "短文本", "metadata": {"source": "test.txt"}}]
        chunks = splitter.split(docs)
        assert len(chunks) == 1
        assert chunks[0]["content"] == "短文本"


class TestCalculator:
    def test_basic_calc(self):
        from app.core.agent.tools.calculator import CalculatorTool
        calc = CalculatorTool()
        result = calc.invoke({"expression": "(15*23+45)/3"})
        assert "130" in result

    def test_math_functions(self):
        from app.core.agent.tools.calculator import CalculatorTool
        calc = CalculatorTool()
        result = calc.invoke({"expression": "sqrt(144)"})
        assert "12" in result

    def test_division_by_zero(self):
        from app.core.agent.tools.calculator import CalculatorTool
        calc = CalculatorTool()
        result = calc.invoke({"expression": "1/0"})
        assert "错误" in result


class TestFileOps:
    def test_write_and_read(self, tmp_path):
        import os
        os.chdir(tmp_path)
        from app.core.agent.tools.file_ops import FileOpsTool
        tool = FileOpsTool()
        tool.invoke({"operation": "write", "path": "test.txt", "content": "hello"})
        result = tool.invoke({"operation": "read", "path": "test.txt"})
        assert "hello" in result

    def test_list_dir(self, tmp_path):
        import os
        os.chdir(tmp_path)
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        from app.core.agent.tools.file_ops import FileOpsTool
        tool = FileOpsTool()
        result = tool.invoke({"operation": "list", "path": "."})
        assert "a.txt" in result
        assert "b.txt" in result
