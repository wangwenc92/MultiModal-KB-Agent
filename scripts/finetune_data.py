"""
从知识库中提取QA对，生成微调训练数据
使用LLM自动基于已有文档生成问答对
"""
import sys
sys.path.insert(0, ".")

import json
import os
from app.models.database import SessionLocal, DocumentModel, ChunkModel
from app.services.llm_service import get_llm_service
from app.utils.logger import log

QA_GENERATE_PROMPT = """基于以下文档内容，生成3个高质量的问答对。

要求：
1. 问题应该是用户实际会问的问题
2. 回答应准确、完整，基于文档内容
3. 覆盖不同难度：简单事实、理解分析、综合推理
4. 输出JSON数组格式

文档内容：
{content}

请输出JSON数组，格式如下：
[
  {{"question": "问题", "answer": "回答"}},
  {{"question": "问题", "answer": "回答"}},
  {{"question": "问题", "answer": "回答"}}
]"""


def generate_training_data(kb_id: str = None, output_path: str = "data/training_data.jsonl"):
    """从数据库中的chunks生成微调训练数据"""
    db = SessionLocal()
    llm = get_llm_service()

    query = db.query(ChunkModel)
    if kb_id:
        query = query.join(DocumentModel).filter(DocumentModel.kb_id == kb_id)

    chunks = query.limit(100).all()
    log.info(f"Found {len(chunks)} chunks for training data generation")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    count = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            try:
                prompt = QA_GENERATE_PROMPT.format(content=chunk.content[:1500])
                response = llm.chat("你是训练数据生成助手。", prompt)

                # 解析JSON
                response = response.strip()
                if response.startswith("```"):
                    response = response.split("\n", 1)[1].rsplit("```", 1)[0].strip()

                qa_pairs = json.loads(response)
                if not isinstance(qa_pairs, list):
                    continue

                for qa in qa_pairs:
                    if "question" in qa and "answer" in qa:
                        record = {
                            "messages": [
                                {"role": "system", "content": "你是一个知识库助手，请基于知识回答用户问题。"},
                                {"role": "user", "content": qa["question"]},
                                {"role": "assistant", "content": qa["answer"]},
                            ]
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        count += 1

                log.info(f"Generated {len(qa_pairs)} QA pairs from chunk {chunk.id[:8]}")

            except Exception as e:
                log.warning(f"Failed to generate QA for chunk {chunk.id[:8]}: {e}")
                continue

    db.close()
    log.info(f"Generated {count} training records → {output_path}")
    return count


def validate_training_data(data_path: str = "data/training_data.jsonl"):
    """验证训练数据格式"""
    valid = 0
    invalid = 0
    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            try:
                record = json.loads(line)
                assert "messages" in record
                assert len(record["messages"]) == 3
                assert record["messages"][0]["role"] == "system"
                assert record["messages"][1]["role"] == "user"
                assert record["messages"][2]["role"] == "assistant"
                valid += 1
            except Exception as e:
                log.warning(f"Invalid record at line {i}: {e}")
                invalid += 1

    log.info(f"Validation: {valid} valid, {invalid} invalid")
    return valid, invalid


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="生成微调训练数据")
    parser.add_argument("--kb-id", type=str, default=None, help="知识库ID（为空则使用全部数据）")
    parser.add_argument("--output", type=str, default="data/training_data.jsonl", help="输出路径")
    parser.add_argument("--validate", action="store_true", help="验证已有训练数据")
    args = parser.parse_args()

    if args.validate:
        validate_training_data(args.output)
    else:
        generate_training_data(args.kb_id, args.output)
