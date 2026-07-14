"""Mode 2 - Chat.

Nhận 1 câu hỏi tự do (không phải trắc nghiệm 4 đáp án), trả về 1 câu trả
lời tự do. KHÔNG dùng AgentState/graph.py (thiết kế riêng cho Mode 1 -
benchmark trắc nghiệm) - Mode 2 gọi LLM trực tiếp qua llm_client.py,
đơn giản, độc lập hoàn toàn.

Xoá file này không ảnh hưởng gì đến Mode 1 (main.py --mode benchmark,
graph.py, agents/reasoning_agent.py vẫn chạy bình thường).
"""

from llm_client import build_llm

CHAT_SYSTEM_PROMPT = (
    "Bạn là trợ lý y tế AI. Trả lời ngắn gọn, dễ hiểu, bằng tiếng Việt. "
    "Nếu câu hỏi liên quan triệu chứng hoặc thuốc, đưa ra hướng xử lý "
    "chung an toàn, và LUÔN nhắc người dùng nên đi khám bác sĩ nếu triệu "
    "chứng nặng, kéo dài, hoặc không chắc chắn - đây không phải chẩn đoán "
    "y khoa chính thức."
)


def run_chat() -> None:
    llm = build_llm(temperature=0.3)
    print("=== Med-AI Chat === (nhập 'exit' để thoát)")

    while True:
        question = input("\nCâu hỏi: ").strip()
        if question.lower() in ("exit", "quit"):
            print("Tạm biệt!")
            break
        if not question:
            continue

        messages = [
            {"role": "system", "content": CHAT_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        response = llm.invoke(messages)
        print(f"\nTrả lời: {response.content}")
