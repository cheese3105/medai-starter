"""Chat mode — REPL tương tác, mỗi lượt chat đi qua runner pipeline.

STM session buffer + LTM fact store được quản lý ở đây (ngoài runner),
chỉ truyền context text vào runner — runner không biết gì về memory
module, chỉ nhận string qua placeholder.
"""

from __future__ import annotations

import uuid
from typing import Optional

from core.config import RunConfig
from core.logger import DebugLogger
from core.runner import Runner
from core.types import EpisodeInput


def run_chat(config: RunConfig) -> None:
    """Chạy REPL chat."""

    logger = DebugLogger(level=config.debug)
    runner = Runner(config, logger)

    # --- STM session ---
    session_history: list[dict] = []  # [{role, content}, ...]
    stm_enabled = (
        config.memory.short_term.enabled
        and config.memory.short_term.scope in ("session", "both")
    )
    max_turns = config.memory.short_term.max_turns

    # --- LTM ---
    ltm = None
    ltm_enabled = config.memory.long_term.enabled
    if ltm_enabled:
        from memory.long_term import LongTermMemory
        ltm = LongTermMemory(
            user_id=config.memory.long_term.user_id,
            chroma_dir=config.memory.long_term.chroma_dir,
            collection_name=config.memory.long_term.collection_name,
            embedding_model=config.embedding_model,
            embedding_base_url=config.embedding_base_url,
            embedding_api_key=config.embedding_api_key,
        )

    # Status line
    mem_status = "OFF"
    if stm_enabled or ltm_enabled:
        parts = []
        if stm_enabled:
            parts.append(f"STM(session, {max_turns} turns)")
        if config.memory.short_term.enabled and config.memory.short_term.scope in ("loop", "both"):
            parts.append("STM(loop)")
        if ltm_enabled:
            mode = config.memory.long_term.mode
            parts.append(f"LTM({mode})")
        mem_status = " + ".join(parts)

    print(f"=== MED-AI Chat === ({config.model} | memory: {mem_status})")
    print("Nhập 'exit' để thoát.\n")

    turn_count = 0

    while True:
        question = input("Câu hỏi: ").strip()
        if question.lower() in ("exit", "quit"):
            print("Tạm biệt!")
            break
        if not question:
            continue

        turn_count += 1
        turn_id = f"chat_{turn_count:04d}"

        # 1. Build LTM context
        ltm_text = ""
        ltm_facts_list: list[str] = []
        if ltm is not None:
            try:
                hits = ltm.retrieve(question, top_k=config.memory.long_term.top_k)
                ltm_facts_list = [h["text"] for h in hits]
                if ltm_facts_list:
                    ltm_text = "Thông tin đã biết về người dùng:\n" + "\n".join(
                        f"- {f}" for f in ltm_facts_list
                    )
                    logger.memory_info(turn_id, f"LTM: kéo về {len(ltm_facts_list)} fact")
            except Exception as e:
                logger.warn(f"LTM retrieve lỗi: {e}")

        # 2. Build STM session context
        stm_text = ""
        if stm_enabled and session_history:
            recent = session_history[-(max_turns * 2):]
            stm_text = "\n".join(
                f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
                for m in recent
            )

        # 3. Run episode
        episode = EpisodeInput(
            question_id=turn_id,
            question=question,
            choices=[],
            gold_answer=None,
        )

        result = runner.run_episode(
            episode,
            stm_session_history=stm_text,
            ltm_facts=ltm_text,
        )

        answer = result.predicted_answer
        print(f"\nTrả lời: {answer}\n")

        # 4. Update STM session
        if stm_enabled:
            session_history.append({"role": "user", "content": question})
            session_history.append({"role": "assistant", "content": answer})

        # 5. Write LTM (nếu read_write)
        if (
            ltm is not None
            and config.memory.long_term.mode == "read_write"
            and config.memory.long_term.write_after_answer
        ):
            try:
                from memory.long_term import maybe_extract_and_save
                maybe_extract_and_save(
                    ltm=ltm,
                    user_message=question,
                    assistant_message=answer,
                    model=config.model,
                    base_url=config.model_base_url,
                    api_key=config.model_api_key,
                    logger=logger,
                    turn_id=turn_id,
                )
            except Exception as e:
                logger.warn(f"LTM write lỗi: {e}")
