"""Short-term Memory - Lưu trữ ngữ cảnh conversation hiện tại.

Quản lý:
- Lịch sử hỏi đáp (Q&A pairs)
- Evidence đã retrieve trong session
- Sliding window (nếu set max_turns) hoặc unlimited (max_turns=None)
- Save/load session từ file JSON
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Union


@dataclass
class ConversationTurn:
    """1 lượt hỏi đáp trong conversation."""
    turn_id: int
    question: str
    answer: str
    explanation: Optional[str] = None
    confidence: Optional[float] = None
    retrieved_docs: Optional[List[Dict]] = None
    verifier_verdict: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        """Convert to dict for JSON serialization."""
        return asdict(self)

    def to_text_summary(self, max_explanation_length: int = 200) -> str:
        """Tóm tắt ngắn gọn để đưa vào context LLM."""
        parts = [f"Q{self.turn_id}: {self.question}"]
        parts.append(f"A{self.turn_id}: {self.answer}")
        if self.explanation:
            explanation = self.explanation[:max_explanation_length]
            if len(self.explanation) > max_explanation_length:
                explanation += "..."
            parts.append(f"Explanation: {explanation}")
        return "\n".join(parts)


class ShortTermMemory:
    """Quản lý short-term memory với sliding window hoặc unlimited mode."""

    def __init__(
        self,
        max_turns: Optional[int] = None,  # None = unlimited, số = sliding window
        max_evidence_per_turn: int = 3,
        follow_up_keywords: Optional[List[str]] = None,
    ):
        """
        Args:
            max_turns: None = giữ hết session (unlimited), số = chỉ giữ N turns gần nhất
            max_evidence_per_turn: Chỉ giữ top-N evidence quan trọng nhất mỗi turn
            follow_up_keywords: Keywords để detect follow-up question
        """
        self.max_turns = max_turns
        self.max_evidence_per_turn = max_evidence_per_turn

        # Unlimited mode: list thông thường, Limited mode: deque với maxlen
        if max_turns is None or max_turns <= 0:
            self.turns: Union[List[ConversationTurn], deque] = []
            self.mode = "unlimited"
        else:
            self.turns: Union[List[ConversationTurn], deque] = deque(maxlen=max_turns)
            self.mode = "sliding_window"

        self.current_turn_id = 0
        self.session_id: Optional[str] = None
        self.session_metadata = {
            "session_start": None,
            "total_questions": 0,
            "mode": self.mode,
        }

        # Default follow-up keywords (có thể override từ config)
        self.follow_up_keywords = follow_up_keywords or [
            "nó", "cái đó", "ở trên", "trước đó", "điều đó",
            "it", "this", "that", "above", "previous", "earlier",
            "the above", "mentioned", "said"
        ]

    def add_turn(
        self,
        question: str,
        answer: str,
        explanation: Optional[str] = None,
        confidence: Optional[float] = None,
        retrieved_docs: Optional[List[Dict]] = None,
        verifier_verdict: Optional[str] = None,
    ) -> ConversationTurn:
        """Thêm 1 lượt hỏi đáp mới vào memory."""
        if self.session_metadata["session_start"] is None:
            self.session_metadata["session_start"] = time.time()

        # Chỉ giữ evidence quan trọng nhất (theo score)
        if retrieved_docs and len(retrieved_docs) > self.max_evidence_per_turn:
            scored = [d for d in retrieved_docs if not d.get("error") and d.get("score") is not None]
            scored.sort(key=lambda d: d["score"])
            retrieved_docs = scored[:self.max_evidence_per_turn]

        turn = ConversationTurn(
            turn_id=self.current_turn_id,
            question=question,
            answer=answer,
            explanation=explanation,
            confidence=confidence,
            retrieved_docs=retrieved_docs,
            verifier_verdict=verifier_verdict,
            timestamp=time.time(),
        )

        self.turns.append(turn)
        self.current_turn_id += 1
        self.session_metadata["total_questions"] += 1

        return turn

    def get_recent_turns(self, n: Optional[int] = None) -> List[ConversationTurn]:
        """Lấy N lượt hỏi đáp gần nhất."""
        turns_list = list(self.turns)
        if n is None:
            return turns_list
        return turns_list[-n:]

    def get_conversation_history_text(
        self,
        max_turns: Optional[int] = None,
        include_explanation: bool = True
    ) -> str:
        """Render lịch sử conversation thành text để đưa vào LLM prompt."""
        recent = self.get_recent_turns(max_turns)
        if not recent:
            return ""

        history_parts = ["Previous conversation:"]
        for turn in recent:
            if include_explanation:
                history_parts.append(turn.to_text_summary())
            else:
                history_parts.append(f"Q{turn.turn_id}: {turn.question}")
                history_parts.append(f"A{turn.turn_id}: {turn.answer}")

        return "\n".join(history_parts)

    def detect_follow_up(self, current_question: str) -> bool:
        """Phát hiện câu hỏi có phải follow-up không (keyword + heuristics).

        Cải tiến so với naive keyword matching:
        - Strip punctuation trước khi so sánh words
        - Multi-word keywords ("cái đó", "ở trên") match trực tiếp trong raw text
        - Single-word keywords: check context xung quanh
          + Đứng đầu câu KHÔNG kèm noun dài ngay sau → follow-up
          + Đứng cuối câu (hoặc trước punctuation) → follow-up
          + Đứng giữa + kèm noun dài ngay sau → likely determiner (skip)
        """
        if not self.turns:
            return False

        import re
        question_lower = current_question.lower().strip()
        # Strip punctuation để tách words chính xác
        words = re.findall(r"[a-zA-ZÀ-ỹ]+", question_lower)

        for keyword in self.follow_up_keywords:
            kw_words = keyword.split()

            # Multi-word keywords ("cái đó", "ở trên", "trước đó") → match in raw text
            if len(kw_words) > 1:
                if keyword in question_lower:
                    return True
                continue

            # Single-word keyword → check in cleaned words list
            if keyword not in words:
                continue

            # "it" and "nó" are always pronouns (never determiners) → always follow-up
            if keyword in ("it", "nó"):
                return True

            kw_index = words.index(keyword)

            # Keyword ở cuối câu ("...about this", "...treat that") → follow-up
            if kw_index >= len(words) - 1:
                return True

            # Keyword ở đầu câu
            if kw_index == 0:
                # "This" + noun dài = determiner ("This disease is...")
                if len(words) > 1 and len(words[1]) > 3 and words[1].isalpha():
                    continue
                # "This?" / "That is wrong" → follow-up
                return True

            # Keyword ở giữa câu
            next_word = words[kw_index + 1] if kw_index < len(words) - 1 else ""
            # "this" + noun dài ("this medication", "this treatment") → determiner, skip
            if len(next_word) > 3 and next_word.isalpha():
                continue

            # Còn lại: follow-up ("about this", "for that")
            return True

        return False

    def get_last_topic(self) -> Optional[str]:
        """Lấy topic của câu hỏi trước (để resolve follow-up)."""
        if not self.turns:
            return None
        return list(self.turns)[-1].question

    def clear(self):
        """Xóa toàn bộ memory (bắt đầu conversation mới)."""
        if self.mode == "unlimited":
            self.turns = []
        else:
            self.turns = deque(maxlen=self.max_turns)

        self.current_turn_id = 0
        self.session_metadata = {
            "session_start": None,
            "total_questions": 0,
            "mode": self.mode,
        }

    def is_empty(self) -> bool:
        """Check xem memory có rỗng không."""
        return len(self.turns) == 0

    def get_stats(self) -> Dict[str, Any]:
        """Lấy statistics về session hiện tại."""
        if not self.turns:
            return {
                "total_turns": 0,
                "session_duration": 0,
                "avg_confidence": None,
            }

        turns_list = list(self.turns)
        confidences = [t.confidence for t in turns_list if t.confidence is not None]

        duration = 0
        if self.session_metadata["session_start"]:
            duration = time.time() - self.session_metadata["session_start"]

        return {
            "total_turns": len(turns_list),
            "session_duration": duration,
            "avg_confidence": sum(confidences) / len(confidences) if confidences else None,
            "mode": self.mode,
            "max_turns": self.max_turns,
        }

    def to_dict(self) -> Dict:
        """Export memory ra dict (để save/load)."""
        return {
            "session_id": self.session_id,
            "turns": [turn.to_dict() for turn in self.turns],
            "current_turn_id": self.current_turn_id,
            "session_metadata": self.session_metadata,
            "config": {
                "max_turns": self.max_turns,
                "max_evidence_per_turn": self.max_evidence_per_turn,
                "mode": self.mode,
            }
        }

    def save(self, path: Union[str, Path]):
        """Lưu memory ra file JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> ShortTermMemory:
        """Load memory từ file JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        config = data.get("config", {})
        memory = cls(
            max_turns=config.get("max_turns"),
            max_evidence_per_turn=config.get("max_evidence_per_turn", 3),
        )

        memory.session_id = data.get("session_id")
        memory.current_turn_id = data["current_turn_id"]
        memory.session_metadata = data["session_metadata"]

        # Restore turns
        for turn_data in data["turns"]:
            turn = ConversationTurn(**turn_data)
            memory.turns.append(turn)

        return memory
