"""Short-term memory — session buffer cho chat mode."""

from __future__ import annotations

from collections import deque


class SessionBuffer:
    """Giữ N lượt hội thoại gần nhất (scope=session)."""

    def __init__(self, max_turns: int = 6):
        if max_turns < 0:
            raise ValueError("max_turns must be >= 0")
        self.max_turns = max_turns
        self._buffer: deque[dict] = deque(maxlen=max_turns * 2 if max_turns > 0 else 0)

    def add_turn(self, user_content: str, assistant_content: str) -> None:
        if self.max_turns == 0:
            return
        self._buffer.append({"role": "user", "content": user_content})
        self._buffer.append({"role": "assistant", "content": assistant_content})

    def as_text(self) -> str:
        return "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in self._buffer
        )

    def as_messages(self) -> list[dict]:
        return list(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer) // 2
