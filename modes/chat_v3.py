"""Mode 2 - Chat (v3 with Memory + Full Agent Pipeline).

Flow:
1. LTM check (cache hit → return immediately)
2. Retrieval Agent (query RAG DB)
3. Reasoning Agent (generate answer with STM + LTM context)
4. Verifier Agent (check evidence support, add disclaimer if needed)
5. Post-processing (add to STM, maybe add to LTM)
"""

import time
from pathlib import Path
from typing import Optional

from memory.short_term_memory import ShortTermMemory

CHAT_SYSTEM_PROMPT = (
    "You are a medical AI assistant. Answer accurately and concisely. "
    "If the question involves symptoms or medication, provide general safe guidance "
    "and ALWAYS remind the user to consult a doctor for serious, persistent, or "
    "uncertain symptoms - this is not an official medical diagnosis."
)


def find_last_session(session_dir: Path) -> Optional[Path]:
    """Find the most recent session file."""
    if not session_dir.exists():
        return None

    session_files = list(session_dir.glob("session_*.json"))
    if not session_files:
        return None

    session_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return session_files[0]


def generate_session_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def handle_command(command: str, memory: ShortTermMemory, session_dir: Path, ltm=None, debug: bool = False) -> str:
    """Handle slash commands."""
    command = command.strip().lower()

    if command in ["/new", "/clear"]:
        memory.clear()
        memory.session_id = generate_session_id()
        return "Session cleared. Starting fresh."

    elif command == "/history":
        if memory.is_empty():
            return "No conversation history yet."
        history = memory.get_conversation_history_text(include_explanation=False)
        stats = memory.get_stats()
        return f"Conversation history ({stats['total_turns']} turns):\n{history}"

    elif command == "/stats":
        stats = memory.get_stats()
        parts = [f"Session Statistics:"]
        parts.append(f"  STM turns: {stats['total_turns']}")
        parts.append(f"  Mode: {stats['mode']} (max_turns={stats['max_turns']})")
        if stats["session_duration"]:
            parts.append(f"  Duration: {stats['session_duration'] / 60:.1f} min")
        if stats["avg_confidence"] is not None:
            parts.append(f"  Avg confidence: {stats['avg_confidence']:.2f}")

        if ltm:
            ltm_stats = ltm.get_stats()
            parts.append(f"\n  LTM cached: {ltm_stats['total_cached']}/{ltm_stats.get('max_cache_size', '?')}")
            parts.append(f"  LTM total hits: {ltm_stats.get('total_cache_hits', 0)}")
            if ltm_stats.get('avg_confidence'):
                parts.append(f"  LTM avg confidence: {ltm_stats['avg_confidence']:.2f}")

        return "\n".join(parts)

    elif command == "/save":
        if memory.session_id is None:
            memory.session_id = generate_session_id()
        session_path = session_dir / f"session_{memory.session_id}.json"
        memory.save(session_path)
        return f"Session saved: {session_path.name}"

    elif command.startswith("/load"):
        parts = command.split()
        if len(parts) > 1:
            session_file = session_dir / parts[1]
        else:
            session_file = find_last_session(session_dir)

        if session_file and session_file.exists():
            loaded = ShortTermMemory.load(session_file)
            memory.turns = loaded.turns
            memory.current_turn_id = loaded.current_turn_id
            memory.session_id = loaded.session_id
            memory.session_metadata = loaded.session_metadata
            return f"Loaded session: {session_file.name} ({len(loaded.turns)} turns)"
        else:
            return "No session file found."

    elif command == "/help":
        return """Available commands:
  /new, /clear  - Start new session (clear STM)
  /history      - View conversation history
  /stats        - View session + LTM statistics
  /save         - Save current session
  /load [file]  - Load session (default: most recent)
  /help         - Show this help
  /exit, /quit  - Exit chat"""

    elif command in ["/exit", "/quit"]:
        return None  # Signal to exit

    else:
        return f"Unknown command: {command}. Type /help for available commands."


def run_chat(config_path: str) -> None:
    from run_config import load_run_config
    from graph_chat import build_chat_graph

    run_config = load_run_config(config_path)
    memory_config = getattr(run_config, 'memory', None)
    debug = getattr(run_config, 'debug', False)

    # Build chat graph (Retrieval → Reasoning → Verifier)
    app = build_chat_graph(run_config)

    # Initialize STM
    memory = None
    session_dir = Path("memory/sessions")
    if memory_config and memory_config.short_term.enabled:
        max_turns = memory_config.short_term.max_turns
        session_dir = Path(memory_config.short_term.session_dir)
        session_dir.mkdir(parents=True, exist_ok=True)
        follow_up_keywords = memory_config.short_term.follow_up_detection.keywords

        memory = ShortTermMemory(
            max_turns=max_turns,
            follow_up_keywords=follow_up_keywords
        )

        print(f"=== Med-AI Chat v3 === ({run_config.model})")
        print(f"Pipeline: Retrieval({'ON' if run_config.retrieval.enabled else 'OFF'}) -> "
              f"Reasoning -> Verifier({'ON' if run_config.verifier.enabled else 'OFF'})")
        print(f"Memory: STM(max_turns={'unlimited' if not max_turns else max_turns})")

        # Auto-load last session
        if memory_config.short_term.auto_load_last_session:
            last_session = find_last_session(session_dir)
            if last_session:
                print(f"\nFound existing session: {last_session.name}")
                choice = input("  [1] Continue session  [2] Start new\nChoice: ").strip()
                if choice == "1":
                    memory = ShortTermMemory.load(last_session)
                    print(f"Loaded {len(memory.turns)} turns from previous session.")
                else:
                    memory.session_id = generate_session_id()
                    print(f"New session: {memory.session_id}")
            else:
                memory.session_id = generate_session_id()
        else:
            memory.session_id = generate_session_id()
    else:
        print(f"=== Med-AI Chat v3 === ({run_config.model})")
        print("Memory: disabled")

    # Initialize LTM
    ltm = None
    if memory_config and memory_config.long_term.enabled:
        from memory.long_term_memory import LongTermMemory
        ltm_config = memory_config.long_term.qa_cache
        ltm = LongTermMemory(
            collection_name=ltm_config.collection_name,
            chroma_dir=ltm_config.chroma_dir,
            min_confidence=ltm_config.min_confidence,
            min_verifier_verdict=ltm_config.min_verifier_verdict,
            similarity_threshold=ltm_config.similarity_threshold,
            max_cache_size=ltm_config.max_cache_size,
            cache_eviction_policy=ltm_config.cache_eviction_policy,
            debug=debug or (memory_config and memory_config.debug),
        )
        print(f"Memory: LTM(cached={ltm.collection.count()})")

    print("\nType /help for commands. Type /exit to quit.\n")

    # Chat loop
    while True:
        question = input("\nQuestion: ").strip()

        if not question:
            continue

        if question.lower() in ("exit", "quit"):
            _save_and_exit(memory, memory_config, session_dir)
            break

        # Handle commands
        if question.startswith('/'):
            if question.strip().lower() in ["/exit", "/quit"]:
                _save_and_exit(memory, memory_config, session_dir)
                break

            if memory is None:
                print("Memory not enabled. Commands unavailable.")
                continue

            result = handle_command(question, memory, session_dir, ltm, debug)
            if result is None:
                _save_and_exit(memory, memory_config, session_dir)
                break
            print(result)
            continue

        # === STEP 1: LTM Cache Check ===
        if ltm:
            cached = ltm.search_similar_qa(question)
            if cached:
                print(f"\n[From cache, similarity={cached['similarity']:.2f}]")
                print(f"\nAnswer: {cached['answer']}")

                # Still add to STM for conversation continuity
                if memory:
                    memory.add_turn(
                        question=question,
                        answer=cached["answer"],
                        explanation=cached.get("explanation"),
                        confidence=cached.get("confidence"),
                        verifier_verdict=cached.get("verifier_verdict"),
                    )
                continue

        # === STEP 2-4: Full Pipeline (Retrieval → Reasoning → Verifier) ===

        # Build conversation context from STM
        conversation_context = ""
        if memory and not memory.is_empty():
            history = memory.get_conversation_history_text(max_turns=5)
            is_follow_up = memory.detect_follow_up(question)

            if history:
                conversation_context = f"Previous conversation:\n{history}"
                if is_follow_up:
                    conversation_context += "\n\n(The user is asking a follow-up question.)"
                    if debug or (memory_config and memory_config.debug):
                        print(f"[STM] Detected follow-up question")

        # Build agent input state
        agent_input = {
            "question": question,
            "question_id": None,
            "choices": [],  # No choices in chat mode
            "answer": None,
            "explanation": None,
            "confidence": None,
            "raw_output": None,
            "latency_ms": None,
            "token_usage": None,
            "estimated_cost": None,
            "retrieved_docs": None,
            "agent_trace": None,
            "query_history": None,
            "retrieval_iterations": None,
            "retrieval_sufficiency": None,
            "verifier_verdict": None,
            "verifier_support_score": None,
            "verifier_notes": None,
            "retrieval_latency_ms": None,
            "retrieval_token_usage": None,
            "reasoning_latency_ms": None,
            "reasoning_token_usage": None,
            "verifier_latency_ms": None,
            "verifier_token_usage": None,
            # Chat-specific
            "conversation_context": conversation_context,
        }

        # Run pipeline
        try:
            result = app.invoke(agent_input)
        except Exception as e:
            print(f"\nError: {e}")
            if memory:
                memory.add_turn(question=question, answer=f"[Error: {e}]")
            continue

        answer = result.get("answer", "Unable to generate answer.")
        confidence = result.get("confidence")
        verdict = result.get("verifier_verdict")

        # Display
        if debug or (memory_config and memory_config.debug):
            print(f"\n[Pipeline] Latency: {result.get('latency_ms', 0):.0f}ms | "
                  f"Confidence: {confidence} | Verdict: {verdict}")

        print(f"\nAnswer: {answer}")

        # === STEP 5: Post-processing ===

        # Add to STM
        if memory:
            memory.add_turn(
                question=question,
                answer=answer,
                explanation=result.get("explanation"),
                confidence=confidence,
                retrieved_docs=result.get("retrieved_docs"),
                verifier_verdict=verdict,
            )

            if debug or (memory_config and memory_config.debug):
                stats = memory.get_stats()
                print(f"[STM] Turn {memory.current_turn_id - 1} added (total: {stats['total_turns']})")

        # Maybe add to LTM (if high quality)
        if ltm and confidence is not None:
            was_cached = ltm.add_qa(
                question=question,
                answer=result.get("answer", ""),  # Original answer without disclaimer
                explanation=result.get("explanation"),
                confidence=confidence,
                verifier_verdict=verdict,
            )
            if was_cached and (debug or (memory_config and memory_config.debug)):
                print(f"[LTM] Answer cached for future reuse")


def _save_and_exit(memory, memory_config, session_dir):
    """Save session and exit."""
    if memory and memory_config and memory_config.short_term.persistence:
        if not memory.is_empty():
            session_path = session_dir / f"session_{memory.session_id}.json"
            memory.save(session_path)
            print(f"Session saved: {session_path.name}")
    print("Goodbye!")
