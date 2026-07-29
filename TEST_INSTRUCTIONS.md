# 🧪 HƯỚNG DẪN TEST MEMORY SYSTEM

## Phase 1: Short-term Memory (STM)

### **Chuẩn bị:**
```bash
# Đảm bảo đã có .env file với API key
# Kiểm tra configs/v3.yaml đã tạo
```

---

## TEST 1: Basic Conversation với STM

**Mục đích:** Test STM giữ conversation context và detect follow-up

**Các bước:**

1. **Chạy chat mode với v3 config:**
```bash
python main.py --mode chat --config configs/v3.yaml
```

2. **Tạo session mới (nếu hỏi):**
   - Nếu không có session cũ → tự động tạo mới
   - Nếu có session cũ → chọn [2] để tạo session mới

3. **Test conversation có follow-up:**

```
Câu hỏi: Triệu chứng của viêm phổi là gì?
→ Bot trả lời về triệu chứng viêm phổi

Câu hỏi: Điều trị nó như thế nào?
→ [STM] Detected follow-up question (debug log)
→ Bot trả lời về điều trị viêm phổi (có context từ câu trước)

Câu hỏi: Còn cách nào khác không?
→ [STM] Detected follow-up question
→ Bot trả lời thêm cách điều trị khác
```

**Expected behavior:**
- ✅ Debug log show "[STM] Detected follow-up"
- ✅ Bot answer có liên quan đến câu hỏi trước
- ✅ Không cần repeat "viêm phổi" trong câu hỏi follow-up

---

## TEST 2: Commands

**Test các commands:**

```
/history
→ Show 3 turns vừa hỏi

/stats
→ Show: total turns, duration, avg confidence

/save
→ "💾 Đã lưu session: session_20240115_..."

/help
→ Show list of commands

/new
→ "✅ Đã tạo session mới"
→ /history → "📭 Chưa có lịch sử" (đã clear)
```

**Expected:**
- ✅ Mỗi command hoạt động đúng
- ✅ /new clear STM
- ✅ /save tạo file trong memory/sessions/

---

## TEST 3: Session Persistence

**Test save/load session:**

1. **Hỏi vài câu:**
```
Q: Triệu chứng của viêm phổi?
Q: Điều trị nó thế nào?
Q: Tác dụng phủ của aspirin?
```

2. **Lưu và thoát:**
```
/save
/exit
→ "💾 Đã lưu session: session_xxx.json"
```

3. **Kiểm tra file đã lưu:**
```bash
ls memory/sessions/
cat memory/sessions/session_xxx.json
```

**Expected:**
- ✅ File JSON có 3 turns
- ✅ Mỗi turn có: turn_id, question, answer, timestamp

4. **Chạy lại và load session:**
```bash
python main.py --mode chat --config configs/v3.yaml
```

```
💾 Tìm thấy session: session_xxx.json
  [1] Tiếp tục session cũ  [2] Tạo session mới
Chọn: 1
✅ Đã load 3 turns từ session cũ
```

5. **Xem history:**
```
/history
→ Show 3 turns đã load
```

6. **Hỏi follow-up:**
```
Câu hỏi: Còn thuốc nào khác?
→ Bot biết context (từ session đã load)
```

**Expected:**
- ✅ Session được load đúng
- ✅ Follow-up vẫn work với context từ session cũ
- ✅ Có thể tiếp tục conversation

---

## TEST 4: Sliding Window Mode

**Test max_turns limit:**

1. **Sửa configs/v3.yaml:**
```yaml
memory:
  short_term:
    max_turns: 3  # Chỉ giữ 3 turns
```

2. **Chạy lại:**
```bash
python main.py --mode chat --config configs/v3.yaml
/new  # Tạo session mới
```

3. **Hỏi 5 câu:**
```
Q1: Triệu chứng viêm phổi?
Q2: Điều trị viêm phổi?
Q3: Triệu chứng tiểu đường?
Q4: Điều trị tiểu đường?
Q5: Triệu chứng cao huyết áp?
```

4. **Check history:**
```
/history
→ Chỉ show Q3, Q4, Q5 (3 turns gần nhất)
→ Q1, Q2 đã bị remove (sliding window)
```

**Expected:**
- ✅ Chỉ giữ 3 turns gần nhất
- ✅ Old turns bị auto-remove

---

## TEST 5: Follow-up Detection

**Test keyword detection:**

**Positive cases (should detect):**
```
"Điều trị nó thế nào?" → detect "nó"
"Còn cách nào khác?" → detect (implicit)
"Cái đó có nguy hiểm không?" → detect "cái đó"
"Ở trên bạn nói gì?" → detect "ở trên"
"What about it?" → detect "it"
"Is this dangerous?" → detect "this"
```

**Negative cases (should NOT detect):**
```
"Triệu chứng viêm phổi?" → new topic
"Triệu chứng tiểu đường?" → different topic
"Cách phòng tránh COVID?" → completely new
```

**Expected:**
- ✅ Positive cases → "[STM] Detected follow-up"
- ✅ Negative cases → NO debug log
- ✅ Context chỉ inject khi detect follow-up

---

## TEST 6: Edge Cases

**Test 1: STM empty at start**
```bash
python main.py --mode chat --config configs/v3.yaml
/new

Câu hỏi: Điều trị nó thế nào?
→ Bot trả lời nhưng KHÔNG có context (STM empty)
→ Debug log: NO "[STM] Detected follow-up" (vì empty)
```

**Test 2: Exit without save**
```bash
# Hỏi vài câu
/exit  # Không /save

# Chạy lại
python main.py --mode chat --config configs/v3.yaml
→ Không thấy session (vì chưa save)
```

**Test 3: Auto-save on exit**
```bash
# configs/v3.yaml: persistence: true

# Hỏi vài câu
/exit (hoặc gõ "exit")

→ "💾 Đã lưu session: session_xxx.json" (auto-save)

# Chạy lại
→ Thấy session (đã auto-save)
```

---

## ✅ CHECKLIST TEST

Phase 1 (STM Basic):
- [ ] Bot chạy được với v3.yaml
- [ ] Debug log show "[STM] Detected follow-up"
- [ ] Bot trả lời đúng context với follow-up
- [ ] /history show correct turns
- [ ] /stats show correct stats
- [ ] /save tạo file JSON
- [ ] /new clear STM
- [ ] Session persistence work (save/load)
- [ ] Sliding window work (max_turns)
- [ ] Auto-save on exit work

---

## 🐛 TROUBLESHOOTING

**Lỗi 1: Module not found**
```bash
ModuleNotFoundError: No module named 'memory'
```
→ Check: memory/__init__.py có tồn tại không?

**Lỗi 2: Config parse error**
```bash
KeyError: 'memory'
```
→ Check: configs/v3.yaml có section memory không?
→ Check: run_config.py đã parse memory config chưa?

**Lỗi 3: Session file not found**
```bash
FileNotFoundError: memory/sessions/session_xxx.json
```
→ Check: folder memory/sessions/ đã tạo chưa?
→ Run: `mkdir -p memory/sessions`

**Lỗi 4: Follow-up không detect**
```bash
# Hỏi "Điều trị nó thế nào?" nhưng không thấy debug log
```
→ Check: configs/v3.yaml có `debug: true` chưa?
→ Check: configs/v3.yaml có `memory.debug: true` chưa?

---

## 📊 EXPECTED OUTPUT EXAMPLES

**Example 1: Successful conversation**
```
$ python main.py --mode chat --config configs/v3.yaml

=== Med-AI Chat v3 (with Memory) === (gemini-3.5-flash-lite)
Memory: STM enabled (max_turns=unlimited)
✅ Session mới: session_20240115_143022

Gõ /help để xem commands. Gõ /exit để thoát.

Câu hỏi: Triệu chứng của viêm phổi là gì?
[Debug] Latency: 1234ms

Trả lời: Triệu chứng viêm phổi bao gồm: sốt cao, ho có đờm, khó thở, đau ngực khi thở sâu hoặc ho. Nên đi khám bác sĩ nếu triệu chứng nặng.

[STM] Added turn 0 (total: 1 turns)

Câu hỏi: Điều trị nó như thế nào?
[STM] Detected follow-up question
[Debug] Latency: 1100ms

Trả lời: Điều trị viêm phổi phụ thuộc mức độ: nhẹ có thể uống kháng sinh tại nhà, nặng cần nhập viện. Cần nghỉ ngơi, uống nhiều nước. Phải theo đơn bác sĩ.

[STM] Added turn 1 (total: 2 turns)

Câu hỏi: /history

📜 Lịch sử conversation (2 turns):
Q0: Triệu chứng của viêm phổi là gì?
A0: Triệu chứng viêm phổi bao gồm: sốt cao...
Q1: Điều trị nó như thế nào?
A1: Điều trị viêm phổi phụ thuộc mức độ...

Câu hỏi: /exit
💾 Đã lưu session: session_20240115_143022.json
Tạm biệt!
```

**Example 2: Load session**
```
$ python main.py --mode chat --config configs/v3.yaml

💾 Tìm thấy session: session_20240115_143022.json
  [1] Tiếp tục session cũ  [2] Tạo session mới
Chọn: 1
✅ Đã load 2 turns từ session cũ

Câu hỏi: /history
📜 Lịch sử conversation (2 turns):
Q0: Triệu chứng của viêm phổi là gì?
A0: ...
Q1: Điều trị nó như thế nào?
A1: ...
```

---

## 🎯 SUCCESS CRITERIA

Phase 1 thành công nếu:
1. ✅ All tests pass
2. ✅ No crashes
3. ✅ Follow-up detection works
4. ✅ Session persistence works
5. ✅ Commands work correctly
6. ✅ Debug logs are clear

Nếu tất cả OK → Ready for Phase 2! 🚀
