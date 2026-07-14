# Med-AI Starter

Đây là repo khởi đầu cho bài toán **trả lời câu hỏi trắc nghiệm y khoa** (Medical QA), sử dụng dataset [MedQA-USMLE-4-options](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options).

---

## 🧠 Tổng quan luồng xử lý (Phase v0)

Phase hiện tại (v0) là pipeline **đơn giản nhất**:

```
Input (câu hỏi + 4 đáp án)
        ↓
  Reasoning Agent (LLM)
        ↓
Output (đáp án A/B/C/D + giải thích + confidence)
```

Chưa có RAG (Retrieval), chưa có Verifier — chỉ 1 LLM suy luận thẳng ra đáp án.

---

## 🗺️ Cấu trúc thư mục

```
medai-starter/
├── main.py                  # Điểm vào chính, chọn chạy mode nào
├── state.py                 # Định nghĩa schema dữ liệu chảy qua pipeline
├── run_config.py            # Đọc file YAML config → đối tượng RunConfig
├── llm_client.py            # Khởi tạo LLM client (dùng chung toàn dự án)
├── graph.py                 # Kết nối các Agent thành pipeline (LangGraph)
│
├── agents/
│   └── reasoning_agent.py   # Agent duy nhất hiện tại: gọi LLM, parse kết quả
│
├── configs/
│   └── v0.yaml              # ⭐ File config chính — chỉnh prompt/model tại đây
│
├── modes/
│   ├── benchmark.py         # Mode 1: chạy hàng loạt, ghi predictions ra file
│   └── chat.py              # Mode 2: chat tự do 1 câu hỏi / 1 trả lời
│
├── predictions_v0_train.jsonl   # Kết quả dự đoán (tự sinh ra khi chạy)
├── gold_train.jsonl             # Đáp án đúng  (tự sinh ra khi chạy)
└── run_config_v0_train.json     # Snapshot config đã dùng (tự sinh ra khi chạy)
```

---

## ⚙️ Cài đặt

```bash
# 1. Tạo môi trường ảo
python3 -m venv venv
source venv/bin/activate

# 2. Cài thư viện
pip install -r requirements.txt

# 3. Cấu hình API key
cp env-example .env
# Mở file .env và điền thông tin thật vào
```

**Nội dung `.env` cần điền:**
```
NINE_ROUTER_BASE_URL=http://...   # URL endpoint LLM
NINE_ROUTER_API_KEY=...           # API key
LLM_MODEL=gemini-flash-latest     # Model mặc định cho Mode chat
```

---

## 🚀 Cách chạy

### Mode 1 — Benchmark (chạy hàng loạt trên dataset)

```bash
# Debug nhanh: chạy 20 câu đầu của split train
python main.py --mode benchmark --config configs/v0.yaml --split train --limit 20

# Chạy chính thức: toàn bộ 1273 câu test (chỉ chạy 1 lần khi đã chắc chắn config)
python main.py --mode benchmark --config configs/v0.yaml --split test
```

### Mode 2 — Chat (hỏi đáp tự do)

```bash
python main.py --mode chat
```

---

## 📂 Output sau khi chạy Benchmark

Mỗi lần chạy benchmark tự động tạo ra 3 file trong thư mục `output/`:

| File | Nội dung |
|---|---|
| `output/predictions_{variant}_{split}.jsonl` | Dự đoán của model: `question_id`, `question`, `choices`, `predicted_answer` (A/B/C/D hoặc INVALID), `explanation`, `confidence`, `latency_ms`, `token_usage`, v.v. |
| `output/gold_{split}.jsonl` | Đáp án đúng: `question_id` + `gold_answer` — **tách riêng**, không đưa vào pipeline |
| `output/run_config_{variant}_{split}.json` | Toàn bộ config đã dùng cho lần chạy này (model, prompt, temperature...) |

Ví dụ 1 dòng trong `predictions_v0_train.jsonl`:
```json
{
  "question_id": "train_0",
  "question": "A 23-year-old man presents with...",
  "choices": ["Option A text", "Option B text", "Option C text", "Option D text"],
  "predicted_answer": "D",
  "explanation": "...",
  "raw_output": "...",
  "latency_ms": 1234.5,
  "token_usage": {...},
  "estimated_cost": null,
  "retrieved_docs": null,
  "agent_trace": null
}
```

> **Lý do tách `gold` ra riêng:** Đảm bảo model không "nhìn thấy" đáp án đúng trong quá trình dự đoán. File gold chỉ dùng khi chấm điểm (`evaluate.py`).

---

## ✏️ Cách chỉnh sửa Prompt

Toàn bộ prompt nằm trong file `configs/v0.yaml`, **không cần sửa code Python**:

```yaml
# configs/v0.yaml
prompt_template: |
  Answer this question:

  Question:
  {question}

  Choices:
  {choices}

  Answer this question carefully and choose ONE best answer.
  Only return JSON in the following format:
  {{
    "answer": "A or B or C or D",
    "explanation": "explanation about the answer",
    "confidence": confidence from 0 to 1
  }}
```

**Lưu ý khi sửa prompt:**
- Giữ nguyên `{question}` và `{choices}` — chúng được điền tự động lúc chạy.
- Giữ nguyên `{{ }}` quanh ví dụ JSON (dấu ngoặc kép để `.format()` không bị nhầm).

---

## 🔧 Cách tạo biến thể mới (V1, V2, ...)

```bash
cp configs/v0.yaml configs/v1.yaml
# Sửa variant, prompt_template, model, temperature trong v1.yaml
python main.py --mode benchmark --config configs/v1.yaml --split train --limit 20
```

Output sẽ tự động tạo ra `predictions_v1_train.jsonl` và `run_config_v1_train.json` — **không đụng chạm** vào kết quả của v0.

---

## 🔬 Nguyên tắc quan trọng

1. **Chỉ có 2 split:** `train` và `test` (không có `dev`). Dùng `--split train --limit N` để debug; chỉ chạy `--split test` đúng **1 lần** khi config đã "khoá".
2. **Không rò rỉ gold answer:** `answer_idx` (đáp án đúng) không bao giờ được đưa vào pipeline → chỉ ghi ra `gold_{split}.jsonl`.
3. **`question_id` ổn định:** Sinh dạng `{split}_{index}` dựa vào thứ tự dòng của HuggingFace dataset — luôn nhất quán giữa các lần chạy.
4. **Config = snapshot:** `run_config_{variant}_{split}.json` lưu lại chính xác prompt + tham số đã dùng → phục vụ tái lập kết quả sau này.

---

## 🗓️ Roadmap

| Giai đoạn | Trạng thái | Mô tả |
|---|---|---|
| **v0 — Baseline** | ✅ Xong | Pipeline đơn giản: Input → Reasoning Agent → Output |
| **evaluate.py** | 🔜 Tiếp theo | Chấm điểm accuracy bằng cách join `predictions_v0_test.jsonl` với `gold_test.jsonl` theo `question_id` |
| **v1 — Retrieval** | 📋 Kế hoạch | Thêm Retrieval Agent (RAG) vào graph |
| **v2 — Verifier** | 📋 Kế hoạch | Thêm Verifier Agent để kiểm tra lại kết quả |

---

## 📝 Bước tiếp theo (cho thành viên mới)

1. **Chạy thử v0** trên 5–20 câu train để hiểu output.
2. **Viết `evaluate.py`** để tính accuracy từ `predictions_v0_test.jsonl` + `gold_test.jsonl`.
3. **Chạy v0 trên toàn bộ test set** (1273 câu) để có baseline chính thức.
4. Dựa vào baseline → thử nghiệm prompt mới hoặc thêm agent mới (v1, v2...).
