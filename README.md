# Med-AI Starter (Phiên bản v2 - Multi-agent: Retrieval → Reasoning → Verifier)

Đây là mã nguồn khởi đầu cho bài toán **trả lời câu hỏi trắc nghiệm y khoa** (Medical QA), sử dụng tập dữ liệu [MedQA-USMLE-4-options](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options).

Phiên bản hiện tại (v2) là một hệ **multi-agent tuần tự 3 bước**:
- **Retrieval Agent**: truy xuất bằng chứng từ ChromaDB (bge-m3), tự đánh giá evidence đã đủ chưa và có thể tự viết lại truy vấn để tìm thêm.
- **Reasoning Agent**: suy luận lâm sàng từ evidence, sinh đáp án + giải thích + confidence.
- **Verifier Agent**: kiểm chứng đáp án có thực sự được evidence hỗ trợ không, rồi chốt đáp án cuối cùng (không tự bịa đáp án khi thiếu bằng chứng).

Cả 3 agent đều bật/tắt độc lập qua file YAML (không sửa code) - v0 và v1 vẫn chạy y hệt như trước.

---

## 🧠 Tổng quan luồng xử lý

Hệ thống hỗ trợ cấu hình linh hoạt hai luồng xử lý chính thông qua file cấu hình YAML:

### 1. Luồng v0 (Baseline - Chỉ suy luận)
```
Yêu cầu (câu hỏi + 4 lựa chọn)
         ↓
   Reasoning Agent (LLM)
         ↓
Kết quả (đáp án A/B/C/D + giải thích + confidence)
```

### 2. Luồng v1 (RAG - Truy xuất và suy luận)
```
Yêu cầu (câu hỏi + 4 lựa chọn)
         ↓
  Retrieval Agent (Truy xuất ChromaDB bằng bge-m3)
         ↓
Điền evidence vào prompt template
         ↓
   Reasoning Agent (LLM)
         ↓
Kết quả (đáp án A/B/C/D + giải thích + confidence)
```

### 3. Luồng v2 (Multi-agent - Retrieval → Reasoning → Verifier)
```
Yêu cầu (câu hỏi + 4 lựa chọn)
         ↓
  Retrieval Agent ⟲ (truy xuất Chroma, tự đánh giá đủ chưa,
                      tự viết lại truy vấn nếu chưa đủ - tối đa N vòng)
         ↓
   Reasoning Agent (LLM sinh đáp án nháp + giải thích + confidence)
         ↓
  Verifier Agent (đối chiếu đáp án với evidence, chốt đáp án cuối,
                   hạ confidence/gắn cờ nếu không được evidence hỗ trợ)
         ↓
Kết quả (đáp án A/B/C/D + giải thích + confidence + verifier_verdict)
```

`retrieval.max_iterations=1` (mặc định) tắt hoàn toàn vòng tự đánh giá của Retrieval Agent -> luồng v1 chạy y hệt như cũ, không tốn thêm chi phí.
`verifier.enabled=false` (mặc định) tắt hoàn toàn Verifier Agent -> luồng v0/v1 không đổi.

---

## 🗺️ Cấu trúc thư mục

```
medai-starter-with-rag/
├── main.py                  # Điểm khởi chạy chính (chọn chạy benchmark hoặc chat)
├── state.py                 # Định nghĩa cấu trúc AgentState đi qua LangGraph
├── run_config.py            # Đọc cấu hình từ file YAML cấu hình chạy (RunConfig)
├── llm_client.py            # Khởi tạo LLM client OpenAI-compatible
├── graph.py                 # Định nghĩa và kết nối luồng xử lý (LangGraph)
│
├── agents/
│   ├── reasoning_agent.py   # Agent suy luận: gọi LLM để đưa ra đáp án nháp
│   ├── retrieval_agent.py   # Agent truy xuất: tìm bằng chứng + tự đánh giá/truy vấn lại (v2)
│   └── verifier_agent.py    # Agent kiểm chứng: đối chiếu đáp án với evidence, chốt đáp án (v2)
│
├── retrieval/
│   ├── ingest_data.py       # Pipeline lưu trữ dữ liệu giáo trình vào database Chroma
│   └── retriever.py         # Tìm kiếm bằng chứng tương đồng từ Chroma index
│
├── configs/
│   ├── v0.yaml              # Cấu hình baseline (không bật RAG)
│   ├── v1.yaml              # Cấu hình RAG (bật retrieval, k=5)
│   └── v2.yaml              # Cấu hình multi-agent (retrieval self-retry + verifier)
│
├── output/                  # Thư mục lưu kết quả chạy benchmark
│   ├── predictions_{variant}_{split}.jsonl  # Dự đoán của Agent
│   ├── gold_{split}.jsonl                    # Đáp án đúng (được tách riêng)
│   └── run_config_{variant}_{split}.json     # Bản lưu cấu hình của lượt chạy
└── requirements.txt         # Các thư viện phụ thuộc của dự án
```

---

## ⚙️ Cài đặt & Cấu hình

### 1. Cài đặt môi trường

Chạy các lệnh dưới đây trong terminal (trong môi trường WSL nếu chạy trên Windows):

```bash
# 1. Tạo môi trường ảo
python3 -m venv venv
source venv/bin/activate

# 2. Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### 2. Cấu hình file `.env`

Sao chép file cấu hình mẫu và điền thông tin thực tế:

```bash
cp env-example .env
```

**Nội dung `.env` cần thiết lập:**

```env
# Cấu hình mô hình suy luận (Reasoning Model)
REASONING_MODEL=gemini-3.5-flash-lite                   # Tên mô hình chính sử dụng
REASONING_MODEL_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai  # Endpoint OpenAI-compatible
REASONING_MODEL_API_KEY=your_reasoning_model_api_key    # API key của mô hình suy luận

# --- Optional (v2) - Tên mô hình riêng và thông tin endpoint cho các Agent phụ ---
# Set nếu muốn chạy Verifier hoặc Retrieval self-check bằng mô hình/provider riêng biệt.
# Nếu để trống (hoặc không định nghĩa BASE_URL/API_KEY), hệ thống tự động fallback sử dụng chung REASONING_MODEL.
VERIFIER_MODEL=gemini-3.5-flash-lite
VERIFIER_MODEL_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
VERIFIER_MODEL_API_KEY=your_verifier_model_api_key

RETRIEVAL_CHECK_MODEL=gemini-3.5-flash-lite
RETRIEVAL_CHECK_MODEL_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
RETRIEVAL_CHECK_MODEL_API_KEY=your_retrieval_check_model_api_key

# Cấu hình mô hình nhúng (Embedding Model) - Cần thiết cho RAG
EMBEDDING_MODEL=baai/bge-m3                             # Tên mô hình nhúng
EMBEDDING_MODEL_API_BASE=https://openrouter.ai/api/v1   # API base của mô hình nhúng
EMBEDDING_MODEL_API_KEY=your_embedding_api_key          # API key dùng để gọi API nhúng
```

---

## 💾 Xây dựng Vector Database (ChromaDB)

Nếu bạn đã có sẵn thư mục chứa cơ sở dữ liệu ChromaDB tại `data/chroma/...` (ví dụ: được giải nén từ file zip dữ liệu đã xây dựng sẵn), bạn có thể bỏ qua bước này.

Cơ sở dữ liệu ChromaDB đã được ingest sẵn: https://drive.google.com/file/d/1pIQYQ7CHJPbWqNW07kff5XA3YS8ER_Bb/view?usp=sharing

Nếu muốn xây dựng cơ sở dữ liệu từ đầu từ tài liệu thô:
1. Chuẩn bị file dữ liệu giáo trình tại `data/corpus/medrag_textbooks/textbooks.jsonl`.
2. Đảm bảo cấu hình `EMBEDDING_MODEL_API_KEY` đã được thiết lập đúng trong `.env`.
3. Chạy lệnh xây dựng vector database (quá trình này chạy offline và có thể mất vài giờ tùy thuộc kích thước tài liệu):

```bash
python3 -m retrieval.ingest_data 
```

---

## 🚀 Hướng dẫn chạy chương trình

### Mode 1 — Benchmark (Đánh giá chất lượng hàng loạt)

Benchmark chạy trên dataset MedQA-USMLE-4-options để đánh giá độ chính xác (Accuracy).

```bash
# Chạy thử nghiệm nhanh: Chạy 20 câu đầu tiên của tập train bằng luồng v0 (Không RAG)
python3 main.py --mode benchmark --config configs/v0.yaml --split train --limit 20

# Chạy thử nghiệm nhanh bằng luồng v1 (Có RAG)
python3 main.py --mode benchmark --config configs/v1.yaml --split train --limit 20

# Chạy chính thức: Toàn bộ 1273 câu của tập test (chỉ chạy khi cấu hình đã được khóa)
python3 main.py --mode benchmark --config configs/v1.yaml --split test
```

### Mode 2 — Chat (Hỏi đáp tự do)

Chế độ chat trực tiếp trong terminal để tương tác nhanh với mô hình:

```bash
# Chạy chat với cấu hình mặc định (configs/v0.yaml)
python3 main.py --mode chat

# Chạy chat với cấu hình bật RAG (configs/v1.yaml)
python3 main.py --mode chat --config configs/v1.yaml
```

---

## 📂 Dữ liệu đầu ra (Output)

Mỗi lượt chạy benchmark sẽ tự động tạo ra một thư mục riêng trong thư mục `output/` có định dạng `output/{variant}_{split}_{timestamp}/` chứa 3 file kết quả:

| File | Nội dung |
|---|---|
| `output/{variant}_{split}_{timestamp}/predictions_{variant}_{split}_{timestamp}.jsonl` | Chứa dự đoán của mô hình: câu hỏi, các lựa chọn, đáp án mô hình chọn (`predicted_answer`), giải thích, độ tin cậy, dữ liệu truy xuất (`retrieved_docs`), chi phí (`estimated_cost`), tổng thời gian / token của cả luồng (`latency_ms`, `token_usage`), và thông số chi tiết của từng agent (`retrieval_latency_ms`, `retrieval_token_usage`, `reasoning_latency_ms`, `reasoning_token_usage`, `verifier_latency_ms`, `verifier_token_usage`). |
| `output/{variant}_{split}_{timestamp}/gold_{split}_{timestamp}.jsonl` | Đáp án đúng thực tế của bộ câu hỏi. Được lưu riêng để làm cơ sở tính điểm mà không bị rò rỉ vào luồng suy luận. |
| `output/{variant}_{split}_{timestamp}/run_config_{variant}_{split}_{timestamp}.jsonl` | Bản lưu snapshot cấu hình đầy đủ của lượt chạy (model, prompt template, temperature...) dưới dạng một dòng JSON. |

---

## ✏️ Cách chỉnh sửa Prompt và Cấu hình Thử nghiệm

Bạn có thể thay đổi prompt trực tiếp trong các file cấu hình YAML tại thư mục `configs/` mà không cần sửa đổi mã nguồn Python.

---

## 🧩 Cấu hình Multi-agent (v2 - Retrieval self-retry + Verifier)

`configs/v2.yaml` bật cả 2 cơ chế mới, độc lập nhau:

```yaml
debug: true                   # Bật/tắt chế độ in log debug hiển thị quá trình chạy chi tiết của từng agent
retrieval:
  enabled: true
  top_k: 5
  max_iterations: 2           # >1 -> bật vòng tự đánh giá + truy vấn lại
  sufficiency_threshold: 0.6  # ngưỡng "đủ evidence" (0-1)
  check_model: null           # null -> dùng chung model+provider với Reasoning Agent hoặc cấu hình qua RETRIEVAL_CHECK_MODEL trong .env
  prompt_template: |          # Template prompt tự đánh giá độ đầy đủ của evidence (sửa trực tiếp trong YAML)
    You are grading whether the retrieved evidence below is...
    {question} ... {choices} ... {evidence}

verifier:
  enabled: true
  mode: "annotate_and_guard"  # xem giải thích bên dưới
  model: null                 # null -> dùng chung model+provider với Reasoning Agent hoặc cấu hình qua VERIFIER_MODEL trong .env
  prompt_template: |          # Template prompt kiểm chứng đáp án (sửa trực tiếp trong YAML)
    You are a careful medical fact-checker...
```

**Chế độ Debug (`debug: true`)**: khi bật, terminal sẽ in chi tiết câu hỏi đang gửi, agent nào đang xử lý, các vòng truy vấn lại của Retrieval (reformulated query), kết quả đánh giá sufficiency, và kết quả chốt/audit của Verifier.

**Retrieval self-retry (`retrieval.max_iterations`)**: sau khi lấy `top_k` evidence, nếu `max_iterations > 1`, agent gọi thêm 1 lời gọi LLM để tự chấm điểm "evidence đã đủ để trả lời chưa" (`sufficiency_score`). Nếu chưa đủ, agent tự viết lại truy vấn (`next_query`) và tìm lại, tối đa `max_iterations` lần, rồi gộp + khử trùng lặp toàn bộ evidence đã tìm được (giữ lại `top_k` evidence tốt nhất theo điểm tương đồng). Đặt `max_iterations: 1` để tắt hẳn cơ chế này (giống hệt hành vi v1, không tốn thêm chi phí). Bạn có thể chỉnh sửa prompt tự đánh giá trực tiếp qua trường `retrieval.prompt_template`.

**Verifier Agent (`verifier.enabled`)**: chạy sau Reasoning Agent, đối chiếu đáp án nháp với evidence đã truy xuất và trả về `verdict` (`supported`/`partial`/`unsupported`) + `support_score`. Với `mode: "annotate_and_guard"` (mặc định, khuyến nghị cho domain y tế): khi `verdict = unsupported`, Verifier **không được phép** tự đổi sang đáp án khác - chỉ hạ `confidence` xuống tối đa 0.3 và gắn cờ để phục vụ error analysis, tránh việc verifier "tự tin sai" thay vì báo không chắc. Dùng `mode: "annotate_only"` nếu chỉ muốn gắn nhãn/điểm mà không đổi confidence. Bạn có thể chỉnh sửa prompt kiểm chứng trực tiếp qua trường `verifier.prompt_template`.

**Dùng model/provider riêng cho Verifier hoặc Retrieval self-check (`verifier.model` / `retrieval.check_model`)**: mặc định (`null`) cả 2 bước phụ này dùng **chung endpoint + API key** với Reasoning Agent (đọc từ `REASONING_MODEL_BASE_URL`/`REASONING_MODEL_API_KEY`). Muốn verifier/retrieval-check gọi sang **1 provider/mô hình hoàn toàn khác**, bạn có thể cấu hình tên model trực tiếp trong YAML (`check_model` / `model`) hoặc đặt tên biến môi trường trong `.env` (`VERIFIER_MODEL` / `RETRIEVAL_CHECK_MODEL`) kèm base_url và api_key của chúng.

**Các field mới được ghi vào `predictions.jsonl` để phân tích:**
- **Thông số flow tổng:** `latency_ms` (tổng thời gian toàn bộ luồng), `token_usage` (tổng token tiêu thụ của cả 3 agent), `estimated_cost` (tổng chi phí ước tính).
- **Thông số chi tiết từng agent:**
  - **Retrieval:** `retrieval_latency_ms`, `retrieval_token_usage`, `query_history`, `retrieval_iterations`, `retrieval_sufficiency`.
  - **Reasoning:** `reasoning_latency_ms`, `reasoning_token_usage`.
  - **Verifier:** `verifier_latency_ms`, `verifier_token_usage`, `verifier_verdict`, `verifier_support_score`, `verifier_notes`.


Chạy thử nhanh để so sánh v1 vs v2:
```bash
python3 main.py --mode benchmark --config configs/v1.yaml --split train --limit 20
python3 main.py --mode benchmark --config configs/v2.yaml --split train --limit 20
```

---

## 🔧 Cách tạo biến thể thử nghiệm mới (v3...)

Để thử nghiệm một cấu hình hoặc prompt mới độc lập mà không ảnh hưởng đến các phiên bản cũ:

1. Tạo file cấu hình mới từ file mẫu:
   ```bash
   cp configs/v1.yaml configs/v2.yaml
   ```
2. Thay đổi giá trị trường `variant` bên trong file cấu hình mới (ví dụ: `variant: "v2"`), điều chỉnh prompt hoặc tham số nhiệt độ (`temperature`).
3. Chạy benchmark sử dụng file cấu hình mới:
   ```bash
   python3 main.py --mode benchmark --config configs/v2.yaml --split train --limit 20
   ```
Các file kết quả sẽ được lưu độc lập dưới tên dạng `predictions_v2_...` trong thư mục `output/`.

---

## 🔬 Các nguyên tắc phát triển cốt lõi

1. **Không rò rỉ đáp án đúng (No gold leakage):** Tuyệt đối không đưa đáp án đúng (`answer_idx`) vào bất kỳ nút nào trong luồng xử lý hoặc cấu trúc `AgentState`.
2. **Quản lý cấu hình dạng Snapshot:** Mọi tham số ảnh hưởng tới kết quả kiểm thử phải được lưu kèm trong file cấu hình của lượt chạy để đảm bảo tính tái lập.
3. **Xử lý lỗi ngoại lệ an toàn:** Lỗi phát sinh từ một câu hỏi đơn lẻ hoặc lỗi kết nối dịch vụ truy xuất (retrieval) không được làm dừng toàn bộ quá trình chạy benchmark.
4. **Phân tách chấm điểm:** Toàn bộ công việc chấm điểm và thống kê được thực hiện riêng biệt thông qua script đánh giá (ví dụ: `evaluate.py`), dựa trên việc kết hợp file `predictions` và file `gold` thông qua trường `question_id`.

---

## 🗓️ Lộ trình dự án (Roadmap)

| Giai đoạn | Trạng thái | Mô tả |
|---|---|---|
| **v0 — Baseline** | ✅ Hoàn thành | Luồng suy luận cơ bản: Input → Reasoning Agent → Output |
| **v1 — Retrieval (RAG)** | ✅ Hoàn thành | Luồng bổ sung tri thức: Input → Retrieval Agent (Chroma DB) → Reasoning Agent → Output |
| **v2 — Verifier** | ✅ Hoàn thành | Luồng multi-agent: Retrieval Agent (tự đánh giá + truy vấn lại) → Reasoning Agent → Verifier Agent kiểm chứng đáp án bằng evidence trước khi xuất dữ liệu |
| **v3 — Memory** | ⬜️ Chưa bắt đầu | Thực hiện lưu history của conversation lại để phục vụ cho việc hỏi các câu hỏi tiếp theo |
| **evaluate.py** | ⬜️ Chưa bắt đầu | Tạo file evaluate.py và run với số lượng dữ liệu lớn để có cơ sở so sánh và đánh giá kết quả chạy từng version |