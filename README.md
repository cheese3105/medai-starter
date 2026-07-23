# Med-AI Starter (Phiên bản v1 - Tích hợp RAG)

Đây là mã nguồn khởi đầu cho bài toán **trả lời câu hỏi trắc nghiệm y khoa** (Medical QA), sử dụng tập dữ liệu [MedQA-USMLE-4-options](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options).

Phiên bản hiện tại (v1) đã được tích hợp thêm cơ chế **Retrieval-Augmented Generation (RAG)** để truy xuất kiến thức y khoa từ giáo trình làm bằng chứng (evidence) hỗ trợ quá trình suy luận của mô hình.

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
│   ├── reasoning_agent.py   # Agent suy luận: gọi LLM để đưa ra đáp án cuối cùng
│   └── retrieval_agent.py   # Agent truy xuất: tìm bằng chứng từ vector database
│
├── retrieval/
│   ├── ingest_data.py       # Pipeline lưu trữ dữ liệu giáo trình vào database Chroma
│   └── retriever.py         # Tìm kiếm bằng chứng tương đồng từ Chroma index
│
├── configs/
│   ├── v0.yaml              # Cấu hình baseline (không bật RAG)
│   └── v1.yaml              # Cấu hình RAG (bật retrieval, k=5)
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
| `output/{variant}_{split}_{timestamp}/predictions_{variant}_{split}_{timestamp}.jsonl` | Chứa dự đoán của mô hình: câu hỏi, các lựa chọn, đáp án mô hình chọn (`predicted_answer`), giải thích, độ tin cậy, dữ liệu truy xuất (`retrieved_docs`), và chi phí / thời gian phản hồi. |
| `output/{variant}_{split}_{timestamp}/gold_{split}_{timestamp}.jsonl` | Đáp án đúng thực tế của bộ câu hỏi. Được lưu riêng để làm cơ sở tính điểm mà không bị rò rỉ vào luồng suy luận. |
| `output/{variant}_{split}_{timestamp}/run_config_{variant}_{split}_{timestamp}.jsonl` | Bản lưu snapshot cấu hình đầy đủ của lượt chạy (model, prompt template, temperature...) dưới dạng một dòng JSON. |

---

## ✏️ Cách chỉnh sửa Prompt và Cấu hình Thử nghiệm

Bạn có thể thay đổi prompt trực tiếp trong các file cấu hình YAML tại thư mục `configs/` mà không cần sửa đổi mã nguồn Python:

```yaml
# configs/v1.yaml
variant: "v1"
prompt_version: "v1_rag"
temperature: 0.0
seed: null

prompt_template: |
  Use the retrieved evidence below if it is relevant to the question.
  If the evidence is irrelevant or conflicting, ignore it and answer from
  your own medical knowledge instead.

  Retrieved evidence:
  {evidence}

  Question:
  {question}

  Choices:
  {choices}

  Answer this question carefully and choose ONE best answer.
  Only return JSON in the following format, do not add any other text outside the JSON:
  {{
    "answer": "A or B or C or D",
    "explanation": "explanation about the answer",
    "confidence": confidence from 0 to 1
  }}

retrieval:
  enabled: true
  top_k: 5
```

**Lưu ý quan trọng khi sửa đổi prompt:**
- Giữ nguyên các thẻ đặt chỗ `{question}`, `{choices}`.
- Với luồng RAG, cần giữ lại thẻ `{evidence}`.
- Giữ nguyên cú pháp ngoặc nhọn kép `{{ }}` bao bọc phần cấu trúc JSON mẫu của câu trả lời để tránh lỗi biên dịch chuỗi `.format()`.

---

## 🔧 Cách tạo biến thể thử nghiệm mới (v2, v3...)

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
| **v2 — Verifier** | 📋 Lên kế hoạch | Luồng tối ưu hóa độ chính xác bằng cách thêm Verifier Agent kiểm chứng đáp án trước khi xuất dữ liệu |
