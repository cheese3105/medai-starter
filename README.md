# MED-AI — Framework Đánh Giá & Chat Y Khoa Đa Tầng (V0 - V3)

MED-AI là một framework thử nghiệm và đánh giá các hệ thống trí tuệ nhân tạo y khoa theo kiến trúc **Modular Pipeline & Multi-Agent**. Hệ thống được thiết kế để đo lường, so sánh hiệu năng qua từng giai đoạn tiến hoá (từ V0 đến V3), hỗ trợ cả chế độ đánh giá tự động (Benchmark trên bộ dữ liệu **MedQA-USMLE**) và chế độ trò chuyện tương tác (Interactive Chat REPL).

---

## 📌 1. Tiến Trình Tiến Hoá (V0 ➔ V3)

Framework được thiết kế với 4 phiên bản kiến trúc chính:

- **V0 — Direct LLM Reasoning**:
  - Mô hình suy luận trực tiếp (Direct Zero-shot / Few-shot Reasoning).
  - Không truy xuất tài liệu bên ngoài, không kiểm chứng, không bộ nhớ.
- **V1 — RAG Evidence Retrieval**:
  - Tích hợp kỹ thuật **Retrieval-Augmented Generation (RAG)**.
  - Truy xuất các đoạn văn bản/sách giáo khoa y khoa từ **ChromaDB vector store** làm minh chứng (`evidence`) để bổ sung ngữ cảnh trước khi suy luận.
- **V2 — Self-Correction & Query Rewriting**:
  - Thêm vòng lặp phản biện (**Verifier Agent**).
  - Verifier kiểm chứng câu trả lời nháp (`draft_answer`) so với minh chứng.
  - Nếu câu trả lời chưa vững chắc (`verdict: unsupported`), **Query Rewriter Agent** sẽ viết lại câu hỏi tìm kiếm để truy xuất lại tài liệu mới và lặp lại quá trình cho đến khi đạt yêu cầu hoặc chạm ngưỡng `max_iterations`.
- **V3 — Memory-Augmented System (STM & LTM)**:
  - Tích hợp hệ thống **Bộ nhớ ngắn hạn (Short-Term Memory - STM)** cho ngữ cảnh vòng lặp (`loop`) và phiên làm việc (`session`).
  - Tích hợp **Bộ nhớ dài hạn (Long-Term Memory - LTM)** lưu trữ bền vững trong ChromaDB: tự động trích xuất các thông tin y khoa quan trọng của người dùng (tiền sử bệnh, dị ứng, thuốc đang dùng, v.v.) qua các cuộc hội thoại để cá nhân hoá câu trả lời trong tương lai.

---

## 📁 2. Cấu Trúc Thư Mục Project

```text
medai/
├── configs/                # Tệp cấu hình thí nghiệm YAML (v0.yaml -> v3-chat.yaml)
│   ├── v0.yaml             # Config V0 Direct Reasoning
│   ├── v1.yaml             # Config V1 RAG
│   ├── v2.yaml / v2-qr.yaml# Config V2 Self-Correction & Query Rewriter
│   ├── v3.yaml / v3-qr.yaml# Config V3 Benchmark với LTM/STM
│   └── v3-chat.yaml        # Config V3 dành riêng cho Chat Mode
├── core/                   # Mô-đun lõi của hệ thống
│   ├── config.py           # Load & validate cấu hình YAML + biến môi trường .env
│   ├── llm_client.py       # Khởi tạo LLM client (qua LangChain / OpenAI API)
│   ├── logger.py           # Logger hệ thống & Ghi nhận kết quả dự đoán JSONL
│   ├── runner.py           # Controller Loop thực thi pipeline cho mỗi Episode
│   └── types.py            # Khai báo schema dữ liệu (EpisodeInput, EpisodeResult, StageOutput)
├── data/                   # Thư mục chứa dữ liệu Vector Store (ChromaDB)
│   └── chroma/             # Cơ sở dữ liệu vector sách y khoa & LTM
├── evaluate/               # Công cụ tính toán & so sánh metrics giữa các biến thể
│   └── evaluate.py         # Đọc predictions .jsonl -> accuracy, McNemar, bootstrap CI, report.md
├── memory/                 # Hệ thống quản lý bộ nhớ
│   ├── short_term.py       # Session Buffer giữ N lượt hội thoại gần nhất
│   └── long_term.py        # Quản lý & trích xuất bộ nhớ dài hạn với ChromaDB
├── modes/                  # Chế độ vận hành chính của ứng dụng
│   ├── benchmark.py        # Chế độ chạy đánh giá tự động trên MedQA-USMLE
│   └── chat.py             # Chế độ Chat REPL tương tác trực tiếp
├── output/                 # Nơi lưu trữ các tệp kết quả dự đoán (.jsonl) và trace log
├── retrieval/              # Mô-đun truy xuất dữ liệu RAG
│   └── retriever.py        # Wrapper kết nối ChromaDB & gọi HTTP Embedding API
├── stages/                 # Các giai đoạn thực thi (Pipeline Stages)
│   ├── base.py             # Abstract Class định nghĩa Stage
│   ├── reasoning.py        # Stage suy luận y khoa
│   ├── retrieval.py        # Stage truy xuất tài liệu
│   ├── verifier.py         # Stage kiểm chứng & phản biện
│   └── query_rewriter.py   # Stage viết lại truy vấn tìm kiếm
├── .env.example            # Tệp mẫu khai báo biến môi trường
├── main.py                 # CLI Entrypoint chính của ứng dụng
├── pyrightconfig.json      # Cấu hình type-checking Python
└── requirements.txt        # Danh sách thư viện phụ thuộc
```

---

## 🛠️ 3. Hướng Dẫn Cài Đặt (Installation)

### Yêu cầu hệ thống:
- **Python**: `>= 3.10`
- API Key / LLM Endpoint tương thích OpenAI format (Ollama, vLLM, OpenRouter, v.v.)

### Các bước thực hiện:

1. **Tạo môi trường ảo (Virtual Environment) & Kích hoạt**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate        # Trên Linux/macOS
   # Hoặc trên Windows PowerShell:
   # .venv\Scripts\Activate.ps1
   ```

2. **Cài đặt các thư viện phụ thuộc**:
   ```bash
   pip install -r requirements.txt
   ```
   (bao gồm cả `numpy`/`scipy` — cần cho `evaluate/evaluate.py` khi tính McNemar's test & bootstrap CI, xem mục 5)

3. **Cấu hình biến môi trường (`.env`)**:
   Sao chép tệp mẫu `.env.example` thành `.env`:
   ```bash
   cp .env.example .env
   ```
   Chỉnh sửa các giá trị trong `.env` phù hợp với mô hình của bạn:
   ```env
    # --- Reasoning Model (Bắt buộc) ---
    # Hỗ trợ tốt các mô hình Reasoning / CoT như DeepSeek Flash, DeepSeek R1, GPT-4o
    REASONING_MODEL=deepseek/deepseek-v4-flash
    REASONING_MODEL_BASE_URL=https://openrouter.ai/api/v1
    REASONING_MODEL_API_KEY=your-openrouter-api-key

    # --- Verifier Model (Tùy chọn, mặc định fallback về REASONING_MODEL) ---
    # Có thể tách riêng model nhẹ không reasoning (deepseek-chat) để tiết kiệm token và tăng tốc
    # VERIFIER_MODEL=deepseek/deepseek-chat
    # VERIFIER_MODEL_BASE_URL=https://openrouter.ai/api/v1
    # VERIFIER_MODEL_API_KEY=your-api-key

    # --- Embedding Model (Bắt buộc cho V1+ RAG và V3 LTM) ---
    EMBEDDING_MODEL=baai/bge-m3
    EMBEDDING_MODEL_BASE_URL=https://openrouter.ai/api/v1
    EMBEDDING_MODEL_API_KEY=your-openrouter-api-key

    # --- Query Rewriter Model (Tùy chọn, mặc định fallback về REASONING_MODEL) ---
    # QUERY_REWRITER_MODEL=deepseek/deepseek-chat
    # QUERY_REWRITER_MODEL_BASE_URL=https://openrouter.ai/api/v1
    # QUERY_REWRITER_MODEL_API_KEY=your-api-key
    ```
4. **Tải dữ liệu ChromaDB cho RAG**:
Cơ sở dữ liệu ChromaDB đã được ingest sẵn: https://drive.google.com/file/d/1pIQYQ7CHJPbWqNW07kff5XA3YS8ER_Bb/view?usp=sharing

Giải nén file zip dữ liệu đã xây dựng sẵn
Copy đường dẫn `data/chroma/...` bỏ vào `medai/data/chroma/...`

---

## 🚀 4. Hướng Dẫn Sử Dụng (How to Run)

Ứng dụng chạy thông qua file [main.py](main.py) với 2 chế độ chính: `--mode benchmark` và `--mode chat`.

### A. Chế Độ Benchmark (Chạy Đánh Giá Tự Động)

Chế độ này tải bộ dữ liệu `GBaker/MedQA-USMLE-4-options` từ HuggingFace Datasets, đưa từng câu hỏi qua Pipeline và tính toán độ chính xác (Accuracy), latency, token usage.

Hệ thống hỗ trợ **Multi-threading (chạy song song đa luồng)**, giúp giảm thời gian chạy từ hàng tiếng đồng hồ xuống còn **vài phút** (nhanh gấp ~15 - 20 lần).

**Cú pháp chung**:
```bash
python main.py --mode benchmark --config <duong_dan_file_config> [cac_tham_so_bổ_sung]
```

**Các tham số CLI tùy chọn**:
- `--split`: Tập dữ liệu (`test` hoặc `train`, mặc định: `test`).
- `--limit`: Giới hạn số lượng câu hỏi cần chạy (ví dụ `--limit 200`).
- `--workers`: Số luồng chạy song song (mặc định: `8`). Đề xuất: `8` để đạt tốc độ tối đa không bị nghẽn API.
- `--output`: Đường dẫn tùy chỉnh tệp lưu kết quả `.jsonl` (mặc định lưu tự động vào `output/predictions_<variant>_<timestamp>.jsonl`).

**Các câu lệnh mẫu (Chạy 200 câu với 8 workers song song)**:

1. **Chạy V0 (Direct Reasoning)**:
   ```bash
   python main.py --mode benchmark --config configs/v0.yaml --split test --limit 200 --workers 8
   ```

2. **Chạy V1 (RAG Evidence Retrieval)**:
   ```bash
   python main.py --mode benchmark --config configs/v1.yaml --split test --limit 200 --workers 8
   ```

3. **Chạy V2-QR (Verification + Query Rewriter)**:
   ```bash
   python main.py --mode benchmark --config configs/v2-qr.yaml --split test --limit 200 --workers 8
   ```

4. **Chạy V3-QR (Toàn bộ Pipeline + RAG + Verifier + Query Rewriter + LTM Read-Only)**:
   ```bash
   python main.py --mode benchmark --config configs/v3-qr.yaml --split test --limit 200 --workers 8
   ```

#### ⚡ Chạy tự động toàn bộ Benchmark Suite (`run_benchmark_suite.py`)
Để chạy liên tục các biến thể, tự động đo thời gian thực thi (wall-clock time) và xuất báo cáo so sánh chi tiết:
```bash
python run_benchmark_suite.py
```
Script sẽ tự động chạy các biến thể, gọi `evaluate.py` và tạo báo cáo `BENCHMARK_REPORT_OPTIMIZED.md`.

---

### B. Chế Độ Chat Tương Tác (Interactive Chat REPL)

Chế độ này mở một giao diện dòng lệnh (REPL) cho phép người dùng trò chuyện trực tiếp với hệ thống y khoa. Hệ thống sẽ áp dụng bộ nhớ ngắn hạn (STM) cho phiên làm việc và tự động lưu/truy xuất thông tin dài hạn (LTM).

**Câu lệnh thực thi**:
```bash
python main.py --mode chat --config configs/v3-chat.yaml
```

**Ví dụ tương tác**:
```text
=== MED-AI Chat === (your-model-name | memory: STM(session, 6 turns) + LTM(read_write))
Nhập 'exit' để thoát.

Câu hỏi: Tôi bị dị ứng với Penicillin và bị hen suyễn từ nhỏ.
Trả lời: Cảm ơn bạn đã chia sẻ. Tôi đã ghi nhận thông tin bạn bị dị ứng với Penicillin và có tiền sử bệnh hen suyễn...

Câu hỏi: Tôi đang bị đau họng, bác sĩ có thể kê đơn thuốc kháng sinh được không?
Trả lời: Dựa trên tiền sử dị ứng Penicillin của bạn, chúng ta tuyệt đối không sử dụng nhóm kháng sinh Penicillin...
```

---

## 📈 5. Đánh Giá & So Sánh Các Biến Thể (`evaluate/evaluate.py`)

Sau khi chạy Benchmark ra các file `.jsonl` trong `output/`, dùng `evaluate/evaluate.py` để tính accuracy, invalid rate, so sánh cặp biến thể (win/loss/tie, McNemar's test, bootstrap 95% CI) và ước tính cost/latency.

**Cú pháp**:
```bash
# Chỉ định file cụ thể
python evaluate/evaluate.py --files output/predictions_v0_*.jsonl output/predictions_v3-qr_*.jsonl

# Hoặc tự quét toàn bộ *.jsonl trong 1 thư mục
python evaluate/evaluate.py --dir output --baseline v0 --report report.md
```

**Các tham số chính**:
- `--files` / `--dir`: chọn file cụ thể (hỗ trợ glob) hoặc quét cả thư mục.
- `--baseline`: tên variant làm mốc so sánh (mặc định: variant có chứa `"v0"`).
- `--official-n`: dùng số câu chính thức của test set (vd `1273` cho MedQA-USMLE full) làm mẫu số accuracy, thay vì số câu thực chạy được.
- `--limit-first-n`: chỉ giữ N câu hỏi đầu tiên (theo `question_id`) ở mọi biến thể để so sánh công bằng khi các file có số câu khác nhau.
- `--out-dir`: thư mục lưu output (mặc định thư mục hiện tại).
- `--report`: tên file Markdown report (để trống `""` nếu không cần).

**Output sinh ra**:
- In bảng tổng hợp accuracy/invalid-rate/latency/cost ra console.
- `summary_metrics.csv` — metrics tổng hợp theo từng variant.
- `pairwise_comparison.csv` — win/loss/tie, McNemar p-value, bootstrap CI giữa baseline và từng biến thể còn lại.
- `report.md` (tùy chọn) — báo cáo Markdown đầy đủ kèm phân tích lỗi (câu được sửa đúng / bị hồi quy so với baseline).
- `full_results.json` — toàn bộ kết quả raw để debug/phân tích thêm.

> Lưu ý: giá cost trong `MODEL_PRICING_PER_1M_TOKENS` (đầu file `evaluate.py`) đang để trống — chỉnh lại theo model bạn dùng nếu muốn script tự ước tính chi phí (khi không có, cột cost hiển thị `N/A`, trừ khi prediction đã có sẵn `estimated_cost` từ `pricing` trong config YAML).

---

## 🔄 6. Luồng Hoạt Động Của Hệ Thống (System Execution Flow)

Hệ thống hoạt động theo mô hình **Controller Loop** được quản lý trong `core/runner.py`. Từ V2-QR/V3-QR trở đi, pipeline theo kiểu **Query-First**: Query Rewriter luôn chạy ĐẦU mỗi iteration (kể cả iteration 1, để hình thành query ban đầu từ câu hỏi gốc) chứ không phải chỉ sau khi Verifier từ chối:

```mermaid
flowchart TD
    A[Bắt đầu Episode hoặc Lượt Chat] --> B[Nạp Ngữ Cảnh Bộ Nhớ: STM và LTM]
    B --> C{Bắt đầu Iteration 1..N}
    C --> D{Query Rewriter Enabled?}
    D -- Có --> E["Sinh/viết lại retrieval_query<br/>iter 1: từ question gốc<br/>iter 2+: từ evidence thất bại + lý do reject"]
    D -- Không --> F{Retrieval Enabled?}
    E --> F
    F -- Có --> G[Query Chroma bằng retrieval_query, lấy Evidence]
    F -- Không --> H[Reasoning Stage]
    G --> H
    H --> I["Tạo Draft Answer + Explanation + Confidence<br/>luôn dùng question gốc"]
    I --> J{Verifier Enabled?}
    J -- Không --> Z[Trả về Kết Quả Cuối Cùng]
    J -- Có --> K[Verifier Stage: Fact-check Draft Answer]
    K --> L{Verdict == supported?}
    L -- Supported --> Z
    L -- Unsupported --> M["Lưu evidence thất bại + lý do reject vào context<br/>ghi loop_scratchpad nếu STM scope loop bật"]
    M --> N{Chạm max_iterations?}
    N -- Chưa --> C
    N -- Rồi --> Z
```

### Chi tiết các bước thực hiện trong mỗi iteration:
1. **Memory Context Preparation** (1 lần/episode, trước vòng lặp):
   - Truy xuất các **LTM facts** tương quan từ ChromaDB.
   - Nạp lịch sử **STM session** gần nhất.
2. **Query Rewriter Stage** (nếu bật, chạy đầu mỗi iteration):
   - Iteration 1: hình thành `retrieval_query` ban đầu từ câu hỏi gốc (evidence còn rỗng).
   - Iteration 2+: viết lại query dựa trên evidence thất bại + lý do reject của Verifier.
   - `question` gốc không bao giờ bị ghi đè, chỉ `retrieval_query` thay đổi.
3. **Retrieval Stage**: nếu bật, gọi mô hình nhúng (`_embed_query`) để chuyển `retrieval_query` (hoặc `question` nếu không có Query Rewriter) thành vector, Semantic Search trên ChromaDB lấy top-k bằng chứng y khoa.
4. **Reasoning Stage**: đưa prompt (câu hỏi gốc, lựa chọn A-B-C-D, evidence, lịch sử bộ nhớ...) vào Reasoning LLM, trích JSON `answer` / `explanation` / `confidence`.
5. **Verifier Stage** (nếu bật): Verifier LLM fact-check draft answer so với evidence.
   - `verdict = supported` → dừng vòng lặp, trả kết quả ngay.
   - `verdict = unsupported` → lưu evidence + lý do reject vào context cho iteration sau, ghi `loop_scratchpad` nếu STM scope `loop`/`both` đang bật, rồi lặp lại từ bước 2 cho đến khi `supported` hoặc chạm `max_iterations`.
   - Không có Verifier → dừng sau 1 iteration duy nhất (hành vi của V0/V1).

---

## 📊 7. Ý Nghĩa Của Tệp Output & Dữ Liệu Kết Quả

Khi chạy ở chế độ **Benchmark**, kết quả sẽ được ghi vào các tệp `.jsonl` trong thư mục `output/` (ví dụ: `output/predictions_v3-qr_1785430074.jsonl`).

### 📌 Cấu trúc Tệp JSONL Output

Mỗi tệp JSONL bao gồm **Phần Header Metadata** ở đầu tệp và **Các Dòng Dữ Liệu (JSON Lines)** đại diện cho kết quả xử lý từng câu hỏi.

#### 1. Header Metadata (Các dòng bình luận đầu tệp)
```text
# ============================================================
# MED-AI predictions — variant=v3-qr
# generated_at: 1788608766
# models: query_rewriter: deepseek/deepseek-v4-flash, retrieval: bge-m3, reasoning: deepseek/deepseek-v4-flash, verifier: deepseek/deepseek-v4-flash
# split: train  limit: 10
# pipeline: query_rewriter -> retrieval(top_k=5) -> reasoning -> verifier(max_iterations=3)
# memory: STM(scope=loop) + LTM(mode=read_only)
# ============================================================
```
- Giúp người dùng biết chính xác phiên bản cấu hình (`variant`), thời gian chạy, model LLM dùng cho từng agent trong pipeline (`models`), sơ đồ pipeline và trạng thái bộ nhớ. Khi không bật memory, dòng `memory` sẽ là `OFF`.

---

#### 2. Giải Thích Các Trường Dữ Liệu Trong Từng Dòng JSON

Tất cả các phiên bản (V0 đến V3) đều tuân theo một Schema chuẩn hoá thống nhất (`EpisodeResult` trong `core/types.py`):

| Tên Trường (Field) | Kiểu Dữ Liệu | Ý Nghĩa & Giá Trị | Các Version Hỗ Trợ |
| :--- | :--- | :--- | :--- |
| **`question_id`** | `str` | Định danh duy nhất của câu hỏi (ví dụ: `test_00001`). | Tất cả (V0-V3) |
| **`variant`** | `str` | Tên phiên bản thử nghiệm (`v0`, `v1`, `v2`, `v2-qr`, `v3`, `v3-qr`). | Tất cả (V0-V3) |
| **`model`** | `str` | Tên mô hình LLM suy luận chính (được cấu hình trong `.env`). | Tất cả (V0-V3) |
| **`predicted_answer`** | `str` | Đáp án do hệ thống dự đoán (ví dụ: `"A"`, `"B"`, `"C"`, `"D"` hoặc `"INVALID"`). | Tất cả (V0-V3) |
| **`explanation`** | `str` | Lời giải thích lập luận chi tiết cho đáp án đã chọn. | Tất cả (V0-V3) |
| **`confidence`** | `float` | Độ tin cậy của câu trả lời (từ `0.0` đến `1.0`). | Tất cả (V0-V3) |
| **`gold_answer`** | `str \| null` | Đáp án chuẩn xác của bộ dữ liệu (Ground Truth). | Tất cả (V0-V3) |
| **`is_correct`** | `bool \| null` | Kết quả so sánh (`true` nếu `predicted_answer == gold_answer`, `false` nếu sai). | Tất cả (V0-V3) |
| **`total_latency_ms`** | `float` | Tổng thời gian xử lý toàn bộ episode (đơn vị: mili-giây). | Tất cả (V0-V3) |
| **`total_token_usage`** | `dict` | Tổng số lượng token đã sử dụng (`{"input": int, "output": int}`). | Tất cả (V0-V3) |
| **`estimated_cost`** | `float \| null` | Chi phí ước tính (USD) dựa trên cấu hình giá `pricing` trong YAML. | Tất cả (V0-V3) |
| **`evidence_used`** | `list[str] \| null` | Danh sách ID các tài liệu minh chứng được RAG lấy ra từ ChromaDB. | V1, V2, V3 |
| **`retrieval_latency_ms`** | `float \| null` | Thời gian thực thi việc truy xuất RAG (mili-giây). | V1, V2, V3 |
| **`retrieval_token_usage`**| `dict \| null` | Số token tiêu tốn cho việc truy xuất (thường là 0 nếu dùng nhúng cục bộ). | V1, V2, V3 |
| **`iteration_count`** | `int \| null` | Số vòng lặp thực tế mà Controller đã chạy cho câu hỏi này. | V2, V3 (Verifier) |
| **`verifier_verdict`** | `str \| null` | Kết quả kiểm chứng cuối cùng từ Verifier (`"supported"` hoặc `"unsupported"`). | V2, V3 (Verifier) |
| **`verdict_history`** | `list[str] \| null` | Lịch sử verdict qua từng vòng lặp (ví dụ: `["unsupported", "supported"]`). | V2, V3 (Verifier) |
| **`stopped_after_max_iterations`** | `bool \| null` | `true` nếu bị dừng do chạm ngưỡng số vòng lặp tối đa mà chưa đạt verdict `supported`. | V2, V3 (Verifier) |
| **`reasoning_latency_ms`** | `float \| null` | Tổng thời gian các lần chạy của Reasoning Stage. | Tất cả |
| **`reasoning_token_usage`** | `dict \| null` | Tổng token sử dụng riêng cho Reasoning Stage. | Tất cả |
| **`verifier_latency_ms`** | `float \| null` | Tổng thời gian thực thi của Verifier Stage. | V2, V3 (Verifier) |
| **`verifier_token_usage`** | `dict \| null` | Tổng token sử dụng riêng cho Verifier Stage. | V2, V3 (Verifier) |
| **`query_rewrite_count`** | `int \| null` | Số lần Query Rewriter đã thực hiện viết lại câu hỏi. | V2-QR, V3-QR |
| **`rewriter_latency_ms`** | `float \| null` | Tổng thời gian thực thi của Query Rewriter Stage. | V2-QR, V3-QR |
| **`rewriter_token_usage`** | `dict \| null` | Tổng token sử dụng riêng cho Query Rewriter Stage. | V2-QR, V3-QR |
| **`stm_scope_used`** | `str \| null` | Dự trữ cho phạm vi STM (`"loop"`, `"session"`, `"both"`). ⚠️ `runner.py` hiện chưa gán giá trị này — luôn `null` trong output. | V3 (chưa populate) |
| **`loop_history_length`** | `int \| null` | Số lượng ghi chú lịch sử vòng lặp (`loop_scratchpad`) được giữ trong ngữ cảnh. | V2, V3 |
| **`ltm_facts_retrieved`** | `list[str] \| null` | Dự trữ cho danh sách fact LTM đã truy xuất. ⚠️ `runner.py` hiện chưa gán giá trị này — luôn `null` trong output. | V3 (chưa populate) |
| **`ltm_write_triggered`** | `bool \| null` | Dự trữ cho cờ báo có ghi fact mới vào LTM hay không. ⚠️ `runner.py` hiện chưa gán giá trị này — luôn `null` trong output. | V3 (chưa populate) |
| **`user_id`** | `str \| null` | Dự trữ cho định danh người dùng phục vụ phân tách LTM. ⚠️ `runner.py` hiện chưa gán giá trị này — luôn `null` trong output. | V3 (chưa populate) |

> Ghi chú: 4 trường trên nằm sẵn trong schema `EpisodeResult` để dùng cho tương lai, nhưng `Runner.run_episode` (`core/runner.py`) hiện không set giá trị cho chúng — quan sát trực tiếp trong các file `.jsonl` ở `output/` sẽ luôn thấy `null` dù STM/LTM đang bật. Thông tin LTM/STM thực tế của chat mode (fact truy xuất, có ghi mới hay không...) hiện chỉ được in ra qua `logger.memory_info()` ở console, chưa được lưu vào output JSONL.

---

## 💡 8. Ghi Chú Phát Triển (Development Notes)

- **Mở rộng Stage mới**: Bạn có thể tạo thêm các Stage tùy chỉnh bằng cách kế thừa `BaseStage` trong `stages/base.py` và đăng ký vào `_STAGE_CLASSES` trong `core/runner.py`.
- **Chế độ Trace Log Chi Tiết**: Khi đặt `debug: verbose` trong tệp YAML config, ứng dụng sẽ tạo thêm tệp `trace_<variant>_<timestamp>.jsonl` chứa toàn bộ prompt thô, raw LLM response và context ở từng bước để hỗ trợ công tác debug.
