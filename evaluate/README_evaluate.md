# evaluate.py — Script đánh giá hệ thống Multi-Agent LLM (MedQA-USMLE)

## Cài đặt
```bash
# Đã bao gồm sẵn trong requirements.txt:
pip install -r requirements.txt
```

## Cách chạy

Các file dự đoán `predictions_*.jsonl` (V0, V1, V2, V2-qr, V3-qr...) mặc định được lưu trong thư mục `output/`.

Để đánh giá toàn bộ các file trong thư mục `output/` và xuất kết quả vào `evaluate-results/`:

```bash
python evaluate.py --dir output --out-dir evaluate-results --baseline v0
```

Hoặc chỉ định file cụ thể:

```bash
python evaluate.py --files output/predictions_v0_*.jsonl output/predictions_v3-qr_*.jsonl --out-dir evaluate-results --baseline v0
```

Nếu muốn accuracy tính trên mẫu số cố định của test set chính thức (1273 câu MedQA-USMLE)
thay vì số câu thực sự chạy được (phòng khi 1 biến thể bị lỗi API rớt vài câu):

```bash
python evaluate.py --dir output --out-dir evaluate-results --official-n 1273
```

Thêm `--limit-first-n 200` để chỉ so sánh trên 200 câu đầu tiên (theo `question_id`):

```bash
python evaluate.py --dir output --out-dir evaluate-results --limit-first-n 200
```

## Output

Trong thư mục `--out-dir` (ví dụ `evaluate-results/` hoặc mặc định là thư mục hiện tại):

- `summary_metrics.csv` — accuracy, invalid rate, latency, token, cost theo từng biến thể
- `pairwise_comparison.csv` — win/loss/tie, McNemar p-value, bootstrap CI của accuracy gain
  giữa baseline và từng biến thể còn lại
- `report.md` — báo cáo Markdown đầy đủ, có thể dán thẳng vào phần "Kết quả" / "Phân tích lỗi"
  của report cuối kỳ
- `full_results.json` — toàn bộ số liệu raw (để phân tích thêm / vẽ biểu đồ)

## Các chỉ số được tính (đúng theo mục 5 của đề bài)

| Chỉ số | Ý nghĩa |
|---|---|
| Accuracy | số câu đúng / tổng số câu (hoặc / `--official-n` nếu chỉ định) |
| Invalid response rate | tỷ lệ `predicted_answer` không nằm trong {A,B,C,D,E} |
| Accuracy gain | Accuracy(variant) − Accuracy(baseline) |
| Win/Loss/Tie | so từng câu giữa baseline và biến thể |
| McNemar's test | kiểm định exact binomial (nếu số câu bất đồng < 25) hoặc chi-square hiệu chỉnh liên tục (nếu ≥ 25) |
| Bootstrap 95% CI | paired bootstrap (10.000 lần resample mặc định) cho accuracy và accuracy gain — CI không chứa 0 nghĩa là gain có ý nghĩa thống kê |
| Cost & Latency | tổng/trung bình token, latency (avg/median/p95), cost ước tính (nếu điền bảng giá) |

## Lưu ý quan trọng

1. **Ước tính cost**: script để trống bảng giá mặc định. Nếu muốn tính `estimated_cost`,
   sửa dict `MODEL_PRICING_PER_1M_TOKENS` ở đầu file `evaluate.py`, ví dụ:
   ```python
   MODEL_PRICING_PER_1M_TOKENS = {
       "deepseek/deepseek-v4-flash": {"input": 0.27, "output": 1.10},
   }
   ```
   Nếu file prediction đã có sẵn field `estimated_cost` thì script tự dùng luôn, không cần bảng giá.

2. **Lệch số câu giữa các biến thể**:
   script tự động in cảnh báo ra `stderr` và **chỉ so sánh trên phần giao nhau** (câu hỏi có ở
   cả 2 biến thể). Để kết quả ablation study đáng tin cậy, tất cả biến thể nên chạy trên
   **đúng cùng một tập câu hỏi** (dev set 100-150 câu khi debug, rồi 1273 câu chính thức khi
   báo cáo kết quả cuối — chỉ chạy 1 lần theo đúng quy trình chống overfitting trong đề bài).

3. **`is_correct` vs tự so sánh**: script ưu tiên dùng field `is_correct` có sẵn trong file
   jsonl (do hệ thống tự chấm lúc sinh). Nếu thiếu field này, script tự so
   `predicted_answer == gold_answer`.

4. **Trùng `question_id` trong cùng 1 file**: script cảnh báo và giữ bản ghi xuất hiện sau
   cùng (đè lên bản ghi trước).

5. Script không phụ thuộc framework nào (không cần LangGraph/LlamaIndex...) — chỉ đọc thuần
   file `.jsonl` output, nên dùng được với bất kỳ pipeline nào miễn đúng schema các trường:
   `question_id`, `variant`, `predicted_answer`, `gold_answer`, `is_correct`, `total_latency_ms`,
   `total_token_usage: {input, output}`.
