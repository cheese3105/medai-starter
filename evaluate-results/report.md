# Báo cáo đánh giá hệ thống Multi-Agent LLM — MedQA-USMLE

Baseline dùng để so sánh: **v0**

## 1. Tổng hợp theo từng biến thể

| Variant | N | Accuracy | Invalid Rate | Avg Latency (ms) | Avg Tokens/Q | Total Cost (USD) | Pipeline |
|---|---|---|---|---|---|---|---|
| v0 | 222 | 92.34% | 0.45% | 13,061 | 753 | N/A | reasoning |
| v1 | 200 | 89.00% | 1.00% | 17,812 | 1,921 | N/A | retrieval(top_k=5) -> reasoning |
| v2-qr | 200 | 88.00% | 0.00% | 53,805 | 7,569 | N/A | retrieval(top_k=5) -> reasoning -> verifier(max_iterations=3) -> query_rewriter |
| v3-qr | 498 | 88.35% | 0.00% | 65,515 | 8,086 | N/A | retrieval(top_k=5) -> reasoning -> verifier(max_iterations=3) -> query_rewriter |

## 2. So sánh cặp với baseline (accuracy gain, McNemar, bootstrap CI)

### v0 (baseline) vs v1

- Số câu hỏi chung: **200**
- Accuracy: v0 = 92.50% (95% CI [88.50%, 96.00%]), v1 = 89.00% (95% CI [84.50%, 93.00%])
- **Accuracy gain**: -3.50 điểm % (95% CI [-8.00%, +1.00%]) — ❌ chưa có ý nghĩa (bootstrap, α=0.05)
- **McNemar test** (exact_binomial): b=14 (chỉ v0 đúng), c=7 (chỉ v1 đúng), p-value = 0.1892 (❌ chưa có ý nghĩa ở α=0.05)
- **Win/Loss/Tie**: v0 thắng 14 câu, v1 thắng 7 câu, hoà (cả 2 đúng) 171 câu, hoà (cả 2 sai) 8 câu

### v0 (baseline) vs v2-qr

- Số câu hỏi chung: **200**
- Accuracy: v0 = 92.50% (95% CI [88.50%, 96.00%]), v2-qr = 88.00% (95% CI [83.50%, 92.01%])
- **Accuracy gain**: -4.50 điểm % (95% CI [-9.00%, +0.00%]) — ❌ chưa có ý nghĩa (bootstrap, α=0.05)
- **McNemar test** (exact_binomial): b=16 (chỉ v0 đúng), c=7 (chỉ v2-qr đúng), p-value = 0.0931 (❌ chưa có ý nghĩa ở α=0.05)
- **Win/Loss/Tie**: v0 thắng 16 câu, v2-qr thắng 7 câu, hoà (cả 2 đúng) 169 câu, hoà (cả 2 sai) 8 câu

### v0 (baseline) vs v3-qr

- Số câu hỏi chung: **221**
- Accuracy: v0 = 92.31% (95% CI [88.69%, 95.48%]), v3-qr = 90.95% (95% CI [86.88%, 94.57%])
- **Accuracy gain**: -1.36 điểm % (95% CI [-5.88%, +2.71%]) — ❌ chưa có ý nghĩa (bootstrap, α=0.05)
- **McNemar test** (exact_binomial): b=13 (chỉ v0 đúng), c=10 (chỉ v3-qr đúng), p-value = 0.6776 (❌ chưa có ý nghĩa ở α=0.05)
- **Win/Loss/Tie**: v0 thắng 13 câu, v3-qr thắng 10 câu, hoà (cả 2 đúng) 191 câu, hoà (cả 2 sai) 7 câu

## 3. Phân tích lỗi (Error Analysis)

### v0 → v1

- Số câu được **sửa đúng** (baseline sai, variant đúng): **7**
- Số câu bị **hồi quy** (baseline đúng, variant sai): **14**

**Ví dụ câu bị hồi quy (tối đa hiển thị một phần):**

| question_id | predicted | gold |
|---|---|---|
| test_00000 | A | B |
| test_00015 | D | B |
| test_00021 | INVALID | A |
| test_00034 | D | A |
| test_00059 | C | A |
| test_00088 | B | C |
| test_00093 | C | D |
| test_00114 | B | C |
| test_00122 | B | D |
| test_00125 | B | A |

### v0 → v2-qr

- Số câu được **sửa đúng** (baseline sai, variant đúng): **7**
- Số câu bị **hồi quy** (baseline đúng, variant sai): **16**

**Ví dụ câu bị hồi quy (tối đa hiển thị một phần):**

| question_id | predicted | gold |
|---|---|---|
| test_00000 | A | B |
| test_00015 | A | B |
| test_00018 | A | B |
| test_00034 | D | A |
| test_00038 | C | D |
| test_00059 | C | A |
| test_00088 | B | C |
| test_00093 | C | D |
| test_00106 | A | B |
| test_00120 | A | D |

### v0 → v3-qr

- Số câu được **sửa đúng** (baseline sai, variant đúng): **10**
- Số câu bị **hồi quy** (baseline đúng, variant sai): **13**

**Ví dụ câu bị hồi quy (tối đa hiển thị một phần):**

| question_id | predicted | gold |
|---|---|---|
| test_00000 | A | B |
| test_00015 | D | B |
| test_00020 | B | C |
| test_00059 | C | A |
| test_00092 | B | C |
| test_00093 | C | D |
| test_00099 | D | B |
| test_00114 | B | C |
| test_00117 | A | C |
| test_00128 | B | A |

---
*Báo cáo được sinh tự động bởi `evaluate.py`.*