# MED-AI - Nghiên Cứu Đánh Giá Hiệu Năng & An Ninh Prompt Injection Trong AI Y Khoa

MED-AI là framework thử nghiệm và đánh giá các hệ thống trí tuệ nhân tạo y tế xây dựng trên kiến trúc Multi-Agent. Dự án tập trung giải quyết hai bài toán: đo lường khả năng suy luận chẩn đoán lâm sàng qua từng thế hệ kiến trúc (từ V0 đến V3) trên tập dữ liệu MedQA-USMLE; và đánh giá mức độ an toàn, tính dễ tổn thương và khả năng tự phòng vệ của hệ thống trước nguy cơ tấn công Indirect Prompt Injection từ các nguồn dữ liệu bên ngoài.

---

## 1. Tổng quan nghiên cứu

Trong các ứng dụng y tế hỗ trợ ra quyết định (CDSS), mô hình ngôn ngữ lớn thường được kết hợp với tài liệu y văn bên ngoài nhằm bổ sung ngữ cảnh và hạn chế hiện tượng ảo giác. Tuy nhiên, việc tích hợp nguồn dữ liệu chưa qua kiểm duyệt mở ra bề mặt tấn công Indirect Prompt Injection, diễn ra khi kẻ tấn công chủ động đưa chỉ dẫn độc hại vào nguồn dữ liệu bên ngoài (external_source).

Ở kịch bản tấn công gián tiếp, kẻ tấn công không can thiệp vào câu hỏi của bác sĩ hay cấu hình hệ thống, mà tác động trực tiếp vào nội dung tài liệu được đưa vào giai đoạn suy luận (Reasoning Stage). Khi mô hình tiếp nhận ngữ cảnh bị đầu độc, chỉ dẫn độc hại sẽ tìm cách chiếm quyền điều khiển, khiến hệ thống bỏ qua nhiệm vụ chẩn đoán ban đầu để chuyển sang thực thi một tác vụ hoàn toàn khác do kẻ tấn công định sẵn.

Dự án tiến hành khảo sát thực nghiệm các kỹ thuật inject phổ biến nhằm đánh giá mức độ ảnh hưởng đến kết quả chẩn đoán y khoa, đồng thời kiểm chứng xem cơ chế phản biện như Verifier và Query Rewriter có thể đóng vai trò như một lớp phòng thủ giúp ngăn chặn tấn công hay không.

---

## 2. Các mô hình được triển khai

Hệ thống cung cấp 4 thế hệ kiến trúc tiến hóa để đối sánh hiệu quả suy luận lâm sàng cùng mức độ an toàn trước nguy cơ tấn công:

- V0 (Direct LLM Reasoning): Mô hình suy luận trực tiếp từ câu hỏi bệnh án theo dạng zero-shot hoặc few-shot, đóng vai trò mốc cơ sở (baseline) khi chưa có sự hỗ trợ của tài liệu ngoài hay bộ nhớ.
- V1-QR (RAG Evidence Retrieval): Kết hợp cơ sở dữ liệu vector ChromaDB để trích xuất các đoạn y văn liên quan làm bằng chứng bổ trợ trước khi suy luận, tích hợp thêm Query Rewriter để chuẩn hóa câu hỏi thành truy vấn ngữ nghĩa tối ưu.
- V2-QR (Self-Correction & Multi-Agent Verification): Thiết lập chu trình tự phản biện khép kín. Verifier Agent đối chiếu câu trả lời nháp với bằng chứng thu thập được. Nếu nhận định chưa đủ căn cứ, Query Rewriter Agent sẽ định hình lại câu truy vấn để tìm kiếm thêm thông tin cho đến khi đạt yêu cầu hoặc chạm ngưỡng số vòng lặp tối đa.
- V3-QR (Memory-Augmented System): Hoàn thiện hệ thống với cơ chế bộ nhớ kép, bao gồm bộ nhớ ngắn hạn (STM) duy trì mạch hội thoại trong phiên làm việc và bộ nhớ dài hạn (LTM) lưu trữ thông tin bệnh sử người dùng trong ChromaDB.

Toàn bộ quy trình được điều phối thống nhất qua `core/runner.py` và có thể tùy biến linh hoạt thông qua các tệp cấu hình YAML trong thư mục `configs/`.

---

## 3. Quick Start

### Cài đặt môi trường

Hệ thống yêu cầu Python phiên bản 3.10 trở lên. Khởi tạo môi trường ảo và cài đặt các thư viện cần thiết:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Cập nhật thông tin API key và endpoint mô hình trong tệp `.env`. Đối với các cấu hình sử dụng RAG (từ V1 trở lên), tải dữ liệu ChromaDB đã dựng sẵn từ [Google Drive](https://drive.google.com/file/d/1pIQYQ7CHJPbWqNW07kff5XA3YS8ER_Bb/view?usp=sharing) và giải nén vào thư mục `data/chroma/`.

### Chạy ứng dụng

Chạy đánh giá benchmark trên MedQA:

```bash
python main.py --mode benchmark --config configs/v0.yaml --limit 10
```

Khởi động giao diện trò chuyện lâm sàng:

```bash
python main.py --mode chat --config configs/v3-chat.yaml
```

Chạy kiểm thử tấn công Prompt Injection:

```bash
.venv/bin/python -m attack.cli --target-config attack/configs/v0-attack.yaml --sample-size 2 --output-dir attack/output/smoke-test
```

---

## 4. Danh sách các document khác

Các tài liệu đi sâu vào kiến trúc hệ thống, hướng dẫn sử dụng và báo cáo nghiên cứu an ninh được tổ chức trong thư mục `docs/`:

- [docs/project-structure.md](docs/project-structure.md): Chi tiết cấu trúc thư mục, nhiệm vụ của từng mô-đun trong `core/`, các stage xử lý và cơ chế quản lý bộ nhớ.
- [docs/quickstart.md](docs/quickstart.md): Hướng dẫn cài đặt chi tiết, giải thích các tùy chọn dòng lệnh và quy trình phân tích thống kê kết quả.
- [docs/attack/](docs/attack/): Tài liệu tổng quan và chi tiết kỹ thuật về mô hình đe dọa, các phương pháp tấn công prompt injection, thuật toán lấy mẫu và hệ thống chỉ số an ninh.
- [docs/defense/](docs/attack/): Tài liệu tổng quan và chi tiết kỹ thuật về mô hình phòng thủ, phương pháp phòng thủ được thử nghiệm và một số các nhận xét liên quan.
