# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV: 2A202603629**
- **Họ và tên: Trần Nguyễn Anh Phong**
- **SEED** (in ở ô đầu notebook): 8806

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | x |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | x |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | x |

## 2. Kết quả chính

*Chép bảng so sánh cuối Phần 5 (F1 test, F1 SUB 40, p95 ms/tin, USD/1k tin) của chính bạn.*
## Bảng tổng hợp kết quả thực nghiệm

### 1. Chất lượng mô hình

| Model | Precision | Recall | F1 (Test) | F1 (SUB 40) |
|---|---:|---:|---:|---:|
| 1. BoW + LogReg | 1.000 | 0.794 | 0.885 | 0.824 |
| 2. BoW + MLP | 0.957 | 0.907 | 0.931 | 0.889 |
| 3a. MiniLM + kNN | 0.892 | 0.938 | 0.915 | **0.974** |
| 3b. MiniLM + LogReg | 0.944 | 0.876 | 0.909 | 0.919 |
| 3c. MiniLM + MLP Head | 0.967 | 0.907 | **0.936** | 0.919 |
| 4a. LLM nhỏ zero-shot | 0.833 | 1.000 | 0.909 | 0.909 |
| 4b. LLM lớn zero-shot | 0.952 | 1.000 | **0.976** | **0.976** |
| 4c. LLM nhỏ few-shot | 0.864 | 0.950 | 0.905 | 0.905 |

### 2. Tốc độ huấn luyện và suy luận

| Model | Train (s) | Tin/s (cả lô) | p50 (ms/tin) |
|---|---:|---:|---:|
| BoW + LogReg | 5.512 | 955.456 | 1.113 |
| BoW + MLP | 14.402 | 1012.825 | 1.782 |
| MiniLM + kNN | 91.412 | 38.149 | 40.953 |
| MiniLM + LogReg | 91.881 | 42.571 | 32.605 |
| MiniLM + MLP Head | 92.368 | 46.281 | 25.684 |
| LLM nhỏ zero-shot | 0.000 | 0.884 | 771.436 |
| LLM lớn zero-shot | 0.000 | 1.008 | 529.386 |
| LLM nhỏ few-shot | 0.000 | 0.526 | 1856.733 |

### 3. Độ trễ p95 và chi phí

| Model | p95 (ms/tin) | USD / 1k tin |
|---|---:|---:|
| BoW + LogReg | 3.453 | 0.0 |
| BoW + MLP | 6.172 | 0.0 |
| MiniLM + kNN | 67.639 | 0.0 |
| MiniLM + LogReg | 57.895 | 0.0 |
| MiniLM + MLP Head | 43.249 | 0.0 |
| LLM nhỏ zero-shot | 2575.087 | 0.0 |
| LLM lớn zero-shot | 2619.396 | 0.0 |
| LLM nhỏ few-shot | 3663.606 | 0.0 |

### 4. Ghi chú khi đánh giá

- Hàng LLM chỉ đo trên SUB (40 tin), vì vậy cần so sánh chất lượng công bằng bằng cột `F1 (SUB 40)`.
- SUB chỉ có 40 tin: 1 tin dự đoán sai có thể làm F1 thay đổi vài điểm. Chênh lệch dưới khoảng 0.05 F1 nên xem là gần hòa, không nên xếp hạng theo số lẻ.
- Inference đo từ văn bản đến nhãn (gồm cả vectorize / encode). p50/p95 là thời gian gọi từng tin một, giống như một request thực tế.
- Train (s) của 3a/3b/3c là thời gian encoder MiniLM encode tập train (cộng thời gian train LogReg / MLP); LLM = 0 vì không train.

### 5. Nhận xét

- LLM lớn zero-shot đạt F1 cao nhất trên SUB 40 (0.976), nhưng độ trễ p95 rất lớn (2619.396 ms/tin).
- MiniLM + MLP Head có p95 = 43.249 ms/tin, đáp ứng SLA p95 dưới 50 ms.
- BoW + LogReg có tốc độ nhanh nhất, p95 chỉ 3.453 ms/tin, nhưng F1 SUB 40 thấp hơn các phương pháp Embedding và LLM.
- MiniLM + kNN có F1 SUB 40 = 0.974 nhưng p95 = 67.639 ms/tin, không đạt SLA 50 ms.

**Kết luận:** Nếu yêu cầu p95 dưới 50 ms/tin và chất lượng tốt, MiniLM + MLP Head là lựa chọn đáng cân nhắc trong các mô hình đã thử nghiệm.

## 3. Dự đoán và thực tế

*Chọn 3 dự đoán `predict(...)` lệch nhiều nhất so với kết quả. Mỗi cái: bạn đoán gì, ra gì, vì sao lệch.*
predict hard_eval: đánh giá quá cao f1 trên tập test, đoán lệch khá nhiều %.
predict tradeoff: đoán mô hình 3c. Encoder (MiniLM) + MLP head, còn thiếu 5 mô hình khác.
predict emb_tricky: dự đoán BoW, kết quả là 2 mô hình như nhau._

## 4. Phân tích 50 case khó

*Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai.*
- Tụt nhiều nhất là BoW + LogReg, giảm 53.8 điểm phần trăm.
## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*
🎯 Đề #5: Embedding · Viết 4 tin nhắn của riêng bạn (2 spam né từ khoá, 2 ham chứa 'free'/'win'). So dự đoán BoW vs Embedding+LogReg và giải thích từng tin.
- Cách làm: dựa trên phần 50 case khó, viết 4 tin nhắn mới, chạy dự đoán với BoW + LogReg và MiniLM + LogReg.
- Kết quả:
+ Trong thử nghiệm nhỏ gồm 4 tin nhắn, Embedding + Logistic Regression đạt Accuracy 100%, cao hơn BoW với 50%.
+ Kết quả cho thấy Embedding có lợi thế trong việc biểu diễn ngữ nghĩa và xử lý những tin nhắn có cách diễn đạt khác với dữ liệu huấn luyện. Trong khi đó, BoW dễ gặp khó khăn khi từ ngữ bị biến đổi hoặc không nằm trong từ điển.
+ Thử nghiệm cũng cho thấy cần xem xét ngữ cảnh của toàn bộ tin nhắn thay vì chỉ dựa vào một vài từ khóa để phân loại Spam/Ham.
Lưu ý: Đây là thử nghiệm cá nhân với chỉ 4 mẫu, nên kết quả mang tính minh họa, chưa đủ để khẳng định Embedding luôn tốt hơn BoW trong mọi trường hợp.
## 6. Phương án cho sản phẩm thật

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*
+ Nhà mạng cần xử lý 10 triệu SMS/ngày, tương đương khoảng 116 tin/giây và yêu cầu p95 dưới 50 ms/tin.

+ Đạt SLA độ trễ: BoW + LogReg (4.268 ms), BoW + MLP (9.563 ms), MiniLM + LogReg (10.584 ms), MiniLM + MLP (12.707 ms), MiniLM + kNN (13.714 ms).

+ Đạt cả độ trễ và chất lượng: MiniLM + kNN là lựa chọn phù hợp nhất, với F1 = 0.974 trên SUB 40 và p95 = 13.714 ms.

+ LLM: Không đạt SLA vì p95 lên đến hàng giây, đồng thời tốc độ xử lý khoảng 1 tin/giây, thấp hơn nhiều so với yêu cầu 116 tin/giây.

+ Về chi phí, bảng thực nghiệm hiển thị USD/1k tin = 0.0 cho LLM. Nếu lấy đúng giá trị hiển thị thì 300 triệu tin/tháng có chi phí 0 USD. Tuy nhiên, đây có thể là giá trị làm tròn hoặc môi trường thử nghiệm miễn phí, không đủ cơ sở để ước tính chi phí triển khai thực tế.

+ Kết luận: Chọn MiniLM + kNN làm mô hình chính vì cân bằng tốt giữa chất lượng và độ trễ.
## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*
| # | Tôi hỏi | AI trả lời | Tôi sửa / kiểm tra |
|---|---|---|---|
| 1 | Nếu dùng Groq thay OpenAI thì sao | Hai cách: SDK `groq` hoặc OpenAI SDK với `base_url`; các khác biệt về tham số token và `reasoning_effort` | Đối chiếu ô cấu hình `PROVIDERS`: template gọi Groq qua OpenAI SDK (`base_url`), nên không cần SDK `groq` |
| 2 | Viết `build_classification_prompt` và `parse_label` (4.2, 4.3) | Prompt zero/few-shot cùng định dạng `Message: """..."""\nLabel:`; parser kiểm tra phủ định trước, dùng `\b`, bỏ tiền tố `Label:` | [Điền: các ví dụ đã chạy thử, ví dụ "not spam", "Label: spam", rỗng] |
| 3 | Viết `llm_classify_batch` (4.4) | Vòng lặp gọi `call_llm` qua `retry_with_backoff`, đếm `n_invalid`, ước tính token và chi phí | [Điền: đã kiểm tra `retry_with_backoff` trả đúng `(text, latency)` chưa] |
| 4 | Ô test giả lập báo lỗi 404 | Giải thích mock không chặn được vì `OpenAI` đã được import vào namespace của notebook, nên lời gọi đi ra Groq thật (cấu hình `OPENAI_BASE_URL`). Sửa bằng `openai.OpenAI()` trong hàm | AI từng gợi ý patch `groq.Groq`, sau khi xem ô cấu hình thì tự đính chính là không cần. Tôi chỉ áp dụng cách sửa `openai.OpenAI()` |
| 5 | Câu hỏi lý thuyết về `temperature=0`, `max_tokens`, `reasoning_effort` | Token suy luận tính vào ngân sách `max_tokens`, đặt quá nhỏ làm câu trả lời rỗng (`finish_reason="length"`) | Đối chiếu với `PROVIDERS` trong template (3 cho OpenAI, 512 cho Groq/Gemini, 1024 cho Claude) |
| 6 | Bắt lỗi 3 (`parse_label_v2`) | Liệt kê lỗi: phủ định bị hiểu ngược, khớp chuỗi con, không bao giờ trả `None` | [Điền: đã tự thử "not spam" để kiểm chứng chưa] |
| 7 | WER theo latency là gì | Giải thích WER và trade-off với latency (không thuộc bài lab này) | Chỉ để hiểu khái niệm, không dùng trong báo cáo |
### Điều tôi hiểu ra hôm nay

Trước đây tôi nghĩ dùng LLM chỉ cần viết prompt. Hôm nay tôi hiểu phần kỹ thuật xung quanh quan trọng không kém: với model suy luận, token "suy nghĩ" ăn vào `max_tokens`, nên đặt giới hạn quá nhỏ sẽ làm câu trả lời trống. Việc kiểm tra output (parser trả `None`, đếm `n_invalid`) cũng quan trọng, vì không thể tin model lúc nào cũng nghe lời.
Tôi cũng hiểu LLM mạnh nhất ở bài toán mới chưa có dữ liệu gán nhãn, nhưng khi yêu cầu là p95 < 50 ms và hàng triệu tin mỗi ngày thì mô hình cổ điển mới là lựa chọn thực tế, còn LLM phù hợp làm tầng phụ cho các case khó. Cuối cùng, một lỗi trong test (mock không ăn) cho tôi thấy cách Python tra cứu tên lúc import và lúc chạy ảnh hưởng đến việc kiểm thử.
