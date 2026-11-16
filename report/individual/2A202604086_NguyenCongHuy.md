# Báo cáo Day 7

**Họ và tên:** Nguyễn Công Huy  
**MSSV:** 2A202604086  
**Lớp:** L3B

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:** Giáo trình *Introduction to Python Programming* của OpenStax ([PDF nguồn](https://assets.openstax.org/oscms-prodcms/media/documents/Introduction_to_Python_Programming-WEB.pdf)), giấy phép CC BY-NC-SA 4.0. PDF có 397 trang vật lý, trong đó 395 trang trích xuất được text. Chunk dùng số trang vật lý, bắt đầu từ 1. Đã tạo [day6_chunks.json](../../starter/day6_chunks.json) từ pipeline Day 6: A_fixed có 807 chunk; B_paragraph có 771; C_paragraph_filter có 771; tổng cộng 2.349 chunk.
- **Vector store, embedding model, phiên bản và số chiều:** Day 6 dùng PostgreSQL + pgvector, cosine distance, bảng `document_chunks`; embedding local `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, 384 chiều. Revision model và phiên bản thư viện chưa được ghi lại.
- **Agent model, cách chạy và phân chia dev/test:** Gemini model lấy từ `GEMINI_MODEL` trong `starter/.env`; API key không ghi vào báo cáo. Một lần gọi Controller trực tiếp với context mẫu đã thành công ở lượt kiểm thử trước, nhưng đó không phải đánh giá agent.
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:** [day6_questions.json](../../starter/day6_questions.json) có 20 câu có gold evidence (15 dev, 5 test). [cases.json](../../starter/data/agent-test/cases.json) có thêm Q21–Q24 là câu thiếu evidence, mỗi split có 2 câu; lý do vắng mặt được kiểm tra trên toàn bộ corpus. Tổng cộng 24 case: 17 dev (15 có đáp án, 2 thiếu evidence) và 7 test (5 có đáp án, 2 thiếu evidence). Gold page chưa khớp nhất quán với số trang vật lý PDF; ví dụ Q01 ghi trang 30 nhưng evidence thuộc trang vật lý 40.

## Retrieval và thử nghiệm

**Kiểm tra kết nối Day 6 thật:** `check_setup.py` kết nối thành công tới PostgreSQL 16.16, pgvector 0.8.7. Đọc chỉ số xác nhận bảng `document_chunks` có 807 `A_fixed`, 771 `B_paragraph`, 771 `C_paragraph_filter`; cột embedding là `vector(384)`. Đã chạy embedding và `Retriever.search` thật trên cả ba strategy, mỗi strategy trả về kết quả. Bộ lọc `chapter=1, section=1.2` trên `C_paragraph_filter` cũng trả về 3 chunk đúng section. Đây chỉ là smoke test tích hợp, không phải benchmark: thử Q01 tìm đủ từ khóa evidence trong top-5 ở `A_fixed`, nhưng không ở hai strategy còn lại. Đã cài `sentence-transformers` và `pgvector` vào `.venv` của repo Day 6 để chạy thử; không thay đổi dữ liệu trong database.

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Median / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline | TF-IDF, top-3, không filter/rerank, budget 800 ký tự | Baseline lexical | dev | 3/15 (20%) | 7/15 (46.7%) | 0/15 (0%) | 1.41 ms / $0 |
| Dense A_fixed | Day 6 dense, top-3, không filter/rerank, budget 3200 ký tự | So sánh embedding retrieval | dev | 3/15 (20%) | 6/15 (40%) | 6/15 (40%) | 87.54 ms / $0 |
| Hybrid A_fixed | RRF TF-IDF + dense A_fixed, candidate-10, top-3, budget 3200 | Kết hợp thứ hạng lexical và dense | dev | 3/15 (20%) | 6/15 (40%) | 6/15 (40%) | 90.83 ms / $0 |
| Dense B_paragraph | Day 6 dense B_paragraph, top-3, không filter/rerank, budget 3200 | Kiểm tra chunk theo đoạn | dev | 3/15 (20%) | 4/15 (26.7%) | 4/15 (26.7%) | 75.70 ms / $0 |
| Dense C_paragraph_filter | Day 6 dense C_paragraph_filter, top-3, filter chapter/section chỉ khi câu hỏi nêu rõ, budget 3200 | Kiểm tra strategy và metadata filter | dev | 3/15 (20%) | 5/15 (33.3%) | 5/15 (33.3%) | 74.76 ms / $0 |
| Hybrid B_paragraph | RRF TF-IDF + dense B_paragraph, candidate-10, top-3, budget 3200 | Hybrid với chunk theo đoạn | dev | 2/15 (13.3%) | 6/15 (40%) | 5/15 (33.3%) | 79.98 ms / $0 |
| Cấu hình chọn | TF-IDF, top-3, không filter/rerank, budget 3200 ký tự | Tốt nhất trong các cấu hình dev đã thử; budget lớn đưa đủ chunk tìm thấy vào context | dev | 3/15 (20%) | 7/15 (46.7%) | 7/15 (46.7%) | 2.19 ms / $0 |
| Cấu hình chọn | TF-IDF, top-3, không filter/rerank, budget 3200 ký tự | Test sau khi chốt cấu hình | test | 0/5 (0%) | 1/5 (20%) | 1/5 (20%) | 1.49 ms / $0 |
| Baseline | TF-IDF, top-3, không filter/rerank, budget 800 ký tự | Đối chứng budget trên test | test | 0/5 (0%) | 1/5 (20%) | 0/5 (0%) | 1.84 ms / $0 |

Benchmark retrieval dùng 15 câu có đáp án ở dev và 5 ở test, cùng corpus A_fixed, evidence labels theo trang vật lý và context packing tuần tự theo thứ hạng; không tính 4 câu thiếu evidence vào mẫu số. Thời gian là median retrieval mỗi query trên máy cục bộ, chưa gồm khởi tạo index/model; chi phí dịch vụ retrieval là $0. Tăng top-k TF-IDF từ 3 lên 5 trên dev không đổi Hit@1, Evidence Hit@3 hay Context Evidence (3/15, 7/15, 7/15), nhưng tăng context trung bình từ khoảng 2597 lên 2760 ký tự, nên giữ top-3. Các trace/summary retrieval nằm trong `starter/data/retrieval-benchmark/` (thư mục local bị Git ignore).

Hai lỗi dev và phân tích:
- **Q03:** TF-IDF xếp các trang 20, 22, 23 vào top-3, bỏ chunk trang 21 có giải thích `end` và xuống dòng; dense A_fixed tìm được evidence trang 21. Thử dense và hybrid nhưng trên toàn bộ dev chúng vẫn đạt Evidence Hit@3 6/15, thấp hơn TF-IDF 7/15. Giả thuyết: lexical overlap của các từ “input/output” kéo kết quả sang ví dụ gần chủ đề; thay retrieval đơn lẻ chưa cải thiện tổng thể.
- **Q15:** cần hai fact ở trang 238 (key không trùng) và 255 (key immutable). Các ranking top-3 được thử không bao phủ đủ cả hai trang; tăng context budget không thể bù cho evidence không được truy hồi, và top-5 TF-IDF vẫn không đổi Evidence Hit@3. Đây là câu multi-evidence mà metric yêu cầu đủ mọi fact, không chỉ một đoạn có liên quan.
- Chốt TF-IDF top-3/budget 3200 theo dev vì đạt kết quả context tốt nhất (7/15), chạy nhanh hơn dense/hybrid và không phát sinh phí dịch vụ. Trên test, baseline và cấu hình chọn cùng tìm được evidence cho 1/5; context budget lớn đưa được câu đó vào context nhưng kết quả tổng thể vẫn thấp, nên chưa khẳng định cải thiện retrieval tổng quát.

## Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:** Trong [agentic.py](../../starter/agentic.py), `run_agent` có mặc định `max_searches=4`, `top_k=3`, context budget 800 ký tự; kết thúc khi agent trả lời, chạm giới hạn tìm kiếm hoặc citation không hợp lệ. Lỗi provider/vector store được truyền lên. `Controller` nhận callback model; `Retriever` nhận embedder và vector store Day 6. Chưa có lần chạy agent thật.

Đã thử chạy batch agent thật cho 17 case dev-baseline bằng Gemini nhưng batch không hoàn tất sau khoảng 40 phút; đã dừng theo yêu cầu. Lượt này chưa ghi trace/summary. Không chạy các cấu hình agent còn lại và không tạo nhãn hoặc metrics giả. Runner hiện lưu từng case hoàn tất, đặt timeout request 120 giây và không ghi API key để lần chạy sau không mất toàn bộ tiến độ.

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| Agent thật / dev và test | Chưa có trace hoàn tất | Chưa đo được | Chưa chấm | Chưa chấm | Chưa đo được | Chưa chấm | Chưa chấm |

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| Chưa có | — | Agent run bị dừng khi request còn chờ; không có trace | — | Không thể audit |

- **Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:** Cases: [cases.json](../../starter/data/agent-test/cases.json). Chưa có trace agent; batch đầu tiên chạy hơn 40 phút rồi bị dừng, chi phí API chưa xác định.  
  Kiểm thử mã (không phải benchmark agent): `$env:PYTHONUTF8='1'; python -m pytest -q` — 24 passed.
- **Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn:**

## Thử thách nâng cao (nếu thực hiện)

- Tài liệu, nguồn, quyền sử dụng, số trang và phiên bản:
- Cách xử lý text, bảng, ảnh, biểu đồ; phạm vi index và phần chưa xử lý:
- Cách chia benchmark, schema evidence, model và cấu hình đã chốt trên dev:

| Nhóm test | Đạt / tổng | Tỷ lệ | Lỗi / nhận xét |
|---|---|---|---|
| Bảng | | | |
| Ảnh, sơ đồ | | | |
| Biểu đồ | | | |
| Multi-hop | | | |
| Text | | | |
| Thiếu bằng chứng | | | |
| **Toàn bộ** | | | |

- Trace, nhãn và evidence từng câu; thời gian, chi phí, lỗi hệ thống:
- Giới hạn, cách tái lập và đường dẫn benchmark:
