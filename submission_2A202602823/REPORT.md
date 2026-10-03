# Báo cáo Lab Day 1 — Nguyễn Mạnh Cường — 2A202602823

## 1. Thiết lập

Các run thí nghiệm chính dùng Python 3.11, PyTorch 2.14 CPU trên Intel Core i5-1340P (16 logical CPU); notebook kiểm tra cuối được chạy lại bằng kernel PyTorch 2.11 CPU. Dữ liệu CoverType được chia cố định theo `split_metadata.csv`: 464.809 mẫu train và 116.203 mẫu eval. Tôi tách 20% từ train bằng phân tầng, seed 42, thu được 371.847 train và 92.962 validation. Chỉ 10 đặc trưng liên tục được chuẩn hoá bằng mean/std của phần train; eval không tham gia chuẩn hoá hay chọn cấu hình.

Model bắt buộc là M-base `54→256→128→7`, ReLU, 47.879 tham số. Baseline dùng CE, He initialization, SGD momentum 0,9, lr 0,1, batch 512, 20 epoch, FP32, không dropout/clip. Mốc lớp đa số trên validation là accuracy 0,4876 và macro-F1 khoảng 0,094. Tôi thử đủ bảy chủ đề: loss, optimizer, batch size, dropout, clipping, mixed precision và initialization.

## 2. Kiểm tra ban đầu và độ nhiễu

| Kiểm tra | Kết quả |
|---|---:|
| Số tham số / shape logits | 47.879 / `(B,7)` |
| Loss bước 0 baseline seed 1 / `ln(7)` | 2,2691 / 1,9459 |
| Overfit 20 mẫu | 1,95853 → 0,00000080 |
| Gradient các tham số | cả 6 weight/bias khác `None` và khác 0 |
| Baseline val accuracy, 3 seed | 0,9089 ± 0,0021 |
| Baseline val macro-F1, 3 seed | 0,8565 ± 0,0030 |
| Ngưỡng nhiễu `2σ` macro-F1 | 0,0060 |

Loss bước 0 seed 1 cao hơn `ln(7)` vì quy định He được áp dụng cả cho Linear output, làm logit ban đầu có phương sai đáng kể; seed 2 và 3 lần lượt cho 1,9782 và 1,9005. Tuy vậy, phép thử overfit và gradient chứng minh forward/backward đúng. Ba baseline `base-s1`, `base-s2`, `base-s3` có best epoch 19–20. Train/val loss vẫn giảm và gap chỉ khoảng 0,02–0,03, nên chưa có quá khớp mạnh. Xem `figures/compare_baseline_seeds_loss.png`.

## 3. Kết quả validation theo chủ đề

### 3.1 Loss: CE và MSE

Tôi dự đoán MSE one-hot học chậm hơn vì loss được lấy trung bình trên `B×7` phần tử, làm gradient nhỏ hơn và không trực tiếp tối ưu log-likelihood như CE. `loss-mse` đạt macro-F1 0,7377, thấp hơn `base-s1` 0,1216, vượt xa nhiễu. Không so trực tiếp trị số MSE 0,0295 với CE vì khác thang đo. Kết quả khớp dự đoán; xem `figures/compare_loss_f1.png`.

### 3.2 Optimizer

Mỗi optimizer được sweep ba learning rate trong 5 epoch, sau đó so bằng run 20 epoch ở lr tốt nhất.

| Optimizer (`exp_id`) | lr | Val macro-F1 | Best epoch |
|---|---:|---:|---:|
| SGD (`opt-sgd-best`) | 0,3 | 0,8052 | 16 |
| SGD momentum (`base-s1`) | 0,1 | 0,8593 | 20 |
| Adam (`opt-adam-best`) | 0,003 | **0,8794** | 20 |
| AdamW wd=0,01 (`opt-adamw-best`) | 0,003 | 0,8723 | 20 |

Adam giảm loss nhanh hơn nhờ chuẩn hoá bước cập nhật theo moment bậc một/hai của từng tham số. AdamW tách weight decay khỏi gradient, nhưng regularization 0,01 chưa giúp hơn Adam trong 20 epoch. Adam được lặp lại ba seed, đạt 0,8750 ± 0,0038; cải thiện ghép cặp so với baseline là `+0,0201`, `+0,0202`, `+0,0154`, đều lớn hơn `2σ` baseline. Vì vậy Adam lr 0,003 được chọn chỉ bằng validation. Xem `figures/compare_optimizer_f1.png` và `figures/compare_final_adam_seeds_f1.png`.

### 3.3 Batch size

Giữ lr 0,1 và 20 epoch, batch 128 (`batch-128`) có khoảng 2.906 update/epoch, đạt F1 0,8563 trong 4,97 s/epoch; khác biệt với baseline nằm trong nhiễu. Batch 2.048 (`batch-2048`) chỉ có khoảng 182 update/epoch, nhanh hơn (1,85 s/epoch) nhưng F1 chỉ 0,8095. Cùng số epoch không có nghĩa cùng số update; kết quả không đủ để phủ nhận linear scaling rule vì chưa tăng lr/warmup cho batch lớn. Xem `figures/compare_batch.png`.

### 3.4 Dropout

Tôi dự đoán dropout không giúp khi baseline chưa quá khớp. `drop-0p3` giảm gap train–val nhưng chỉ đạt F1 0,7730, thấp hơn baseline rõ rệt: regularization làm giảm capacity/tốc độ tối ưu trong khi chưa có vấn đề overfit cần chữa. Train loss được đo ở `eval()` nên so sánh không bị nhiễu bởi mask dropout. Xem `figures/compare_dropout_f1.png`.

### 3.5 Gradient clipping

Grad norm baseline khoảng 0,56; vì vậy chọn `c=0,5`. Ở lr 0,1, `clip-0p5` có pre-clip norm khoảng 0,58–0,70, chứng minh clipping thực sự kích hoạt, nhưng F1 0,8514 không hơn baseline. Ở lr 1,0, không clip (`highlr-1-no-clip`) dao động và chỉ đạt 0,6255; cùng lr với clip (`highlr-1-clip-0p5`) đạt 0,7986. Clipping giảm tác hại của bước cập nhật lớn nhưng không thay thế chọn lr phù hợp. Xem `figures/compare_clipping_loss.png` và `figures/compare_clipping_grad_norm.png`.

### 3.6 Mixed precision

`amp-bf16-cpu` đạt F1 0,8507, gần baseline nhưng mất 28,46 s/epoch so với 2,34 s/epoch FP32. CPU hiện tại không có backend BF16 hiệu quả; vì vậy tôi không kết luận mixed precision nhanh hơn. `peak_mem_MB=0` vì phép đo dùng CUDA API. FP16 không chạy trong tiến trình CPU: FP16 có exponent range hẹp, gradient nhỏ dễ underflow nên cần GradScaler; BF16 giữ exponent 8 bit gần FP32 nên thường không cần scale. Xem `figures/compare_amp_time.png`.

### 3.7 Initialization

| Init (`exp_id`) | Activation std sau 3 Linear | Step-0 loss | Val F1 |
|---|---|---:|---:|
| zeros (`init-zeros`) | 0; 0; 0 | 1,9459 | 0,0936 |
| normal 0,01 (`init-normal`) | 0,0342; 0,00375; 0,00027 | 1,9460 | 0,8423 |
| Xavier (`init-xavier`) | 0,274; 0,217; 0,191 | 2,0222 | 0,8480 |
| He (`base-s1`) | 0,658; 0,638; 0,577 | 2,2691 | **0,8593** |

Zeros giữ các neuron đối xứng và ReLU tại 0 chặn gradient hidden, nên mạng chỉ học bias output và trở thành bộ phân loại lớp đa số. Normal làm variance co mạnh, nhưng mạng chỉ ba lớp nên vẫn phục hồi. Xavier dùng variance `2/(fan_in+fan_out)`; He dùng `2/fan_in`, phù hợp hơn với việc ReLU loại khoảng nửa activation. Xem `figures/compare_init_f1.png`.

## 4. Đánh giá cuối trên eval

Cấu hình được cố định trước khi mở eval. Tôi rerun seed 1, nạp epoch có val loss tốt nhất và gọi `scripts/evaluate.py` đúng cho baseline và cấu hình cuối.

| Cấu hình | Seed | Val F1 | Eval F1 | Eval accuracy |
|---|---:|---:|---:|---:|
| Baseline (`base-s1`) | 1 | 0,8593 | 0,8646 | 0,9080 |
| Adam (`opt-adam-best`) | 1 | 0,8794 | **0,8789** | **0,9163** |

Adam cải thiện eval macro-F1 `+0,0143`, lớn hơn `2σ` validation 0,0060 nhưng chưa đạt ngưỡng +0,02 tối đa của rubric. Chỉ có một eval seed cho mỗi cấu hình nên chưa ước lượng được nhiễu eval; đây là hạn chế. Val và eval gần nhau, phù hợp vì split cố định có phân phối lớp tương tự.

### 4.1 Phân tích lỗi theo lớp của Adam

| Lớp | Support | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|
| 0 | 42.368 | 0,9144 | 0,9097 | 0,9121 |
| 1 | 56.661 | 0,9251 | 0,9318 | 0,9284 |
| 2 | 7.151 | 0,9103 | 0,9162 | 0,9132 |
| 3 | 549 | 0,7841 | 0,8798 | 0,8292 |
| 4 | 1.899 | 0,7955 | 0,8094 | **0,8024** |
| 5 | 3.473 | 0,8591 | 0,8235 | 0,8409 |
| 6 | 4.102 | 0,9486 | 0,9044 | 0,9260 |

Lớp 4 khó nhất và bị nhầm nhiều nhất với lớp 1 (307/1.899 mẫu). Lớp 3 rất hiếm nhưng có recall tốt; precision thấp do nhận nhầm 101 mẫu lớp 2. Mất cân bằng làm lớp 0/1 chi phối update, còn các loại rừng có địa hình gần nhau khó tách bằng MLP đơn giản. Hướng cải thiện hợp lý là class-weight/focal loss hoặc sampling cân bằng, nhưng phải chọn bằng validation mới và không chỉnh sau eval. Ma trận đầy đủ nằm trong `eval_result.json`.

## 5. Câu hỏi dẫn dắt

1. Khi tune lr công bằng, Adam thắng với F1 0,8794. Nếu dùng chung lr 0,1, Adam có thể không ổn định và kết luận sẽ phản ánh lr sai hơn là optimizer.
2. Dropout không giúp khi chưa quá khớp; nên dùng khi train loss tiếp tục giảm nhưng val loss tăng và gap mở rộng.
3. Clipping giới hạn global gradient norm để tránh bước đột biến. Cặp lr 1,0 cho thấy clip nâng F1 0,6255→0,7986 nhưng không sửa được lr quá lớn.
4. Mixed precision không nhanh hơn trên CPU này do thiếu phần cứng BF16 phù hợp và mạng nhỏ; phép đo cho thấy chậm hơn khoảng 12×.
5. Zeros hỏng vì symmetry và ReLU(0). He có variance lớn hơn Xavier để bù phần activation bị ReLU loại; khác biệt quan trọng hơn ở mạng sâu.
6. Nếu loss không giảm sau 2.000 bước, ba kiểm tra đầu tiên là: (i) kiểm tra dữ liệu/nhãn, shape, dtype và loss bước 0; (ii) overfit 20 mẫu để tách lỗi model/vòng lặp khỏi bài toán tổng quát; (iii) kiểm tra gradient từng tham số và grad norm trước update để phát hiện graph bị ngắt, neuron chết, lr quá nhỏ/lớn. Sau đó mới thay optimizer/regularization.

## 6. Hạn chế và điều bất ngờ

Step-0 loss He phụ thuộc seed và không luôn gần `ln(7)` vì output cũng dùng He. Normal nhỏ vẫn học khá tốt do mạng chỉ ba lớp. LR sweep chỉ chạy 5 epoch; batch size không được so ở cùng số update; mixed precision được đo trên CPU thay vì GPU; eval chỉ có một seed. Nếu có thêm thời gian, tôi sẽ tune Adam quanh 0,002–0,004, thử scheduler và class-balanced loss bằng validation, rồi chỉ đánh giá eval sau khi chốt một cấu hình mới độc lập.

## 7. Phụ lục

Sản phẩm gồm `experiments.xlsx`, `predictions_eval.csv`, `eval_result.json`, `baseline_eval_result.json`, 31 JSON history, 31 ảnh thí nghiệm và các ảnh comparison. Không lưu checkpoint hay dữ liệu processed trong gói nộp.
