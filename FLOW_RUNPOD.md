# Flow relocation và hướng dẫn chạy RunPod

## 1. Mục tiêu

Upload ảnh, chọn vật thể bằng scribble hoặc text, xóa vật thể nguồn, đặt một
mask đích đã scale tại điểm người dùng chọn và sinh vật thể trong vùng đó.
Có hai backend xóa nguồn: **LaMa** và **BrushNet**, để so sánh trực quan.

Điểm đích là **tâm mask vật thể**, không phải chân hoặc điểm tiếp xúc mặt đất.
Hướng vật thể không được xoay bằng phép biến đổi hình học.

## 2. Flow

```mermaid
flowchart TD
    I[Ảnh upload + scribble hoặc text] --> S[SEEM: mask nguồn chính xác]
    S --> R[Bounding box nguồn + padding: mask xóa chữ nhật]
    I --> L[LaMa: ảnh + mask xóa]
    R --> L
    I --> B[BrushNet: ảnh + mask xóa + prompt nền]
    R --> B
    L --> C[So sánh hai nền; chọn backend]
    B --> C
    S --> T[Scale mask chính xác; dịch tâm đến điểm đích]
    D[Depth metric hoặc scale thủ công] --> T
    C --> G[BrushNet tại vùng đích]
    T --> G
    P[Positive và negative prompt đích] --> G
    G --> O[Kết quả + ảnh trung gian + metadata ZIP]
```

Các GPU task chạy lần lượt. Sau SEEM, model và cache language embedding được
chuyển về CPU. Mỗi worker LaMa/Depth/BrushNet kết thúc trước bước tiếp theo.
Khi chọn BrushNet cho cả xóa và sinh đích, hai worker riêng được chạy tuần tự;
không giữ hai pipeline BrushNet đồng thời trong VRAM, nhưng phải nạp lại model.

## 3. Phân biệt các mask

| Mask | Cách tạo | Vai trò |
|---|---|---|
| source_mask.png | SEEM hoặc mask upload đã sửa | Hình dáng nguồn chính xác |
| removal_mask.png | Chữ nhật bao tất cả pixel nguồn, nới padding và giới hạn theo biên ảnh | Xóa nguồn; không truyền hình đầu/thân/chân |
| target_mask.png | Scale mask nguồn quanh tâm, dịch đến tâm đích | Hình dáng và vị trí đích |
| generation_mask.png | Toàn bộ mask đích nới nhẹ trong generate, hoặc vòng viền trong preserve | Pixel BrushNet được phép sửa tại đích |

**Padding xóa không phải chỉ dilation silhouette.** Chỉ nới silhouette vài pixel
vẫn giữ hình con vật. Ở bản này, mọi pixel bên trong bounding box đều là trắng,
kể cả khoảng trống giữa chân, rồi mới thêm padding. Mask chính xác ban đầu
không bị sửa; nó tiếp tục được dùng cho target.

Đổi lại, hình chữ nhật xóa nhiều nền hơn mask chính xác. Padding quá lớn làm mất
ngữ cảnh và có thể giảm chất lượng. Bắt đầu với **8 px** ở kích thước ảnh làm việc.
BrushNet vẫn có thể sinh vật thể ngoài ý muốn; đổi mask không phải bảo đảm xóa.

## 4. Hai backend xóa nguồn

### LaMa

* Big-LaMa TorchScript, môi trường .venv-lama riêng.
* Nhận ảnh + mask xóa, không dùng text prompt.
* Chỉ ghép output trong vùng xóa, giữ nguyên bên ngoài.

### BrushNet

* Dùng pipeline SD 1.5 và checkpoint BrushNet hiện có.
* Mask xóa là hình chữ nhật giống LaMa; RGB vùng đó được đặt về đen khi inference.
* Positive prompt chỉ mô tả nền cần nối lại.
* Negative prompt xóa độc lập với negative prompt đích.
* Không tự chuyển sang checkpoint random-mask; checkpoint hiện tại được tái sử dụng.

Ví dụ cho ảnh con cừu trên cỏ:

```text
Positive prompt nền:
A continuous green lawn, dense grass matching the surrounding texture, lighting and perspective.

Negative prompt XÓA:
animal, sheep, dog, tiger, person, body, head, legs, fur, duplicate objects, artifacts
```

Không dùng câu mô tả kết quả mong muốn làm negative prompt.

## 5. Tại vị trí đích

### generate — mặc định

BrushNet nhận nền đã xóa + toàn bộ mask đích nới nhẹ + text. Nó sinh lại vật thể
trong vùng đó. Không dùng RGB nguồn làm conditioning ảnh tham chiếu.
Mask quy định vùng sửa; chưa bắt buộc đúng silhouette và nhận dạng vật thể nguồn.

```text
Positive prompt ĐÍCH:
A sheep standing naturally on green grass.

Negative prompt ĐÍCH:
frame, basket, cage, rope, duplicate objects, artifacts, distorted shapes
```

Nếu để trống positive đích, code lấy mô tả vật thể và thêm câu về lighting/texture.
Nút xem prompt thực tế hiển thị positive/negative nguồn và đích tương ứng.

### preserve — để đối chiếu

Scale và dán RGB + alpha nguồn lên nền sạch, sau đó BrushNet sửa một vòng viền.
Giữ phần lõi; vòng viền vẫn có thể sinh nội dung sai. Padding đích mặc định 3 px.
Ảnh pasted.png là mốc cắt–dán, không phải input tham chiếu cho generate.

## 6. Chọn RunPod

* Khởi đầu: 1 GPU **RTX 2000 Ada 16 GB** hoặc A5000 24 GB; inference 512.
* Host RAM ít nhất 32 GB, ưu tiên nhiều hơn nếu có vì CPU offload.
* Template phù hợp: Ubuntu 22.04, Python 3.10, CUDA 12.1 bản devel để build SEEM.
* HTTP ports: **7860** (Gradio), 8888 nếu cần Jupyter; TCP 22 nếu dùng SSH.
* Dùng persistent storage tại /workspace cho code, venv, cache và checkpoint.

16 GB cần xác nhận bằng inference thực tế. Đây không phải cam kết đủ ở mọi
kích thước ảnh. Theo dõi nvidia-smi cùng số đo peak trong JSON từng worker.

## 7. Pod đã chạy được phiên bản có LaMa

Dừng demo cũ bằng Ctrl+C. Kích hoạt đúng môi trường đã chạy demo_v1.py.

```bash
cd /workspace/Segment-Everything-Everywhere-All-At-Once
git fetch origin
git checkout exp_v1
git pull --ff-only origin exp_v1

export BRUSHNET_REPO=/workspace/BrushNet
export HF_HOME=/workspace/hf-cache
export RELOCATION_OUTPUT_DIR=/workspace/relocation_outputs

python3 demo_relocation.py --preflight
python3 demo_relocation.py --server-name 0.0.0.0 --port 7860 --max-side 512
```

Không cần cài lại dependencies cho lựa chọn BrushNet xóa nguồn mới.

Nếu chưa cài LaMa ở lần cập nhật trước, chạy thêm trước preflight:

```bash
bash setup_relocation.sh --only-lama
```

## 8. Pod mới

```bash
cd /workspace
git clone --branch exp_v1 https://github.com/KhoaLeDang2375/Segment-Everything-Everywhere-All-At-Once.git
git clone https://github.com/TencentARC/BrushNet.git
cd Segment-Everything-Everywhere-All-At-Once

export BRUSHNET_REPO=/workspace/BrushNet
export HF_HOME=/workspace/hf-cache
export RELOCATION_OUTPUT_DIR=/workspace/relocation_outputs

python3 --version
nvidia-smi
python3 -m pip install 'numpy<2' 'setuptools<70'
python3 -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
bash setup_relocation.sh

python3 demo_relocation.py --preflight
python3 demo_relocation.py --server-name 0.0.0.0 --port 7860 --max-side 512
```

Nếu SEEM đã được setup và chạy được nhưng chưa có worker mới, dùng
bash setup_relocation.sh --skip-seem thay cho setup đầy đủ.

Mở RunPod → Connect → HTTP service 7860. Ctrl+Shift+R sau khi cập nhật code.
Giữ Gradio 3.50.2; notice nâng cấp hoặc gợi ý share=True không phải lỗi.

## 9. Quy trình so sánh trên giao diện

1. Upload ảnh, scribble trong vật thể; nhập mô tả rồi lấy mask.
2. Kiểm tra mask nguồn, upload mask sửa nếu cần.
3. Chọn padding chữ nhật và xem mask xóa.
4. Nhập prompt nền/negative xóa. Nút **So sánh LaMa và BrushNet cạnh nhau** chạy
   cả hai tuần tự trên cùng ảnh và mask, hiển thị hai kết quả.
5. Chọn backend dùng cho đích. Bấm nút xóa backend đã chọn để cập nhật preview;
   khi chạy cuối, code luôn kiểm tra cache theo backend và tham số hiện tại.
6. Chọn tâm đích, chọn Indoor/Outdoor depth đúng cảnh hoặc chỉnh scale thủ công.
   Nếu vật thể vượt biên, preview vẫn hiển thị phần bị cắt và hướng dẫn ở ô trạng
   thái. Bấm **Thu nhỏ để vừa ảnh** để giảm scale, giữ nguyên tâm đã chọn; hoặc
   bật cho phép cắt biên. Chạy thật vẫn yêu cầu một trong hai lựa chọn này.
   Tâm sát biên đến mức scale 0.2 không vừa thì phải đổi tâm hoặc chấp nhận cắt.
7. Giữ cùng scale, tâm đích, prompt đích, seed và tham số khi so sánh kết quả cuối.
8. Chạy generate hoặc preserve; tải ZIP mỗi lần để lưu thí nghiệm.

LaMa và BrushNet có quy tắc resize nội bộ khác nhau để tương thích từng model;
không coi đây là benchmark chất lượng có kiểm soát tuyệt đối. Metadata lưu
inference_size, prompts, seed, checkpoint, thời gian và peak VRAM.
Cache LaMa bỏ qua text vì model không dùng text. Cache BrushNet có prompt,
negative, seed, steps, guidance, conditioning, padding và checkpoint.

## 10. Đọc output và xử lý lỗi

* background.png: đánh giá việc xóa nguồn trước tiên.
* removal_raw.png: raw output; chỉ vùng removal_mask được ghép lại.
* target_mask.png: xem scale/tâm; generation_mask.png: vùng sửa tại đích.
* pasted.png: so sánh hình học với vật thể nguồn.
* harmonization_raw.png và result.png: output đích raw và kết quả đã giới hạn vùng.
* removal_result.json: backend xóa, mask shape, checkpoint và số đo.
* removal_brushnet_request/result.json hoặc removal_lama_request/result.json:
  provenance lượt xóa, được đổi tiền tố để không bị worker đích ghi đè.
* brushnet_request/result.json: lượt tại đích; metadata.json tổng hợp thí nghiệm.

GPU workers dùng môi trường riêng. Logs nằm trong thư mục job trên Pod, không
đưa vào ZIP. Preflight chỉ kiểm tra imports/path/CUDA, không xác nhận chất lượng.

Hai lỗi Gradio đã sửa: mask=None trước nét vẽ được chuyển thành mask rỗng trước
preprocess; negative canvas không có nét vẽ bị bỏ qua, nét vẽ thật được kiểm tra
kích thước/nội dung ảnh với dung sai nhỏ. Negative canvas của ảnh khác vẫn bị từ chối.

## 11. Giới hạn

Depth tại tâm đích lấy từ nền nhìn thấy; không chắc bằng depth tâm vật thể tương
lai. Scale tự động là đề xuất. Mask đích không chứa RGB/identity. Pipeline chưa
giải quyết đầy đủ bóng, occlusion, điểm tiếp xúc, orientation 3D hoặc identity
khi generate. Nên xác nhận nền sạch rồi mới đánh giá bước sinh đích.

Tham khảo code chi tiết và dependency isolation: [RELOCATION.md](RELOCATION.md).
Kế hoạch revision: [RELOCATION_PLAN.md](RELOCATION_PLAN.md).
