"""Gradio 3.50.2 UI, matching the existing setup.sh environment."""
import json
import gradio as gr
import numpy as np
from PIL import Image
from .controller import Controller, working_image
from .geometry import overlay, arrow_preview, centroid
from .config import DEPTH_MODELS


class SketchImage(gr.Image):
    """Gradio 3 sketch canvases send mask=None until the first stroke."""
    def get_block_name(self):
        return 'image'

    def preprocess(self, value):
        if value is not None and self.tool == 'sketch':
            if isinstance(value, str):
                value = {'image': value, 'mask': None}
            if not value.get('image'):
                return None
            if not value.get('mask'):
                image = gr.processing_utils.decode_base64_to_image(value['image'])
                value = dict(value, mask=gr.processing_utils.encode_pil_to_base64(Image.new('L', image.size, 0)))
        return super().preprocess(value)


def target_point(index, image):
    """Reject missing/NaN browser coordinates before converting them to pixels."""
    try:
        coordinates = np.asarray(index, dtype=float)
    except (TypeError, ValueError):
        coordinates = np.array([])
    if coordinates.shape != (2,) or not np.isfinite(coordinates).all():
        raise ValueError('Chưa nhận được tọa độ click. Chờ ảnh tải xong rồi bấm trực tiếp lên ảnh đích; nếu vẫn lỗi, tải lại trang bằng Ctrl+Shift+R.')
    x, y = coordinates
    if not (0 <= x < image.shape[1] and 0 <= y < image.shape[0]):
        raise ValueError('Điểm chọn nằm ngoài ảnh.')
    return int(x), int(y)


def build_ui(settings):
    control = Controller(settings)

    def reset(canvas):
        if canvas is None:
            return None, None, None, None, None, None, None, None, None, '', None, None, [], 1, 'Upload ảnh để bắt đầu.', None
        raw, image = working_image(canvas)
        return None, None, raw, image, None, None, None, None, None, '', None, None, [], 1, 'Ảnh mới: vẽ scribble trong vật thể hoặc nhập text rồi lấy mask.', None

    def segment(canvas, negative, text, target, progress=gr.Progress()):
        try:
            progress(.05, desc='SEEM đang lấy mask...')
            state = control.segment(canvas, negative, text)
            preview = arrow_preview(state['image'], state['mask'], target)
            message = f"Mask đã tạo · {state['segmentation_mode']} · {state['image'].shape[1]}×{state['image'].shape[0]}. Bấm vào ảnh đích để đặt tâm vật thể."
            return state, overlay(state['image'], state['mask']), state['mask'].astype(np.uint8)*255, preview, message, None
        except Exception as error:
            raise gr.Error(str(error)) from error

    def import_mask(canvas, uploaded, target):
        try:
            state = control.import_mask(canvas, uploaded)
            return state, overlay(state['image'], state['mask']), state['mask'].astype(np.uint8)*255, arrow_preview(state['image'], state['mask'], target), 'Đã dùng mask upload. Trắng là vật thể, đen là nền.', None
        except Exception as error:
            raise gr.Error(str(error)) from error

    def choose_target(canvas, state, event: gr.SelectData):
        try:
            _, image = working_image(canvas)
            point = target_point(getattr(event, 'index', None), image)
            preview = arrow_preview(state['image'], state['mask'], point) if state else Image.fromarray(image)
            return point, preview, '', f'Tâm đích: ({point[0]}, {point[1]}). Nếu cần scale tự động, ước lượng depth lại cho điểm này.'
        except Exception as error:
            raise gr.Error(str(error)) from error

    def estimate(canvas, state, target, mode, progress=gr.Progress()):
        try:
            progress(.1, desc='Depth đang ước lượng khoảng cách...')
            updated, info, preview = control.estimate(canvas, state, target, mode)
            return updated, preview, info['scale'], json.dumps(info, ensure_ascii=False, indent=2)
        except Exception as error:
            raise gr.Error(str(error)) from error

    def preview(canvas, state, target, scale, clipping):
        try:
            return control.preview(canvas, state, target, scale, clipping)
        except Exception as error:
            raise gr.Error(str(error)) from error

    def remove(canvas, state, margin, progress=gr.Progress()):
        try:
            progress(.1, desc='LaMa đang xóa nguồn...')
            updated, background = control.remove(canvas, state, margin)
            return updated, background, 'LaMa đã xóa nguồn. Kiểm tra nền trước khi sinh vật thể tại đích.'
        except Exception as error:
            raise gr.Error(str(error)) from error

    def prompts(text, prompt, negative, mode):
        actual = prompt.strip() or f'{text.strip()}, naturally integrated with the surrounding scene, coherent lighting and texture.'
        return f'LaMa: không dùng text prompt.\nChế độ đích: {mode}\nPositive: {actual}\nNegative (những gì cần tránh): {negative.strip()}'

    def run(canvas, state, target, scale, text, bg, prompt, negative, steps,
            guidance, conditioning, seed, removal_margin, target_margin, clipping, mode, progress=gr.Progress()):
        try:
            progress(.05, desc='Chuẩn bị relocation...')
            progress(.2, desc='LaMa xóa nguồn, sau đó BrushNet xử lý đích. Lần nạp model đầu có thể mất vài phút.')
            final, gallery, archive, job = control.relocate(canvas, state, target, scale, text,
                bg, prompt, negative, steps, guidance, conditioning, seed, removal_margin, target_margin, clipping, mode)
            progress(1, desc='Hoàn tất')
            return final, gallery, archive, f'Đã lưu kết quả và metadata: {job}'
        except Exception as error:
            raise gr.Error(str(error)) from error

    with gr.Blocks(title='SEEM + LaMa + BrushNet Relocation') as app:
        gr.Markdown('# Di chuyển vật thể · SEEM + LaMa + BrushNet\nUpload → lấy mask → LaMa xóa nguồn → chọn tâm và scale → BrushNet sinh tại đích. Chế độ generate có thể tạo vật thể khác với vật thể gốc.')
        state = gr.State(None)
        target = gr.State(None)
        with gr.Row():
            canvas = SketchImage(source='upload', tool='sketch', type='numpy',
                label='Ảnh nguồn · vẽ nét bên trong vật thể (không vẽ mũi tên ở canvas này)', interactive=True)
            # Gradio 3.50.2's static image binds click to a wrapping button;
            # its coordinate helper reads naturalWidth from that button and
            # emits [null, null]. The interactive editor binds click to img.
            target_view = gr.Image(source='upload', tool='editor', type='numpy',
                label='Chọn tâm đích · bấm lên ảnh (không thay ảnh hoặc dùng crop)', interactive=True)
        text = gr.Textbox(label='Mô tả vật thể (English)', placeholder='the red apple, the glasses, the person on the left...')
        gr.Markdown('Có scribble: SEEM dùng spatial prompt. Không có scribble: SEEM dùng text grounding. LaMa xóa bằng mask, không dùng text. Negative scribble là nét loại trừ cho SEEM, khác với negative prompt của BrushNet.')
        with gr.Accordion('Sửa vùng chọn / dùng mask có sẵn', open=False):
            negative_canvas = SketchImage(source='upload', tool='sketch', type='numpy',
                label='Negative scribble · vẽ vào vùng SEEM cần loại khỏi vật thể', interactive=True)
            uploaded_mask = gr.Image(source='upload', type='numpy', label='Mask đã sửa · trắng là vật thể, đen là nền')
            mask_button = gr.Button('Dùng mask upload')
        segment_button = gr.Button('1. Lấy mask bằng SEEM', variant='primary')
        with gr.Row():
            segmented = gr.Image(type='pil', label='Kiểm tra vùng chọn')
            source_mask = gr.Image(type='numpy', label='Mask nguồn')
        removal_margin = gr.Slider(0, 64, value=8, step=1, label='Nới mask xóa LaMa (pixel ảnh làm việc)')
        remove_button = gr.Button('2. Xóa nguồn bằng LaMa / xem nền')
        removal_preview = gr.Image(type='pil', label='Nền LaMa · cần sạch vật thể nguồn trước khi sinh đích')
        with gr.Row():
            depth_mode = gr.Dropdown(list(DEPTH_MODELS), value='Metric outdoor', label='Depth · chọn Indoor nếu cảnh trong nhà')
            depth_button = gr.Button('3. Đề xuất scale từ depth (tùy chọn)')
        with gr.Row():
            depth_preview = gr.Image(type='pil', label='Depth preview · chỉ để xem, không dùng PNG để lấy tỉ số')
            depth_info = gr.Textbox(label='Ước lượng depth', lines=6, interactive=False)
        scale = gr.Slider(.2, 3, value=1, step=.01, label='Scale cuối · tâm mask được đặt tại điểm đích')
        clipping = gr.Checkbox(value=False, label='Cho phép vật thể bị cắt ở biên ảnh')
        preview_button = gr.Button('4. Xem trước vị trí / scale')
        with gr.Row():
            pasted_preview = gr.Image(type='pil', label='Preview hình học · mốc cắt–dán, chưa phải kết quả generate')
            target_mask = gr.Image(type='pil', label='Mask sau scale và dịch chuyển')
        bg = gr.State('')
        mode = gr.Radio(['generate', 'preserve'], value='generate', label='generate: sinh toàn bộ mask đích · preserve: cắt–dán và sửa viền để so sánh')
        prompt = gr.Textbox(label='Positive prompt sinh đích (English, để trống sẽ dùng mô tả vật thể)', placeholder='A sheep standing naturally on green grass.')
        negative = gr.Textbox(value='frame, cage, basket, rope, duplicate objects, artifacts, blurry edges, distorted shapes', label='Negative prompt · chỉ nhập nội dung cần tránh, không nhập kết quả mong muốn')
        prompt_button = gr.Button('Xem prompt thực tế trước khi chạy')
        prompt_info = gr.Textbox(label='Prompt sẽ dùng', lines=4, interactive=False)
        with gr.Accordion('Tham số nâng cao', open=False):
            with gr.Row():
                steps = gr.Slider(10, 80, value=30, step=1, label='Diffusion steps tại đích')
                guidance = gr.Slider(1, 15, value=7.5, step=.5, label='Guidance')
                conditioning = gr.Slider(.1, 2, value=1, step=.1, label='BrushNet conditioning')
            with gr.Row():
                seed = gr.Number(value=1234, precision=0, label='Seed BrushNet tại đích')
                target_margin = gr.Slider(0, 64, value=3, step=1, label='Nới vùng sinh đích / sửa viền (pixel ảnh làm việc)')
        run_button = gr.Button('5. LaMa → BrushNet tại mask đích', variant='primary')
        final = gr.Image(type='pil', label='Kết quả relocation')
        gallery = gr.Gallery(label='Các bước trung gian', columns=3, height=600)
        archive = gr.File(label='Tải ảnh, mask, raw depth và metadata (ZIP)')
        status = gr.Textbox(label='Trạng thái', interactive=False)

        canvas.upload(reset, [canvas], [state, target, negative_canvas, target_view, segmented,
            source_mask, final, archive, depth_preview, depth_info, pasted_preview, target_mask, gallery, scale, status, removal_preview])
        canvas.clear(reset, [canvas], [state, target, negative_canvas, target_view, segmented,
            source_mask, final, archive, depth_preview, depth_info, pasted_preview, target_mask, gallery, scale, status, removal_preview])
        segment_button.click(segment, [canvas, negative_canvas, text, target],
            [state, segmented, source_mask, target_view, status, removal_preview])
        mask_button.click(import_mask, [canvas, uploaded_mask, target],
            [state, segmented, source_mask, target_view, status, removal_preview])
        remove_button.click(remove, [canvas, state, removal_margin], [state, removal_preview, status])
        prompt_button.click(prompts, [text, prompt, negative, mode], [prompt_info])
        target_view.select(choose_target, [canvas, state], [target, target_view, depth_info, status])
        depth_button.click(estimate, [canvas, state, target, depth_mode], [state, depth_preview, scale, depth_info])
        preview_button.click(preview, [canvas, state, target, scale, clipping], [pasted_preview, target_mask])
        run_button.click(run, [canvas, state, target, scale, text, bg, prompt, negative, steps,
            guidance, conditioning, seed, removal_margin, target_margin, clipping, mode], [final, gallery, archive, status])
    return app
