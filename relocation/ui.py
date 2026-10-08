"""Gradio 3.50.2 UI, matching the existing setup.sh environment."""
import json
import gradio as gr
import numpy as np
from PIL import Image
from .controller import Controller, working_image
from .geometry import overlay, arrow_preview, centroid
from .config import DEPTH_MODELS


def build_ui(settings):
    control = Controller(settings)

    def reset(canvas):
        if canvas is None:
            return None, None, None, None, None, None, None, None, None, '', None, None, [], 1, 'Upload ảnh để bắt đầu.'
        raw, image = working_image(canvas)
        return None, None, raw, image, None, None, None, None, None, '', None, None, [], 1, 'Ảnh mới: vẽ scribble trong vật thể hoặc nhập text rồi lấy mask.'

    def segment(canvas, negative, text, target, progress=gr.Progress()):
        try:
            progress(.05, desc='SEEM đang lấy mask...')
            state = control.segment(canvas, negative, text)
            preview = arrow_preview(state['image'], state['mask'], target)
            message = f"Mask đã tạo · {state['segmentation_mode']} · {state['image'].shape[1]}×{state['image'].shape[0]}. Bấm vào ảnh đích để đặt tâm vật thể."
            return state, overlay(state['image'], state['mask']), state['mask'].astype(np.uint8)*255, preview, message
        except Exception as error:
            raise gr.Error(str(error)) from error

    def import_mask(canvas, uploaded, target):
        try:
            state = control.import_mask(canvas, uploaded)
            return state, overlay(state['image'], state['mask']), state['mask'].astype(np.uint8)*255, arrow_preview(state['image'], state['mask'], target), 'Đã dùng mask upload. Trắng là vật thể, đen là nền.'
        except Exception as error:
            raise gr.Error(str(error)) from error

    def choose_target(canvas, state, event: gr.SelectData):
        try:
            _, image = working_image(canvas)
            point = (int(event.index[0]), int(event.index[1]))
            if not (0 <= point[0] < image.shape[1] and 0 <= point[1] < image.shape[0]):
                raise ValueError('Điểm chọn nằm ngoài ảnh.')
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

    def run(canvas, state, target, scale, text, bg, prompt, negative, steps,
            guidance, conditioning, seed, removal_margin, target_margin, clipping, progress=gr.Progress()):
        try:
            progress(.05, desc='Chuẩn bị relocation...')
            progress(.2, desc='BrushNet: xóa nguồn và hòa trộn đích. Lần nạp model đầu có thể mất vài phút.')
            final, gallery, archive, job = control.relocate(canvas, state, target, scale, text,
                bg, prompt, negative, steps, guidance, conditioning, seed, removal_margin, target_margin, clipping)
            progress(1, desc='Hoàn tất')
            return final, gallery, archive, f'Đã lưu kết quả và metadata: {job}'
        except Exception as error:
            raise gr.Error(str(error)) from error

    with gr.Blocks(title='SEEM + BrushNet Relocation') as app:
        gr.Markdown('# Di chuyển vật thể · SEEM v1 + BrushNet\nUpload ảnh → lấy mask → chọn tâm đích → chỉnh scale → di chuyển. Giữ RGB của vật thể, dùng BrushNet để lấp nền và hòa trộn viền.')
        state = gr.State(None)
        target = gr.State(None)
        with gr.Row():
            canvas = gr.Image(source='upload', tool='sketch', type='numpy',
                label='Ảnh nguồn · vẽ nét bên trong vật thể (không vẽ mũi tên ở canvas này)', interactive=True)
            target_view = gr.Image(type='numpy', label='Chọn tâm đích · bấm lên ảnh', interactive=False)
        text = gr.Textbox(label='Mô tả vật thể (English)', placeholder='the red apple, the glasses, the person on the left...')
        gr.Markdown('Có scribble: SEEM dùng spatial prompt; text dùng cho BrushNet. Không có scribble: SEEM dùng text grounding. Mũi tên được tạo từ tâm mask đến điểm bạn bấm.')
        with gr.Accordion('Sửa vùng chọn / dùng mask có sẵn', open=False):
            negative_canvas = gr.Image(source='upload', tool='sketch', type='numpy',
                label='Negative scribble · vẽ vào vùng SEEM cần loại khỏi vật thể', interactive=True)
            uploaded_mask = gr.Image(source='upload', type='numpy', label='Mask đã sửa · trắng là vật thể, đen là nền')
            mask_button = gr.Button('Dùng mask upload')
        segment_button = gr.Button('1. Lấy mask bằng SEEM', variant='primary')
        with gr.Row():
            segmented = gr.Image(type='pil', label='Kiểm tra vùng chọn')
            source_mask = gr.Image(type='numpy', label='Mask nguồn')
        with gr.Row():
            depth_mode = gr.Dropdown(list(DEPTH_MODELS), value='Metric indoor', label='Depth · chọn đúng loại cảnh')
            depth_button = gr.Button('2. Đề xuất scale từ depth (tùy chọn)')
        with gr.Row():
            depth_preview = gr.Image(type='pil', label='Depth preview · chỉ để xem, không dùng PNG để lấy tỉ số')
            depth_info = gr.Textbox(label='Ước lượng depth', lines=6, interactive=False)
        scale = gr.Slider(.2, 3, value=1, step=.01, label='Scale cuối · tâm mask được đặt tại điểm đích')
        clipping = gr.Checkbox(value=False, label='Cho phép vật thể bị cắt ở biên ảnh')
        preview_button = gr.Button('3. Xem trước vị trí / scale')
        with gr.Row():
            pasted_preview = gr.Image(type='pil', label='Preview hình học · vật thể nguồn chưa xóa')
            target_mask = gr.Image(type='pil', label='Mask sau scale và dịch chuyển')
        bg = gr.Textbox(value='A seamless continuation of the surrounding background, matching the existing surface, texture, lighting and perspective.', label='Prompt nền sau xóa (English)')
        prompt = gr.Textbox(label='Prompt hòa trộn đích (English, có thể để trống)', placeholder='A red apple resting naturally on the wooden table...')
        negative = gr.Textbox(value='artifacts, blurry edges, duplicate objects, distorted shapes', label='Negative prompt')
        with gr.Accordion('Tham số nâng cao', open=False):
            with gr.Row():
                steps = gr.Slider(10, 80, value=30, step=1, label='Diffusion steps mỗi lượt')
                guidance = gr.Slider(1, 15, value=7.5, step=.5, label='Guidance')
                conditioning = gr.Slider(.1, 2, value=1, step=.1, label='BrushNet conditioning')
            with gr.Row():
                seed = gr.Number(value=1234, precision=0, label='Seed (lượt 2 dùng seed + 1)')
                removal_margin = gr.Slider(0, 64, value=12, step=1, label='Nới mask nguồn (pixel ảnh làm việc)')
                target_margin = gr.Slider(1, 64, value=12, step=1, label='Vùng hòa trộn biên đích (pixel ảnh làm việc)')
        run_button = gr.Button('4. Di chuyển · BrushNet hai lượt', variant='primary')
        final = gr.Image(type='pil', label='Kết quả relocation')
        gallery = gr.Gallery(label='Các bước trung gian', columns=3, height=600)
        archive = gr.File(label='Tải ảnh, mask, raw depth và metadata (ZIP)')
        status = gr.Textbox(label='Trạng thái', interactive=False)

        canvas.upload(reset, [canvas], [state, target, negative_canvas, target_view, segmented,
            source_mask, final, archive, depth_preview, depth_info, pasted_preview, target_mask, gallery, scale, status])
        canvas.clear(reset, [canvas], [state, target, negative_canvas, target_view, segmented,
            source_mask, final, archive, depth_preview, depth_info, pasted_preview, target_mask, gallery, scale, status])
        segment_button.click(segment, [canvas, negative_canvas, text, target],
            [state, segmented, source_mask, target_view, status])
        mask_button.click(import_mask, [canvas, uploaded_mask, target],
            [state, segmented, source_mask, target_view, status])
        target_view.select(choose_target, [canvas, state], [target, target_view, depth_info, status])
        depth_button.click(estimate, [canvas, state, target, depth_mode], [state, depth_preview, scale, depth_info])
        preview_button.click(preview, [canvas, state, target, scale, clipping], [pasted_preview, target_mask])
        run_button.click(run, [canvas, state, target, scale, text, bg, prompt, negative, steps,
            guidance, conditioning, seed, removal_margin, target_margin, clipping], [final, gallery, archive, status])
    return app
