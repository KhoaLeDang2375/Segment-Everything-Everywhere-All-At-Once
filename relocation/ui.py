"""Gradio 3.50.2 UI, matching the existing setup.sh environment."""
import json
import gradio as gr
import numpy as np
from PIL import Image
from .controller import Controller, working_image
from .geometry import overlay, arrow_preview, centroid
from .config import DEPTH_MODELS
from .geometry import bounding_removal_mask, mask_image, scale_to_fit


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
            return None, None, None, None, None, None, None, None, None, '', None, None, [], 1, 'Upload ảnh để bắt đầu.', None, None, None, None
        raw, image = working_image(canvas)
        return None, None, raw, image, None, None, None, None, None, '', None, None, [], 1, 'Ảnh mới: vẽ scribble trong vật thể hoặc nhập text rồi lấy mask.', None, None, None, None

    def segment(canvas, negative, text, target, progress=gr.Progress()):
        try:
            progress(.05, desc='SEEM đang lấy mask...')
            state = control.segment(canvas, negative, text)
            preview = arrow_preview(state['image'], state['mask'], target)
            message = f"Mask đã tạo · {state['segmentation_mode']} · {state['image'].shape[1]}×{state['image'].shape[0]}. Bấm vào ảnh đích để đặt tâm vật thể."
            return state, overlay(state['image'], state['mask']), state['mask'].astype(np.uint8)*255, preview, message, None, None, None, None
        except Exception as error:
            raise gr.Error(str(error)) from error

    def import_mask(canvas, uploaded, target):
        try:
            state = control.import_mask(canvas, uploaded)
            return state, overlay(state['image'], state['mask']), state['mask'].astype(np.uint8)*255, arrow_preview(state['image'], state['mask'], target), 'Đã dùng mask upload. Trắng là vật thể, đen là nền.', None, None, None, None
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
            pasted, mask, info = control.preview(canvas, state, target, scale, clipping, return_info=True)
            if not info['clipped']:
                message = f'Preview hợp lệ · scale {scale:.3f}, toàn bộ vật thể nằm trong ảnh.'
            elif clipping:
                message = f'Preview bị cắt ở biên · scale {scale:.3f}. Bạn đã bật cho phép cắt biên; có thể chạy với phần mask còn trong ảnh.'
            elif info['fit_scale'] is not None:
                message = f'Preview cho thấy vật thể vượt biên. Bấm “Thu nhỏ để vừa ảnh” (scale khoảng {info["fit_scale"]:.3f}), đổi tâm hoặc bật cho phép cắt biên trước khi chạy.'
            else:
                message = 'Tâm quá sát biên: chọn tâm xa biên hơn hoặc bật cho phép cắt biên. Preview chỉ hiển thị phần nằm trong ảnh.'
            return pasted, mask, message
        except ValueError as error:
            return None, None, str(error)
        except Exception as error:
            raise gr.Error(str(error)) from error

    def fit(canvas, state, target, scale, clipping):
        try:
            control.validate(canvas, state)
            if target is None:
                raise ValueError('Chọn tâm đích trước.')
            fitted = scale_to_fit(state['mask'], target, scale)
            pasted, mask, _ = control.preview(canvas, state, target, fitted, clipping, return_info=True)
            return fitted, pasted, mask, f'Đã chỉnh scale từ {scale:.3f} xuống {fitted:.3f} để vừa ảnh. Tâm đích được giữ nguyên; có thể chạy tiếp.'
        except ValueError as error:
            return gr.update(), gr.update(), gr.update(), str(error)
        except Exception as error:
            raise gr.Error(str(error)) from error

    def remove(canvas, state, margin, backend, bg, removal_negative, steps, guidance, conditioning, seed, progress=gr.Progress()):
        try:
            progress(.1, desc=f'{backend} đang xóa nguồn...')
            updated, background = control.remove(canvas, state, margin, backend, bg, removal_negative, steps, guidance, conditioning, seed)
            return updated, background, mask_image(bounding_removal_mask(state['mask'], margin)), f'{backend} đã xóa nguồn bằng mask chữ nhật có padding. Kiểm tra nền trước khi sinh đích.'
        except Exception as error:
            raise gr.Error(str(error)) from error

    def compare(canvas, state, margin, backend, bg, removal_negative, steps, guidance, conditioning, seed, progress=gr.Progress()):
        try:
            progress(.1, desc='So sánh: LaMa...')
            updated, lama_image = control.remove(canvas, state, margin, 'lama', bg, removal_negative, steps, guidance, conditioning, seed)
            progress(.5, desc='So sánh: BrushNet...')
            updated, brush_image = control.remove(canvas, updated, margin, 'brushnet', bg, removal_negative, steps, guidance, conditioning, seed)
            updated, selected = control.remove(canvas, updated, margin, backend, bg, removal_negative, steps, guidance, conditioning, seed)
            return updated, selected, lama_image, brush_image, mask_image(bounding_removal_mask(state['mask'], margin)), f'Đã so sánh cùng ảnh và mask chữ nhật. Nền dùng cho đích: {backend}.'
        except Exception as error:
            raise gr.Error(str(error)) from error

    def prompts(text, prompt, negative, mode, backend, bg, removal_negative):
        actual = prompt.strip() or f'{text.strip()}, naturally integrated with the surrounding scene, coherent lighting and texture.'
        removal = 'LaMa: không dùng text prompt.' if backend == 'lama' else f'BrushNet xóa — Positive: {bg.strip()}\nNegative xóa: {removal_negative.strip()}'
        return f'{removal}\nChế độ đích: {mode}\nPositive đích: {actual}\nNegative đích: {negative.strip()}'

    def background_preset(choice):
        if choice == 'Cỏ / bãi cỏ':
            return ('A continuous green lawn, dense natural grass matching the surrounding texture, lighting and perspective, an empty grassy area.',
                    'animal, sheep, dog, tiger, person, body, head, legs, fur, wall, concrete, building, panel, frame, cage, basket, artifacts')
        return gr.update(), gr.update()

    def run(canvas, state, target, scale, text, bg, prompt, negative, steps,
            guidance, conditioning, seed, removal_margin, target_margin, clipping, mode, backend, removal_negative,
            use_ip_adapter, ip_adapter_scale, progress=gr.Progress()):
        try:
            progress(.05, desc='Chuẩn bị relocation...')
            progress(.2, desc=f'{backend} xóa nguồn, sau đó BrushNet xử lý đích. Lần nạp model đầu có thể mất vài phút.')
            final, gallery, archive, job = control.relocate(canvas, state, target, scale, text,
                bg, prompt, negative, steps, guidance, conditioning, seed, removal_margin, target_margin, clipping, mode, backend, removal_negative,
                use_ip_adapter, ip_adapter_scale)
            progress(1, desc='Hoàn tất')
            return final, gallery, archive, f'Đã lưu kết quả và metadata: {job}'
        except ValueError as error:
            return gr.update(), gr.update(), gr.update(), str(error)
        except Exception as error:
            raise gr.Error(str(error)) from error

    with gr.Blocks(title='SEEM + LaMa + BrushNet Relocation') as app:
        gr.Markdown('# Di chuyển vật thể · SEEM + LaMa / BrushNet\nUpload → lấy mask → so sánh xóa nguồn → chọn tâm và scale → BrushNet sinh tại đích. Mask xóa là chữ nhật; mask đích giữ hình vật thể. Generate có thể tạo vật thể khác với vật thể gốc.')
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
        backend = gr.Radio(['lama', 'brushnet'], value='lama', label='Backend xóa nguồn · nền của backend này sẽ dùng cho bước đích')
        removal_margin = gr.Slider(0, 64, value=8, step=1, label='Padding chữ nhật bao vật thể (pixel ảnh làm việc)')
        bg = gr.Textbox(value='A seamless continuation of the surrounding background, matching the existing surface, texture, lighting and perspective.', label='Positive prompt nền · chỉ dùng khi BrushNet xóa nguồn', placeholder='A continuous green lawn matching the surrounding grass.')
        removal_negative = gr.Textbox(value='animal, person, body, head, legs, fur, duplicate objects, artifacts', label='Negative prompt XÓA · tách riêng với negative prompt đích')
        bg_preset = gr.Dropdown(['Tự nhập', 'Cỏ / bãi cỏ'], value='Tự nhập', label='Gợi ý prompt xóa theo nền · chọn đúng cảnh, rồi có thể sửa text')
        remove_button = gr.Button('2. Xóa nguồn bằng backend đã chọn')
        compare_button = gr.Button('So sánh LaMa và BrushNet cạnh nhau')
        removal_mask_view = gr.Image(type='pil', label='Mask xóa chung · chữ nhật có padding, không dùng silhouette')
        removal_preview = gr.Image(type='pil', label='Nền được chọn · cần sạch vật thể nguồn trước khi sinh đích')
        with gr.Row():
            comparison_lama = gr.Image(type='pil', label='So sánh · LaMa')
            comparison_brushnet = gr.Image(type='pil', label='So sánh · BrushNet')
        with gr.Row():
            depth_mode = gr.Dropdown(list(DEPTH_MODELS), value='Metric outdoor', label='Depth · chọn Indoor nếu cảnh trong nhà')
            depth_button = gr.Button('3. Đề xuất scale từ depth (tùy chọn)')
        with gr.Row():
            depth_preview = gr.Image(type='pil', label='Depth preview · chỉ để xem, không dùng PNG để lấy tỉ số')
            depth_info = gr.Textbox(label='Ước lượng depth', lines=6, interactive=False)
        scale = gr.Slider(.2, 3, value=1, step=.01, label='Scale cuối · tâm mask được đặt tại điểm đích')
        clipping = gr.Checkbox(value=False, label='Cho phép vật thể bị cắt ở biên ảnh')
        preview_button = gr.Button('4. Xem trước vị trí / scale')
        fit_button = gr.Button('Thu nhỏ để vừa ảnh · giữ nguyên tâm đích')
        with gr.Row():
            pasted_preview = gr.Image(type='pil', label='Preview hình học · mốc cắt–dán, chưa phải kết quả generate')
            target_mask = gr.Image(type='pil', label='Mask sau scale và dịch chuyển')
        mode = gr.Radio(['generate', 'preserve'], value='generate', label='generate: sinh toàn bộ mask đích · preserve: cắt–dán và sửa viền để so sánh')
        with gr.Row():
            use_ip_adapter = gr.Checkbox(value=True, label='IP-Adapter Plus · dùng ảnh vật thể nguồn khi sinh đích')
            ip_adapter_scale = gr.Slider(0, 1, value=.6, step=.05, label='Độ ảnh hưởng ảnh tham chiếu · bắt đầu 0.6')
        gr.Markdown('IP-Adapter hỗ trợ giữ diện mạo, không đảm bảo giống từng chi tiết hay đúng silhouette tuyệt đối. Chỉ áp dụng tại đích. Lượt xóa BrushNet dùng checkpoint BrushNetX riêng; hãy mô tả đúng nền trong prompt xóa.')
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
        run_button = gr.Button('5. Xóa nguồn → BrushNet tại mask đích', variant='primary')
        final = gr.Image(type='pil', label='Kết quả relocation')
        gallery = gr.Gallery(label='Các bước trung gian', columns=3, height=600)
        archive = gr.File(label='Tải ảnh, mask, raw depth và metadata (ZIP)')
        status = gr.Textbox(label='Trạng thái', interactive=False)

        canvas.upload(reset, [canvas], [state, target, negative_canvas, target_view, segmented,
            source_mask, final, archive, depth_preview, depth_info, pasted_preview, target_mask, gallery, scale, status, removal_preview, comparison_lama, comparison_brushnet, removal_mask_view])
        canvas.clear(reset, [canvas], [state, target, negative_canvas, target_view, segmented,
            source_mask, final, archive, depth_preview, depth_info, pasted_preview, target_mask, gallery, scale, status, removal_preview, comparison_lama, comparison_brushnet, removal_mask_view])
        segment_button.click(segment, [canvas, negative_canvas, text, target],
            [state, segmented, source_mask, target_view, status, removal_preview, comparison_lama, comparison_brushnet, removal_mask_view])
        mask_button.click(import_mask, [canvas, uploaded_mask, target],
            [state, segmented, source_mask, target_view, status, removal_preview, comparison_lama, comparison_brushnet, removal_mask_view])
        removal_inputs = [canvas, state, removal_margin, backend, bg, removal_negative, steps, guidance, conditioning, seed]
        bg_preset.change(background_preset, [bg_preset], [bg, removal_negative])
        remove_button.click(remove, removal_inputs, [state, removal_preview, removal_mask_view, status])
        compare_button.click(compare, removal_inputs, [state, removal_preview, comparison_lama, comparison_brushnet, removal_mask_view, status])
        prompt_button.click(prompts, [text, prompt, negative, mode, backend, bg, removal_negative], [prompt_info])
        target_view.select(choose_target, [canvas, state], [target, target_view, depth_info, status])
        depth_button.click(estimate, [canvas, state, target, depth_mode], [state, depth_preview, scale, depth_info])
        preview_button.click(preview, [canvas, state, target, scale, clipping], [pasted_preview, target_mask, status])
        fit_button.click(fit, [canvas, state, target, scale, clipping], [scale, pasted_preview, target_mask, status])
        run_button.click(run, [canvas, state, target, scale, text, bg, prompt, negative, steps,
            guidance, conditioning, seed, removal_margin, target_margin, clipping, mode, backend, removal_negative,
            use_ip_adapter, ip_adapter_scale], [final, gallery, archive, status])
    return app
