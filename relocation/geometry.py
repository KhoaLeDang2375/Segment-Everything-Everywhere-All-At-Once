"""Image geometry without torch, diffusers or OpenCV dependencies."""
import hashlib
import numpy as np
from PIL import Image, ImageFilter, ImageDraw


def image_hash(image):
    image = np.ascontiguousarray(image, dtype=np.uint8)
    return hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()


def ink(mask):
    """Read RGB and transparent/black Gradio sketch masks."""
    if mask is None:
        return None
    arr = np.asarray(mask)
    if arr.ndim == 2:
        return arr > 0
    rgb = arr[..., :3].max(axis=-1)
    if arr.shape[-1] == 4:
        alpha = arr[..., 3]
        if alpha.min() != alpha.max() or (rgb.max() == 0 and alpha.max() > 0):
            return alpha > 0
    return rgb > 0


def mask_image(mask):
    return Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255, 'L')


def resize_mask(mask, size):
    return np.asarray(mask_image(mask).resize(size, Image.Resampling.NEAREST)) > 127


def centroid(mask):
    y, x = np.nonzero(mask)
    if len(x) == 0:
        raise ValueError('Mask rỗng. Hãy segment lại hoặc vẽ thêm scribble.')
    return float(x.mean()), float(y.mean())


def bounding_removal_mask(mask, padding):
    """Hide object silhouette with a filled, padded axis-aligned rectangle."""
    padding = int(padding)
    if not 0 <= padding <= 64:
        raise ValueError('Padding vùng xóa phải trong khoảng 0–64 pixel.')
    y, x = np.nonzero(mask)
    if not len(x):
        raise ValueError('Mask nguồn rỗng.')
    h, w = mask.shape
    region = np.zeros((h, w), dtype=bool)
    region[max(0, y.min()-padding):min(h, y.max()+padding+1),
           max(0, x.min()-padding):min(w, x.max()+padding+1)] = True
    return region


def dilate(mask, radius):
    radius = int(radius)
    if not 0 <= radius <= 64:
        raise ValueError('Mask margin phải nằm trong khoảng 0–64 pixel.')
    img = mask_image(mask)
    if radius:
        img = img.filter(ImageFilter.MaxFilter(2 * radius + 1))
    return np.asarray(img) > 127


def erode(mask, radius):
    return ~dilate(~np.asarray(mask, dtype=bool), radius)


def overlay(image, mask):
    arr = np.asarray(image, dtype=np.uint8).copy()
    selected = np.asarray(mask) > 0
    arr[selected] = (arr[selected] * .45 + np.array([255, 65, 65]) * .55).astype(np.uint8)
    return Image.fromarray(arr)


def arrow_preview(image, source_mask, target):
    out = overlay(image, source_mask)
    if target is None:
        return out
    draw = ImageDraw.Draw(out)
    a = np.array(centroid(source_mask))
    b = np.array(target, dtype=float)
    delta = b - a
    distance = np.linalg.norm(delta)
    if distance > 1:
        direction = delta / distance
        normal = np.array([-direction[1], direction[0]])
        length = min(18, max(6, distance * .12))
        p = b - direction * length
        draw.line([tuple(a), tuple(b)], fill=(30, 255, 110), width=3)
        draw.polygon([tuple(b), tuple(p + normal * length * .45), tuple(p - normal * length * .45)], fill=(30, 255, 110))
    draw.ellipse((b[0] - 4, b[1] - 4, b[0] + 4, b[1] + 4), fill=(30, 255, 110))
    return out


def transform_foreground(image, mask, target, scale, allow_clipping=False):
    """Warp premultiplied RGB and alpha together; map source centroid to target."""
    if not np.isfinite(scale) or not .2 <= scale <= 3:
        raise ValueError('Scale phải nằm trong khoảng 0.2–3.0.')
    arr = np.asarray(image, dtype=np.float32)
    h, w = arr.shape[:2]
    tx, ty = map(float, target)
    if not (0 <= tx < w and 0 <= ty < h):
        raise ValueError('Điểm đích phải nằm trong ảnh.')
    cx, cy = centroid(mask)
    # Pillow samples pixel centers at x+0.5/y+0.5. Correct that offset so the
    # mask centroid stays at the requested target for non-unit scales too.
    half_pixel = .5 - .5 / scale
    inverse = (1 / scale, 0, cx - tx / scale + half_pixel, 0, 1 / scale, cy - ty / scale + half_pixel)
    alpha = np.asarray(mask, dtype=np.float32)

    def warp(channel):
        return np.asarray(Image.fromarray(channel.astype(np.float32), 'F').transform(
            (w, h), Image.Transform.AFFINE, inverse, Image.Resampling.BILINEAR, fillcolor=0
        ))

    moved_alpha = np.clip(warp(alpha), 0, 1)
    premultiplied = np.stack([warp(arr[..., i] * alpha) for i in range(3)], axis=-1)
    rgb = np.divide(premultiplied, moved_alpha[..., None], out=np.zeros_like(premultiplied), where=moved_alpha[..., None] > 1e-5)
    # Compute intended geometric bounds, rather than clamping contour vertices.
    yy, xx = np.nonzero(mask)
    bounds = [scale * (xx.min() - cx) + tx, scale * (yy.min() - cy) + ty,
              scale * (xx.max() - cx) + tx, scale * (yy.max() - cy) + ty]
    clipped = bounds[0] < 0 or bounds[1] < 0 or bounds[2] > w - 1 or bounds[3] > h - 1
    if clipped and not allow_clipping:
        raise ValueError('Vật thể vượt biên ảnh. Giảm scale/đổi điểm đích hoặc bật cho phép cắt biên.')
    if moved_alpha.sum() < 1:
        raise ValueError('Vật thể sau biến đổi quá nhỏ hoặc nằm ngoài canvas.')
    return rgb.astype(np.uint8), moved_alpha, {'source_center': [cx, cy], 'target_center': [tx, ty], 'scale': float(scale), 'clipped': bool(clipped), 'target_bounds': bounds}


def scale_to_fit(mask, target, requested_scale):
    """Shrink to fit the canvas while keeping the selected centroid fixed."""
    h, w = mask.shape
    tx, ty = map(float, target)
    if not (0 <= tx < w and 0 <= ty < h):
        raise ValueError('Điểm đích phải nằm trong ảnh.')
    if not np.isfinite(requested_scale) or not .2 <= requested_scale <= 3:
        raise ValueError('Scale phải nằm trong khoảng 0.2–3.0.')
    cx, cy = centroid(mask)
    yy, xx = np.nonzero(mask)
    limits = [3.0]
    for available, extent in [(tx, cx-xx.min()), (w-1-tx, xx.max()-cx),
                              (ty, cy-yy.min()), (h-1-ty, yy.max()-cy)]:
        if extent > 0:
            limits.append(available/extent)
    maximum = min(limits)
    if maximum < .2:
        raise ValueError('Tâm đích quá sát biên để giữ toàn bộ vật thể ở scale tối thiểu 0.2. Chọn tâm xa biên hơn hoặc bật cho phép cắt biên.')
    # Small inward margin avoids floating-point roundoff at the exact boundary.
    return float(min(requested_scale, max(.2, maximum*.999)))


def composite(background, foreground, alpha):
    alpha = np.asarray(alpha, dtype=np.float32)[..., None]
    return np.clip(np.asarray(background) * (1 - alpha) + np.asarray(foreground) * alpha, 0, 255).astype(np.uint8)


def harmonization_mask(alpha, margin):
    """Allow generation around the pasted object while retaining its RGB core."""
    support = alpha > .01
    core = erode(alpha > .95, max(1, int(margin) // 2))
    ring = dilate(support, margin) & ~core
    # A tiny object may have no eroded core: preserve a small interior instead.
    if not core.any():
        core = alpha > .95
        ring &= ~core
    return ring, core


def blend_repair(original, generated, repair_mask, feather=2):
    """Diffusion output may change globally; only keep pixels in the repair region."""
    mask = mask_image(repair_mask)
    if feather:
        # Inward feather: no diffusion pixels escape the recorded repair region.
        blurred = np.asarray(mask.filter(ImageFilter.GaussianBlur(feather)), dtype=np.float32) / 255
        alpha = blurred * np.asarray(repair_mask, dtype=np.float32)
    else:
        alpha = np.asarray(repair_mask, dtype=np.float32)
    return composite(original, generated, alpha)


def suggest_scale(depth, mask, target, is_metric=True, bounds=(.5, 2.0)):
    """Ratios are only meaningful for metric distances, never display-normalized maps."""
    depth = np.asarray(depth, dtype=np.float32)
    if depth.shape != np.asarray(mask).shape:
        raise ValueError('Depth và mask phải có cùng kích thước.')
    if target is None:
        raise ValueError('Chọn điểm đích trước khi ước lượng depth.')
    cx, cy = centroid(mask)
    tx, ty = map(int, target)
    h, w = depth.shape
    if not (0 <= tx < w and 0 <= ty < h):
        raise ValueError('Điểm đích nằm ngoài ảnh.')
    radius = max(3, int(min(h, w) * .015))
    source = depth[erode(mask, 1)]
    if source.size < 8:
        source = depth[np.asarray(mask, dtype=bool)]
    target_slice = (slice(max(0, ty-radius), min(h, ty+radius+1)),
                    slice(max(0, tx-radius), min(w, tx+radius+1)))
    # The original object hides its supporting surface. Do not pretend that
    # depth on that object is the background distance at an overlapping target.
    target_values = depth[target_slice][~np.asarray(mask, dtype=bool)[target_slice]]

    def stats(values):
        values = values[np.isfinite(values) & (values > 0)]
        if not values.size:
            return None, None
        median = float(np.median(values))
        uncertainty = float(np.median(np.abs(values-median)) / max(median, 1e-6))
        return median, uncertainty

    za, ua = stats(source)
    zb, ub = stats(target_values)
    info = {'source_depth': za, 'target_depth': zb, 'source_relative_mad': ua,
            'target_relative_mad': ub, 'metric': bool(is_metric), 'scale': 1.0}
    if not is_metric:
        info['reason'] = 'Depth tương đối chỉ dùng để xem; không lấy tỉ số làm scale vật lý. Chỉnh scale thủ công.'
    elif za is None or zb is None or max(ua, ub) > .35:
        info['reason'] = 'Depth vùng nguồn/đích không ổn định; dùng scale 1 và chỉnh thủ công.'
    else:
        raw = za / zb
        info.update(raw_scale=raw, scale=float(np.clip(raw, *bounds)), reason='Scale = Z(nguồn)/Z(đích); đã giới hạn trong [0.5, 2]. Đây là ước lượng cần xem lại.')
    return info
