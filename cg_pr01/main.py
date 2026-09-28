from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_image(path: Path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Файл не найден: {path.resolve()}")
    try:
        with Image.open(path) as src:
            src.load()
            return src.convert("RGB"), src.format, src.mode, "A" in src.getbands()
    except UnidentifiedImageError as exc:
        raise ValueError(f"Файл не является поддерживаемым изображением: {path}") from exc


def change_brightness(image_array, delta):
    result = image_array.astype(np.int16) + delta
    result = np.clip(result, 0, 255)
    return result.astype(np.uint8)


def resize_keep_aspect(image: Image.Image, target_width: int) -> Image.Image:
    if not isinstance(target_width, int) or target_width <= 0:
        raise ValueError(f"Некорректная ширина: {target_width}. Должна быть положительным целым числом.")
    scale = target_width / image.width
    target_height = max(1, round(image.height * scale))
    return image.resize((target_width, target_height), resample=Image.Resampling.LANCZOS)


def to_grayscale_numpy(array):
    r = array[:, :, 0].astype(np.float32)
    g = array[:, :, 1].astype(np.float32)
    b = array[:, :, 2].astype(np.float32)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip(gray, 0, 255).astype(np.uint8)


def process_region(array, y1, y2, x1, x2):
    h, w = array.shape[:2]
    if not (0 <= y1 < y2 <= h) or not (0 <= x1 < x2 <= w):
        raise ValueError(f"Координаты области выходят за пределы изображения ({h}x{w}): y[{y1}:{y2}], x[{x1}:{x2}]")

    result = array.copy()
    gray_region = to_grayscale_numpy(result[y1:y2, x1:x2])
    result[y1:y2, x1:x2, 0] = gray_region
    result[y1:y2, x1:x2, 1] = gray_region
    result[y1:y2, x1:x2, 2] = gray_region
    return result


def save_result(image, filename):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image, mode="L" if image.ndim == 2 else "RGB")
    image.save(OUTPUT_DIR / filename)


def rotate_image(image, angle, expand=True):
    if not isinstance(angle, (int, float)):
        raise ValueError(f"Некорректный угол поворота: {angle!r}")
    return image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=expand, fillcolor=(255, 255, 255))


def main():
    input_path = Path("assets/source.png")

    try:
        image, img_format, orig_mode, has_alpha = load_image(input_path)
    except (FileNotFoundError, ValueError) as exc:
        print("Ошибка загрузки изображения:", exc)
        return

    print("==============================================")
    print("Задание 1. Загрузка и исследование изображения")
    print("==============================================")
    print("Имя файла:", input_path.name)
    print("Формат изображения:", img_format)
    print("Размер изображения (image.size):", image.size)
    print("Цветовой режим Pillow (исходный):", orig_mode)
    print("Ширина:", image.width, "| Высота:", image.height)
    print("Альфа-канал:", "есть" if has_alpha else "отсутствует")

    array = np.asarray(image)
    print("\n--- Массив NumPy ---")
    print("shape:", array.shape)
    print("dtype:", array.dtype)
    print("min:", array.min(), "| max:", array.max())
    print("Количество каналов:", array.shape[2])

    h, w = array.shape[:2]
    points = [(0, 0), (h // 2, w // 2), (h - 1, w - 1)]
    print("\n--- RGB трёх пикселей ---")
    for y, x in points:
        r, g, b = array[y, x]
        print(f"Пиксель (x={x}, y={y}): R={r}, G={g}, B={b}")

    print("\n==============================================")
    print("Задание 2. Работа с цветовыми каналами")
    print("==============================================")
    red_channel, green_channel, blue_channel = array[:, :, 0], array[:, :, 1], array[:, :, 2]
    for name, channel in {"R": red_channel, "G": green_channel, "B": blue_channel}.items():
        print(f"{name}: min={channel.min()}, max={channel.max()}, mean={channel.mean():.2f}")

    red_only, green_only, blue_only = np.zeros_like(array), np.zeros_like(array), np.zeros_like(array)
    red_only[:, :, 0] = red_channel
    green_only[:, :, 1] = green_channel
    blue_only[:, :, 2] = blue_channel
    save_result(red_only, "red_channel.png")
    save_result(green_only, "green_channel.png")
    save_result(blue_only, "blue_channel.png")

    print("\n==============================================")
    print("Задание 3. Преобразование яркости")
    print("==============================================")
    darker = change_brightness(array, -60)
    brighter = change_brightness(array, 60)
    print("Темнее:  min/max", darker.min(), darker.max())
    print("Светлее: min/max", brighter.min(), brighter.max())
    save_result(darker, "darker.png")
    save_result(brighter, "brighter.png")

    print("\n==============================================")
    print("Задание 4. Изменение размера изображения")
    print("==============================================")
    try:
        resized_50 = resize_keep_aspect(image, round(image.width * 0.5))
        resized_150 = resize_keep_aspect(image, round(image.width * 1.5))
        resized_fixed = resize_keep_aspect(image, 400)
    except ValueError as exc:
        print("Ошибка изменения размера:", exc)
        resized_50 = resized_150 = resized_fixed = image
    print("50%:", resized_50.size, "| 150%:", resized_150.size, "| ширина 400:", resized_fixed.size)
    save_result(resized_50, "resized_50.png")
    save_result(resized_150, "resized_150.png")
    save_result(resized_fixed, "resized_fixed_width.png")

    print("\n==============================================")
    print("Задание 5. Поворот изображения")
    print("==============================================")
    try:
        rotated_45 = rotate_image(image, 45, expand=True)
        rotated_90 = rotate_image(image, 90, expand=True)
        rotated_custom = rotate_image(image, 120, expand=True)
        rotated_45_no_expand = rotate_image(image, 45, expand=False)
    except ValueError as exc:
        print("Ошибка поворота:", exc)
        rotated_45 = rotated_90 = rotated_custom = rotated_45_no_expand = image
    print("45° expand=True:", rotated_45.size, "| expand=False:", rotated_45_no_expand.size)
    save_result(rotated_45, "rotated_45.png")
    save_result(rotated_90, "rotated_90.png")
    save_result(rotated_custom, "rotated_120.png")
    save_result(rotated_45_no_expand, "rotated_45_no_expand.png")

    print("\n==============================================")
    print("Задание 6. Оттенки серого")
    print("==============================================")
    gray_array = to_grayscale_numpy(array)
    print("gray shape:", gray_array.shape, "| dtype:", gray_array.dtype)
    save_result(gray_array, "gray.png")

    print("\n==============================================")
    print("Задание 7. Обработка области изображения")
    print("==============================================")
    try:
        local_array = process_region(array, h // 4, h // 4 * 3, w // 4, w // 4 * 3)
    except ValueError as exc:
        print("Ошибка обработки области:", exc)
        local_array = array
    save_result(local_array, "local_transform.png")

    print("\n==============================================")
    print("Задание 8. Итоговая композиция")
    print("==============================================")
    results = [
        (image, "Исходное изображение"),
        (resized_50, "Уменьшенное (50%)"),
        (rotated_45, "Повёрнутое (45°)"),
        (gray_array, "Grayscale"),
        (brighter, "Яркость +60"),
        (local_array, "Локальная обработка области"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, (img, title) in zip(axes.flat, results):
        if isinstance(img, np.ndarray):
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        else:
            ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "summary.png", dpi=150, bbox_inches="tight")
    print("Композиция сохранена в:", OUTPUT_DIR / "summary.png")


if __name__ == "__main__":
    main()