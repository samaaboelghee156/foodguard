from io import BytesIO
import os

import numpy as np
from PIL import Image, UnidentifiedImageError
from transformers import pipeline

from ..schemas import AnalyzeFoodResponse, FoodPrediction


class FoodClassifier:
    def __init__(self) -> None:
        self.model_name = os.getenv("FOOD_MODEL_NAME", "nateraw/food")
        self._pipeline = pipeline(
            task="image-classification",
            model=self.model_name,
            framework="pt",
            top_k=5,
        )

    def analyze(self, image_bytes: bytes) -> AnalyzeFoodResponse:
        image = self._load_image(image_bytes)
        raw_predictions = self._pipeline(image)
        predictions = [
            FoodPrediction(
                label=str(item["label"]).replace("_", " ").lower(),
                confidence=float(item["score"]),
            )
            for item in raw_predictions
        ]
        if not predictions:
            raise ValueError("The model did not return any food predictions.")

        return AnalyzeFoodResponse(
            food_name=predictions[0].label,
            confidence=predictions[0].confidence,
            top_predictions=predictions,
            visual_warnings=self._detect_visual_warnings(image, predictions[0].label),
        )

    def _load_image(self, image_bytes: bytes) -> Image.Image:
        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except UnidentifiedImageError as exc:
            raise ValueError("Could not read the uploaded image.") from exc
        return image

    def _detect_visual_warnings(self, image: Image.Image, food_name: str) -> list[str]:
        thumbnail = image.copy()
        thumbnail.thumbnail((384, 384))
        pixels = np.asarray(thumbnail).astype(np.float32)
        red = pixels[:, :, 0]
        green = pixels[:, :, 1]
        blue = pixels[:, :, 2]
        brightness = (red + green + blue) / 3
        height, width = brightness.shape

        green_mask = (green > 85) & (green > red * 1.25) & (green > blue * 1.12)
        blue_mask = (blue > 70) & (green > 55) & (red < 120) & (blue > red * 1.12)
        dark_mask = (red < 55) & (green < 55) & (blue < 55)
        gray_rot_mask = (brightness > 55) & (brightness < 150) & (np.abs(red - green) < 18) & (np.abs(green - blue) < 18)
        white_fuzzy_mask = (brightness > 185) & (np.std(pixels, axis=2) < 18)
        strict_food = self._needs_strict_mold_check(food_name)
        mold_relevant_food = self._mold_is_common_for_food(food_name)

        # Uniform green foods should not be called mold. Mold-like green is usually clustered
        # into isolated patches with surrounding contrast, not spread across most of the food.
        green_ratio = float(np.count_nonzero(green_mask) / green_mask.size)
        uniform_green_surface = green_ratio > 0.28 and self._grid_coverage(green_mask, 6) > 0.55

        warnings: list[str] = []
        if strict_food:
            # Burgers, sandwiches, salads, pizza, wraps, tacos, etc. often contain
            # lettuce, pickles, herbs, sauces, cheese, toasted bread, grill marks,
            # and shadows. For these foods, color alone is too noisy, so only
            # strong blue/gray colony-like clusters can trigger a spoilage warning.
            if self._has_suspicious_patch(blue_mask, brightness, width, height, min_ratio=0.025, max_ratio=0.10, cell_threshold=0.28, contrast_threshold=22):
                warnings.append("Possible mold/spoilage detected. Do not eat.")
            elif self._has_suspicious_patch(gray_rot_mask, brightness, width, height, min_ratio=0.045, max_ratio=0.12, cell_threshold=0.34, contrast_threshold=24):
                warnings.append("Possible mold/spoilage detected. Do not eat.")
            return warnings

        if mold_relevant_food and not uniform_green_surface and self._has_suspicious_patch(green_mask, brightness, width, height, min_ratio=0.016, max_ratio=0.18, cell_threshold=0.20, contrast_threshold=18):
            warnings.append("Possible mold/spoilage detected. Do not eat.")
        elif self._has_suspicious_patch(blue_mask, brightness, width, height, min_ratio=0.014, max_ratio=0.16, cell_threshold=0.20, contrast_threshold=18):
            warnings.append("Possible mold/spoilage detected. Do not eat.")
        elif mold_relevant_food and self._has_suspicious_patch(dark_mask, brightness, width, height, min_ratio=0.012, max_ratio=0.16, cell_threshold=0.22, contrast_threshold=22):
            warnings.append("Possible mold/spoilage detected. Do not eat.")
        elif mold_relevant_food and self._has_suspicious_patch(gray_rot_mask, brightness, width, height, min_ratio=0.03, max_ratio=0.18, cell_threshold=0.26, contrast_threshold=20):
            warnings.append("Possible mold/spoilage detected. Do not eat.")
        elif mold_relevant_food and self._has_suspicious_patch(white_fuzzy_mask, brightness, width, height, min_ratio=0.024, max_ratio=0.16, cell_threshold=0.25, contrast_threshold=16):
            warnings.append("Possible mold/spoilage detected. Do not eat.")
        return warnings

    def _needs_strict_mold_check(self, food_name: str) -> bool:
        normalized = food_name.lower()
        return any(
            item in normalized
            for item in [
                "burger",
                "hamburger",
                "cheeseburger",
                "sandwich",
                "salad",
                "pizza",
                "wrap",
                "taco",
                "burrito",
                "hot dog",
                "nachos",
                "quesadilla",
                "gyro",
                "kebab",
            ]
        )

    def _mold_is_common_for_food(self, food_name: str) -> bool:
        normalized = food_name.lower()
        return any(
            item in normalized
            for item in [
                "bread",
                "toast",
                "fruit",
                "apple",
                "banana",
                "berry",
                "strawberry",
                "grape",
                "orange",
                "cheese",
                "rice",
                "leftover",
                "pastry",
                "cake",
                "muffin",
                "donut",
                "pancake",
                "waffle",
                "pie",
            ]
        )

    def _grid_coverage(self, mask: np.ndarray, grid_size: int) -> float:
        covered = 0
        total = grid_size * grid_size
        height, width = mask.shape
        for row in range(grid_size):
            for col in range(grid_size):
                y0 = row * height // grid_size
                y1 = (row + 1) * height // grid_size
                x0 = col * width // grid_size
                x1 = (col + 1) * width // grid_size
                if np.count_nonzero(mask[y0:y1, x0:x1]) / max(1, mask[y0:y1, x0:x1].size) > 0.04:
                    covered += 1
        return covered / total

    def _has_suspicious_patch(
        self,
        mask: np.ndarray,
        brightness: np.ndarray,
        width: int,
        height: int,
        min_ratio: float = 0.012,
        max_ratio: float = 0.35,
        cell_threshold: float = 0.16,
        contrast_threshold: float = 12,
    ) -> bool:
        total_ratio = float(np.count_nonzero(mask) / mask.size)
        if total_ratio < min_ratio or total_ratio > max_ratio:
            return False

        grid_size = 8
        for row in range(grid_size):
            for col in range(grid_size):
                y0 = row * height // grid_size
                y1 = (row + 1) * height // grid_size
                x0 = col * width // grid_size
                x1 = (col + 1) * width // grid_size
                cell = mask[y0:y1, x0:x1]
                cell_ratio = float(np.count_nonzero(cell) / max(1, cell.size))
                if cell_ratio < cell_threshold:
                    continue

                y_pad0 = max(0, y0 - (y1 - y0))
                y_pad1 = min(height, y1 + (y1 - y0))
                x_pad0 = max(0, x0 - (x1 - x0))
                x_pad1 = min(width, x1 + (x1 - x0))
                cell_brightness = float(np.mean(brightness[y0:y1, x0:x1]))
                area_brightness = float(np.mean(brightness[y_pad0:y_pad1, x_pad0:x_pad1]))
                if abs(cell_brightness - area_brightness) > contrast_threshold or total_ratio < min_ratio * 4:
                    return True
        return False
