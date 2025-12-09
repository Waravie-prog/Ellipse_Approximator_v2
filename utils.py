"""
Вспомогательные функции для анализа и визуализации
"""

import numpy as np
from skimage import measure, io
import matplotlib.pyplot as plt
import json

def analyze_blob_properties(image_path):
    """Анализ свойств blob-объекта"""
    image = io.imread(image_path)
    if len(image.shape) == 3:
        image = image[:, :, 0]
    
    binary = image > 128
    labeled = measure.label(binary)
    regions = measure.regionprops(labeled)
    
    if regions:
        largest = max(regions, key=lambda x: x.area)
        
        print("\n" + "=" * 40)
        print("АНАЛИЗ BLOB-ОБЪЕКТА")
        print("=" * 40)
        print(f"Файл: {image_path}")
        print(f"Размер изображения: {image.shape}")
        print(f"Площадь объекта: {largest.area} пикселей")
        print(f"Периметр: {largest.perimeter:.2f}")
        print(f"Эксцентриситет: {largest.eccentricity:.3f}")
        print(f"Отношение сторон: {largest.major_axis_length/largest.minor_axis_length:.3f}")
        print(f"Ориентация: {largest.orientation:.3f} радиан")
        print(f"Bounding box: {largest.bbox}")
        
        # Рекомендация по количеству эллипсов
        recommended_ellipses = max(2, min(8, int(largest.area / 500)))
        print(f"Рекомендуемое количество эллипсов: {recommended_ellipses}")
        print("=" * 40)
        
        return largest.area, recommended_ellipses
    return 0, 0

def plot_ellipse_parameters(json_file):
    """Визуализация параметров эллипсов из JSON файла"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    ellipses = data['ellipses']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Размеры эллипсов
    sizes_a = [e['semi_axes']['a'] for e in ellipses]
    sizes_b = [e['semi_axes']['b'] for e in ellipses]
    ids = [e['id'] for e in ellipses]
    
    axes[0, 0].bar(ids, sizes_a, alpha=0.7, label='Большая ось (a)')
    axes[0, 0].bar(ids, sizes_b, alpha=0.7, label='Малая ось (b)')
    axes[0, 0].set_xlabel('ID эллипса')
    axes[0, 0].set_ylabel('Размер осей')
    axes[0, 0].set_title('Размеры полуосей эллипсов')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Эксцентриситет
    eccentricities = [e['eccentricity'] for e in ellipses]
    axes[0, 1].bar(ids, eccentricities, alpha=0.7, color='green')
    axes[0, 1].set_xlabel('ID эллипса')
    axes[0, 1].set_ylabel('Эксцентриситет')
    axes[0, 1].set_title('Эксцентриситет эллипсов')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Углы поворота
    angles = [e['angle_deg'] for e in ellipses]
    axes[1, 0].scatter(ids, angles, s=100, alpha=0.7)
    axes[1, 0].set_xlabel('ID эллипса')
    axes[1, 0].set_ylabel('Угол поворота (градусы)')
    axes[1, 0].set_title('Углы поворота эллипсов')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Распределение центров
    centers_x = [e['center']['x'] for e in ellipses]
    centers_y = [e['center']['y'] for e in ellipses]
    axes[1, 1].scatter(centers_x, centers_y, s=100, alpha=0.7)
    for i, (x, y) in enumerate(zip(centers_x, centers_y)):
        axes[1, 1].annot(f'E{i+1}', (x, y), xytext=(5, 5), textcoords='offset points')
    axes[1, 1].set_xlabel('X координата')
    axes[1, 1].set_ylabel('Y координата')
    axes[1, 1].set_title('Расположение центров эллипсов')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Анализ параметров эллипсов ({len(ellipses)} эллипсов)', fontsize=14)
    plt.tight_layout()
    plt.savefig('ellipse_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Пример использования
    analyze_blob_properties('complex_blob.png')