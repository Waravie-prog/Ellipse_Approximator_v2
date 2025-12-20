"""
ГЕНЕРАТОР ТЕСТОВЫХ ИЗОБРАЖЕНИЙ С КРУГЛЫМИ ПОРАМИ
Создает 3 типа изображений с разной степенью пересечения пор + новые конфигурации с 3-4 порами
"""

import numpy as np
from skimage import draw, io, morphology
import matplotlib.pyplot as plt
import os
import datetime

def create_touching_pores():
    """Создает изображение с двумя порами, касающимися в одной точке"""
    print("Создаю изображение с касающимися порами...")
    
    # Создаем черное изображение большего размера для лучшей видимости
    image = np.zeros((400, 400), dtype=np.uint8)
    
    # Параметры первой поры (круга)
    center1_y, center1_x = 200, 150
    radius1 = 60
    
    # Вторая пора касается первой в одной точке (расстояние = 2 * radius)
    center2_y, center2_x = 200, 270  # 150 + 2*60 = 270
    radius2 = 60
    
    # Рисуем первую пору
    rr, cc = draw.disk((center1_y, center1_x), radius1, shape=image.shape)
    image[rr, cc] = 255
    
    # Рисуем вторую пору
    rr, cc = draw.disk((center2_y, center2_x), radius2, shape=image.shape)
    image[rr, cc] = 255
    
    # Сохраняем изображение
    io.imsave('touching_pores.png', image)
    print("✓ Создано изображение: 'touching_pores.png'")
    
    # Создаем визуализацию
    plt.figure(figsize=(10, 8))
    plt.imshow(image, cmap='gray')
    plt.title('Две поры, касающиеся в одной точке\n(расстояние между центрами = 120 пикселей)', 
              fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # Добавляем информацию о параметрах
    distance = np.sqrt((center2_x - center1_x)**2 + (center2_y - center1_y)**2)
    plt.text(10, 380, f'Расстояние между центрами: {distance:.1f} пикселей', 
             fontsize=12, color='red', weight='bold', backgroundcolor='white')
    plt.text(10, 395, f'Радиус каждой поры: {radius1} пикселей', 
             fontsize=12, color='red', weight='bold', backgroundcolor='white')
    
    plt.savefig('touching_pores_preview.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return image

def create_slightly_overlapping_pores():
    """Создает изображение с двумя порами, немного пересекающимися"""
    print("Создаю изображение с немного пересекающимися порами...")
    
    image = np.zeros((400, 400), dtype=np.uint8)
    
    # Параметры первой поры
    center1_y, center1_x = 200, 150
    radius1 = 60
    
    # Вторая пора немного пересекается с первой (расстояние меньше суммы радиусов)
    center2_y, center2_x = 200, 250  # Расстояние 100 пикселей (120 - 20)
    radius2 = 60
    
    # Рисуем поры
    rr, cc = draw.disk((center1_y, center1_x), radius1, shape=image.shape)
    image[rr, cc] = 255
    
    rr, cc = draw.disk((center2_y, center2_x), radius2, shape=image.shape)
    image[rr, cc] = 255
    
    io.imsave('slightly_overlapping_pores.png', image)
    print("✓ Создано изображение: 'slightly_overlapping_pores.png'")
    
    # Визуализация
    plt.figure(figsize=(10, 8))
    plt.imshow(image, cmap='gray')
    plt.title('Две поры, немного пересекающиеся\n(расстояние между центрами = 100 пикселей)', 
              fontsize=14, fontweight='bold')
    plt.axis('off')
    
    distance = np.sqrt((center2_x - center1_x)**2 + (center2_y - center1_y)**2)
    overlap = 2 * radius1 - distance
    plt.text(10, 380, f'Расстояние между центрами: {distance:.1f} пикселей', 
             fontsize=12, color='red', weight='bold', backgroundcolor='white')
    plt.text(10, 395, f'Перекрытие: {overlap:.1f} пикселей', 
             fontsize=12, color='red', weight='bold', backgroundcolor='white')
    
    plt.savefig('slightly_overlapping_pores_preview.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return image

def create_highly_overlapping_pores():
    """Создает изображение с двумя порами, сильно пересекающимися"""
    print("Создаю изображение с сильно пересекающимися порами...")
    
    image = np.zeros((400, 400), dtype=np.uint8)
    
    # Параметры первой поры
    center1_y, center1_x = 200, 150
    radius1 = 60
    
    # Вторая пора сильно пересекается с первой
    center2_y, center2_x = 200, 210  # Расстояние 60 пикселей
    radius2 = 60
    
    # Рисуем поры
    rr, cc = draw.disk((center1_y, center1_x), radius1, shape=image.shape)
    image[rr, cc] = 255
    
    rr, cc = draw.disk((center2_y, center2_x), radius2, shape=image.shape)
    image[rr, cc] = 255
    
    io.imsave('highly_overlapping_pores.png', image)
    print("✓ Создано изображение: 'highly_overlapping_pores.png'")
    
    # Визуализация
    plt.figure(figsize=(10, 8))
    plt.imshow(image, cmap='gray')
    plt.title('Две поры, сильно пересекающиеся\n(расстояние между центрами = 60 пикселей)', 
              fontsize=14, fontweight='bold')
    plt.axis('off')
    
    distance = np.sqrt((center2_x - center1_x)**2 + (center2_y - center1_y)**2)
    overlap = 2 * radius1 - distance
    plt.text(10, 380, f'Расстояние между центрами: {distance:.1f} пикселей', 
             fontsize=12, color='red', weight='bold', backgroundcolor='white')
    plt.text(10, 395, f'Перекрытие: {overlap:.1f} пикселей', 
             fontsize=12, color='red', weight='bold', backgroundcolor='white')
    
    plt.savefig('highly_overlapping_pores_preview.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return image

def create_single_pore():
    """Создает изображение с одной порой для тестирования"""
    print("Создаю изображение с одной порой...")
    
    image = np.zeros((400, 400), dtype=np.uint8)
    
    center_y, center_x = 200, 200
    radius = 80
    
    rr, cc = draw.disk((center_y, center_x), radius, shape=image.shape)
    image[rr, cc] = 255
    
    io.imsave('single_pore.png', image)
    print("✓ Создано изображение: 'single_pore.png'")
    
    return image

def create_three_touching_pores():
    """Создает изображение с тремя порами, касающимися друг друга"""
    print("Создаю изображение с тремя касающимися порами...")
    
    image = np.zeros((400, 400), dtype=np.uint8)
    
    # Параметры пор (расположены в вершинах равностороннего треугольника)
    radius = 50
    
    # Центры пор: каждая пара касается в одной точке
    center1_y, center1_x = 200, 150    # Верхняя пора
    center2_y, center2_x = 250, 235    # Правая нижняя пора  
    center3_y, center3_x = 150, 235    # Левая нижняя пора
    
    # Рисуем первую пору
    rr, cc = draw.disk((center1_y, center1_x), radius, shape=image.shape)
    image[rr, cc] = 255
    
    # Рисуем вторую пору
    rr, cc = draw.disk((center2_y, center2_x), radius, shape=image.shape)
    image[rr, cc] = 255
    
    # Рисуем третью пору
    rr, cc = draw.disk((center3_y, center3_x), radius, shape=image.shape)
    image[rr, cc] = 255
    
    io.imsave('three_touching_pores.png', image)
    print("✓ Создано изображение: 'three_touching_pores.png'")
    
    # Визуализация
    plt.figure(figsize=(10, 8))
    plt.imshow(image, cmap='gray')
    plt.title('Три поры, касающиеся друг друга\n(расстояние между центрами = 100 пикселей)', 
              fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # Расчет расстояний между центрами
    dist12 = np.sqrt((center2_x - center1_x)**2 + (center2_y - center1_y)**2)
    dist13 = np.sqrt((center3_x - center1_x)**2 + (center3_y - center1_y)**2)
    dist23 = np.sqrt((center3_x - center2_x)**2 + (center3_y - center2_y)**2)
    
    plt.text(10, 370, f'Расстояние между центрами 1-2: {dist12:.1f} пикселей', 
             fontsize=12, color='red', weight='bold', backgroundcolor='white')
    plt.text(10, 385, f'Расстояние между центрами 1-3: {dist13:.1f} пикселей', 
             fontsize=12, color='red', weight='bold', backgroundcolor='white')
    plt.text(10, 400, f'Расстояние между центрами 2-3: {dist23:.1f} пикселей', 
             fontsize=12, color='red', weight='bold', backgroundcolor='white')
    
    plt.savefig('three_touching_pores_preview.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return image

def create_three_slightly_overlapping_pores():
    """Создает изображение с тремя порами, немного пересекающимися"""
    print("Создаю изображение с тремя немного пересекающимися порами...")
    
    image = np.zeros((400, 400), dtype=np.uint8)
    
    radius = 55
    
    # Центры расположены ближе, чем 2*radius, для небольшого перекрытия
    center1_y, center1_x = 200, 140    # Верхняя пора
    center2_y, center2_x = 260, 220    # Правая нижняя пора  
    center3_y, center3_x = 140, 220    # Левая нижняя пора
    
    rr, cc = draw.disk((center1_y, center1_x), radius, shape=image.shape)
    image[rr, cc] = 255
    
    rr, cc = draw.disk((center2_y, center2_x), radius, shape=image.shape)
    image[rr, cc] = 255
    
    rr, cc = draw.disk((center3_y, center3_x), radius, shape=image.shape)
    image[rr, cc] = 255
    
    io.imsave('three_slightly_overlapping_pores.png', image)
    print("✓ Создано изображение: 'three_slightly_overlapping_pores.png'")
    
    # Визуализация
    plt.figure(figsize=(10, 8))
    plt.imshow(image, cmap='gray')
    plt.title('Три поры, немного пересекающиеся\n(расстояние между центрами ~90 пикселей)', 
              fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # Расчет расстояний и перекрытий
    dist12 = np.sqrt((center2_x - center1_x)**2 + (center2_y - center1_y)**2)
    dist13 = np.sqrt((center3_x - center1_x)**2 + (center3_y - center1_y)**2)
    dist23 = np.sqrt((center3_x - center2_x)**2 + (center3_y - center2_y)**2)
    
    overlap12 = 2 * radius - dist12
    overlap13 = 2 * radius - dist13
    overlap23 = 2 * radius - dist23
    
    plt.text(10, 370, f'Расстояние 1-2: {dist12:.1f} пикселей', 
             fontsize=12, color='red', weight='bold', backgroundcolor='white')
    plt.text(10, 385, f'Расстояние 1-3: {dist13:.1f} пикселей', 
             fontsize=12, color='red', weight='bold', backgroundcolor='white')
    plt.text(10, 400, f'Расстояние 2-3: {dist23:.1f} пикселей', 
             fontsize=12, color='red', weight='bold', backgroundcolor='white')
    plt.text(250, 385, f'Перекрытие: ~{overlap12:.1f} пикселей', 
             fontsize=12, color='blue', weight='bold', backgroundcolor='white')
    
    plt.savefig('three_slightly_overlapping_pores_preview.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return image

def create_three_highly_overlapping_pores():
    """Создает изображение с тремя порами, сильно пересекающимися"""
    print("Создаю изображение с тремя сильно пересекающимися порами...")
    
    image = np.zeros((400, 400), dtype=np.uint8)
    
    radius = 70
    
    # Центры расположены очень близко для сильного перекрытия
    center1_y, center1_x = 200, 180    # Верхняя пора
    center2_y, center2_x = 230, 220    # Правая нижняя пора  
    center3_y, center3_x = 170, 220    # Левая нижняя пора
    
    rr, cc = draw.disk((center1_y, center1_x), radius, shape=image.shape)
    image[rr, cc] = 255
    
    rr, cc = draw.disk((center2_y, center2_x), radius, shape=image.shape)
    image[rr, cc] = 255
    
    rr, cc = draw.disk((center3_y, center3_x), radius, shape=image.shape)
    image[rr, cc] = 255
    
    io.imsave('three_highly_overlapping_pores.png', image)
    print("✓ Создано изображение: 'three_highly_overlapping_pores.png'")
    
    # Визуализация
    plt.figure(figsize=(10, 8))
    plt.imshow(image, cmap='gray')
    plt.title('Три поры, сильно пересекающиеся\n(расстояние между центрами ~50-60 пикселей)', 
              fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # Расчет расстояний и перекрытий
    dist12 = np.sqrt((center2_x - center1_x)**2 + (center2_y - center1_y)**2)
    dist13 = np.sqrt((center3_x - center1_x)**2 + (center3_y - center1_y)**2)
    dist23 = np.sqrt((center3_x - center2_x)**2 + (center3_y - center2_y)**2)
    
    overlap12 = 2 * radius - dist12
    overlap13 = 2 * radius - dist13
    overlap23 = 2 * radius - dist23
    
    plt.text(10, 370, f'Расстояние 1-2: {dist12:.1f} пикселей', 
             fontsize=12, color='red', weight='bold', backgroundcolor='white')
    plt.text(10, 385, f'Расстояние 1-3: {dist13:.1f} пикселей', 
             fontsize=12, color='red', weight='bold', backgroundcolor='white')
    plt.text(10, 400, f'Расстояние 2-3: {dist23:.1f} пикселей', 
             fontsize=12, color='red', weight='bold', backgroundcolor='white')
    plt.text(250, 385, f'Перекрытие: ~{overlap12:.1f} пикселей', 
             fontsize=12, color='blue', weight='bold', backgroundcolor='white')
    
    plt.savefig('three_highly_overlapping_pores_preview.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return image

def create_four_slightly_overlapping_pores():
    """Создает изображение с четырьмя порами, немного пересекающимися"""
    print("Создаю изображение с четырьмя немного пересекающимися порами...")
    
    image = np.zeros((400, 400), dtype=np.uint8)
    
    radius = 45
    
    # Центры расположены в вершинах квадрата с небольшим перекрытием
    center1_y, center1_x = 160, 160    # Верхний-левый
    center2_y, center2_x = 160, 240    # Верхний-правый
    center3_y, center3_x = 240, 160    # Нижний-левый
    center4_y, center4_x = 240, 240    # Нижний-правый
    
    rr, cc = draw.disk((center1_y, center1_x), radius, shape=image.shape)
    image[rr, cc] = 255
    
    rr, cc = draw.disk((center2_y, center2_x), radius, shape=image.shape)
    image[rr, cc] = 255
    
    rr, cc = draw.disk((center3_y, center3_x), radius, shape=image.shape)
    image[rr, cc] = 255
    
    rr, cc = draw.disk((center4_y, center4_x), radius, shape=image.shape)
    image[rr, cc] = 255
    
    io.imsave('four_slightly_overlapping_pores.png', image)
    print("✓ Создано изображение: 'four_slightly_overlapping_pores.png'")
    
    # Визуализация
    plt.figure(figsize=(10, 8))
    plt.imshow(image, cmap='gray')
    plt.title('Четыре поры, немного пересекающиеся\n(расстояние между центрами = 80 пикселей)', 
              fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # Расчет расстояний
    dist_horizontal = np.sqrt((center2_x - center1_x)**2 + (center2_y - center1_y)**2)
    dist_vertical = np.sqrt((center3_x - center1_x)**2 + (center3_y - center1_y)**2)
    dist_diagonal = np.sqrt((center4_x - center1_x)**2 + (center4_y - center1_y)**2)
    
    overlap_horizontal = 2 * radius - dist_horizontal
    
    plt.text(10, 370, f'Расстояние по горизонтали: {dist_horizontal:.1f} пикселей', 
             fontsize=12, color='red', weight='bold', backgroundcolor='white')
    plt.text(10, 385, f'Расстояние по вертикали: {dist_vertical:.1f} пикселей', 
             fontsize=12, color='red', weight='bold', backgroundcolor='white')
    plt.text(10, 400, f'Расстояние по диагонали: {dist_diagonal:.1f} пикселей', 
             fontsize=12, color='red', weight='bold', backgroundcolor='white')
    plt.text(250, 385, f'Перекрытие: ~{overlap_horizontal:.1f} пикселей', 
             fontsize=12, color='blue', weight='bold', backgroundcolor='white')
    
    plt.savefig('four_slightly_overlapping_pores_preview.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return image

def create_four_highly_overlapping_pores():
    """Создает изображение с четырьмя порами, сильно пересекающимися"""
    print("Создаю изображение с четырьмя сильно пересекающимися порами...")
    
    image = np.zeros((400, 400), dtype=np.uint8)
    
    radius = 60
    
    # Центры расположены очень близко для сильного перекрытия
    center1_y, center1_x = 180, 180    # Верхний-левый
    center2_y, center2_x = 180, 220    # Верхний-правый
    center3_y, center3_x = 220, 180    # Нижний-левый
    center4_y, center4_x = 220, 220    # Нижний-правый
    
    rr, cc = draw.disk((center1_y, center1_x), radius, shape=image.shape)
    image[rr, cc] = 255
    
    rr, cc = draw.disk((center2_y, center2_x), radius, shape=image.shape)
    image[rr, cc] = 255
    
    rr, cc = draw.disk((center3_y, center3_x), radius, shape=image.shape)
    image[rr, cc] = 255
    
    rr, cc = draw.disk((center4_y, center4_x), radius, shape=image.shape)
    image[rr, cc] = 255
    
    io.imsave('four_highly_overlapping_pores.png', image)
    print("✓ Создано изображение: 'four_highly_overlapping_pores.png'")
    
    # Визуализация
    plt.figure(figsize=(10, 8))
    plt.imshow(image, cmap='gray')
    plt.title('Четыре поры, сильно пересекающиеся\n(расстояние между центрами = 40 пикселей)', 
              fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # Расчет расстояний и перекрытий
    dist_horizontal = np.sqrt((center2_x - center1_x)**2 + (center2_y - center1_y)**2)
    dist_vertical = np.sqrt((center3_x - center1_x)**2 + (center3_y - center1_y)**2)
    
    overlap_horizontal = 2 * radius - dist_horizontal
    overlap_vertical = 2 * radius - dist_vertical
    
    plt.text(10, 370, f'Расстояние по горизонтали: {dist_horizontal:.1f} пикселей', 
             fontsize=12, color='red', weight='bold', backgroundcolor='white')
    plt.text(10, 385, f'Расстояние по вертикали: {dist_vertical:.1f} пикселей', 
             fontsize=12, color='red', weight='bold', backgroundcolor='white')
    plt.text(10, 400, f'Перекрытие по горизонтали: {overlap_horizontal:.1f} пикселей', 
             fontsize=12, color='blue', weight='bold', backgroundcolor='white')
    plt.text(250, 400, f'Перекрытие по вертикали: {overlap_vertical:.1f} пикселей', 
             fontsize=12, color='blue', weight='bold', backgroundcolor='white')
    
    plt.savefig('four_highly_overlapping_pores_preview.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return image

def analyze_created_masks():
    """Анализирует созданные изображения и выводит информацию"""
    from skimage import measure
    
    masks = [
        'single_pore.png', 
        'touching_pores.png', 
        'slightly_overlapping_pores.png', 
        'highly_overlapping_pores.png',
        'three_touching_pores.png',
        'three_slightly_overlapping_pores.png',
        'three_highly_overlapping_pores.png',
        'four_slightly_overlapping_pores.png',
        'four_highly_overlapping_pores.png'
    ]
    
    print("\n" + "="*60)
    print("АНАЛИЗ СОЗДАННЫХ ИЗОБРАЖЕНИЙ")
    print("="*60)
    
    for mask_file in masks:
        if os.path.exists(mask_file):
            try:
                image = io.imread(mask_file)
                if len(image.shape) == 3:
                    image = image[:, :, 0]
                
                binary = image > 128
                labeled = measure.label(binary)
                regions = measure.regionprops(labeled)
                
                if regions:
                    print(f"\n📊 {mask_file}:")
                    print(f"   Размер изображения: {image.shape}")
                    
                    if len(regions) == 1:
                        region = regions[0]
                        print(f"   Обнаружена 1 пора")
                        print(f"   Площадь: {region.area} пикселей")
                        print(f"   Эквивалентный диаметр: {region.equivalent_diameter:.1f} пикселей")
                    else:
                        print(f"   Обнаружено {len(regions)} отдельных региона")
                        total_area = sum(region.area for region in regions)
                        print(f"   Общая площадь: {total_area} пикселей")
                        
                        for i, region in enumerate(regions, 1):
                            print(f"   Пора {i}: площадь = {region.area} пикселей, "
                                  f"диаметр = {region.equivalent_diameter:.1f} пикселей")
                        
            except Exception as e:
                print(f"❌ Ошибка при анализе {mask_file}: {e}")

def main():
    """Основная функция"""
    print("=" * 70)
    print("🎯 ГЕНЕРАТОР ТЕСТОВЫХ ИЗОБРАЖЕНИЙ С КРУГЛЫМИ ПОРАМИ")
    print("=" * 70)
    
    try:
        # Создаем тестовые изображения
        print("\n🚀 Начинаю создание тестовых изображений...")
        
        create_single_pore()
        create_touching_pores()
        create_slightly_overlapping_pores()
        create_highly_overlapping_pores()
        
        # Новые изображения
        create_three_touching_pores()
        create_three_slightly_overlapping_pores()
        create_three_highly_overlapping_pores()
        create_four_slightly_overlapping_pores()
        create_four_highly_overlapping_pores()
        
        # Анализируем созданные изображения
        analyze_created_masks()
        
        print("\n" + "=" * 70)
        print("✅ ВСЕ ИЗОБРАЖЕНИЯ УСПЕШНО СОЗДАНЫ!")
        print("=" * 70)
        print("\n📁 СОЗДАННЫЕ ФАЙЛЫ:")
        print("   • single_pore.png              - одна круглая пора")
        print("   • touching_pores.png           - две касающиеся поры")
        print("   • slightly_overlapping_pores.png - две немного пересекающиеся поры")
        print("   • highly_overlapping_pores.png - две сильно пересекающиеся поры")
        print("   • three_touching_pores.png     - три касающиеся поры")
        print("   • three_slightly_overlapping_pores.png - три немного пересекающиеся поры")
        print("   • three_highly_overlapping_pores.png - три сильно пересекающиеся поры")
        print("   • four_slightly_overlapping_pores.png - четыре немного пересекающиеся поры")
        print("   • four_highly_overlapping_pores.png - четыре сильно пересекающиеся поры")
        
        print("\n   • *_preview.png               - превью с параметрами")
        
        print("\n🎯 ДАЛЬНЕЙШИЕ ДЕЙСТВИЯ:")
        print("   Теперь запустите: python circle_approximator.py")
        print("   для тестирования алгоритма аппроксимации на этих изображениях")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ ПРОИЗОШЛА ОШИБКА: {e}")
        print("\n🔧 УСТАНОВИТЕ ЗАВИСИМОСТИ:")
        print("   pip install numpy scikit-image matplotlib")
        print("=" * 70)

if __name__ == "__main__":
    main()