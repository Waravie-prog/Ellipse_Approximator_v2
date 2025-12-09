"""
Генератор тестовых изображений с круглыми порами
Создает 3 типа изображений с разной степенью пересечения пор
"""

import numpy as np
from skimage import draw, io, morphology
import matplotlib.pyplot as plt
import os

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

def analyze_created_masks():
    """Анализирует созданные изображения и выводит информацию"""
    from skimage import measure
    
    masks = ['touching_pores.png', 'slightly_overlapping_pores.png', 
             'highly_overlapping_pores.png', 'single_pore.png']
    
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
        print("\n   • *_preview.png               - превью с параметрами")
        
        print("\n🎯 ДАЛЬНЕЙШИЕ ДЕЙСТВИЯ:")
        print("   Теперь запустите: python circle_approximator.py")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ ПРОИЗОШЛА ОШИБКА: {e}")
        print("\n🔧 УСТАНОВИТЕ ЗАВИСИМОСТИ:")
        print("   pip install numpy scikit-image matplotlib")
        print("=" * 70)

if __name__ == "__main__":
    main()