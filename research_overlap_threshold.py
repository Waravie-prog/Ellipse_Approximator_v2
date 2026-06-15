"""
ИССЛЕДОВАНИЕ ПОРОГА СЛИЯНИЯ ПОР
С АВТОМАТИЧЕСКИМ ОПРЕДЕЛЕНИЕМ КОЛИЧЕСТВА ОКРУЖНОСТЕЙ
ИСПРАВЛЕННАЯ ВЕРСИЯ
"""
import numpy as np
from skimage import draw, io, measure
import matplotlib.pyplot as plt
import pandas as pd
import os
import datetime
from circle_approximator import CircleGeneticApproximator


def analyze_blob_complexity(binary_mask):
    """
    Анализирует сложность blob-объекта и определяет количество окружностей
    """
    # ИСПРАВЛЕНИЕ: measure.label() возвращает кортеж в новых версиях skimage
    labeled, num_regions = measure.label(binary_mask.astype(int), return_num=True)
    regions = measure.regionprops(labeled)
    
    if not regions:
        return 1, {'error': 'No regions found'}
    
    region = regions[0]
    area = region.area
    perimeter = region.perimeter
    compactness = (perimeter ** 2) / (4 * np.pi * area) if area > 0 else 1
    eccentricity = region.eccentricity
    solidity = region.solidity
    
    print(f"   Геометрические характеристики:")
    print(f"    Площадь: {area} пикселей")
    print(f"    Компактность: {compactness:.3f}")
    print(f"    Эксцентриситет: {eccentricity:.3f}")
    print(f"    Сплошность (solidity): {solidity:.3f}")
    
    # Определение количества окружностей на основе компактности
    if compactness < 1.3:
        recommended_n = 1
        print(f"   Рекомендовано окружностей: {recommended_n} (простая форма)")
    elif compactness < 1.8:
        recommended_n = 2
        print(f"   Рекомендовано окружностей: {recommended_n} (умеренно сложная)")
    else:
        recommended_n = 2
        print(f"   Рекомендовано окружностей: {recommended_n} (сложная форма)")
    
    metrics = {
        'area': area,
        'compactness': compactness,
        'eccentricity': eccentricity,
        'solidity': solidity,
        'recommended_n': recommended_n
    }
    
    return recommended_n, metrics


def create_two_pores_with_distance(distance, radius=60, image_size=(400, 400)):
    """Создает изображение с двумя порами на заданном расстоянии"""
    image = np.zeros(image_size, dtype=np.uint8)
    
    center1_y, center1_x = image_size[0] // 2, image_size[1] // 3
    center2_y, center2_x = image_size[0] // 2, center1_x + distance
    
    rr, cc = draw.disk((center1_y, center1_x), radius, shape=image.shape)
    image[rr, cc] = 255
    
    rr, cc = draw.disk((center2_y, center2_x), radius, shape=image.shape)
    image[rr, cc] = 255
    
    return image, (center1_y, center1_x), (center2_y, center2_x)


def run_approximation_on_mask(binary_mask, approximator):
    """Запускает аппроксимацию с автоопределением N"""
    # Анализируем сложность
    recommended_n, metrics = analyze_blob_complexity(binary_mask > 128)
    
    print(f"\n Анализ сложности завершен")
    print(f"  Рекомендованное N: {recommended_n}")
    
    # Запускаем аппроксимацию с рекомендованным N
    approximator.target_mask = binary_mask > 128
    approximator.mask_height, approximator.mask_width = binary_mask.shape
    approximator.height, approximator.width = binary_mask.shape
    approximator.distance_map = None
    approximator.initial_centers = None
    
    # Предобработка
    from scipy.ndimage import distance_transform_edt
    from scipy.ndimage import maximum_filter
    
    approximator.distance_map = distance_transform_edt(approximator.target_mask)
    max_distance = np.max(approximator.distance_map)
    local_max = maximum_filter(approximator.distance_map, size=15) == approximator.distance_map
    local_max[approximator.distance_map < 0.45 * max_distance] = False
    coords = np.column_stack(np.where(local_max))
    
    filtered_coords = []
    min_distance = max(15, approximator.mask_width * 0.15)
    
    for coord in coords:
        if not filtered_coords:
            filtered_coords.append(coord)
            continue
        distances = np.sqrt(np.sum((np.array(filtered_coords) - coord)**2, axis=1))
        if np.min(distances) > min_distance:
            filtered_coords.append(coord)
    
    approximator.initial_centers = filtered_coords[:recommended_n] if len(filtered_coords) >= recommended_n else filtered_coords
    
    # Запуск оптимизации
    best_individual, fitness_history, iou_history, overlap_history, best_iou, extra_history, uncovered_history = \
        approximator.optimize_precision(
            num_circles=recommended_n,
            initial_centers=approximator.initial_centers,
            verbose=False
        )
    
    # Подсчет перекрытия
    max_overlap = 0
    if best_individual is not None and len(best_individual) >= 6:
        circle1 = best_individual[0:3]
        circle2 = best_individual[3:6]
        max_overlap = approximator.calculate_circle_overlap(circle1, circle2)
    
    return {
        'solution': best_individual,
        'iou': best_iou,
        'max_overlap': max_overlap,
        'recommended_n': recommended_n,
        'metrics': metrics
    }


def create_visualization(binary_mask, result, distance, save_path, radius=60):
    """Создает визуализацию результатов (3 панели)"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # 1. Исходная маска
    axes[0].imshow(binary_mask, cmap='gray')
    axes[0].set_title(f'Исходная маска\n(d={distance}px)', fontsize=10, fontweight='bold')
    axes[0].axis('off')
    
    # 2. Аппроксимация
    if result['solution'] is not None:
        approx_mask = np.zeros_like(binary_mask, dtype=bool)
        num_circles = len(result['solution']) // 3
        
        for i in range(num_circles):
            x, y, r = result['solution'][i*3:(i+1)*3]
            rr, cc = draw.disk((int(y), int(x)), int(r), shape=approx_mask.shape)
            approx_mask[rr, cc] = True
        
        axes[1].imshow(approx_mask, cmap='plasma')
        axes[1].set_title(f'Результат: {result["recommended_n"]} пора(ы)\nАппроксимация (IoU={result["iou"]:.3f})', 
                         fontsize=10, fontweight='bold')
        axes[1].axis('off')
        
        # 3. Ошибка
        diff = np.logical_xor(binary_mask > 128, approx_mask)
        axes[2].imshow(diff, cmap='Reds')
        axes[2].set_title(f'Ошибка\n(Overlap={result["max_overlap"]:.2f})', fontsize=10, fontweight='bold')
        axes[2].axis('off')
    
    plt.suptitle(f'Автоопределение N={result["recommended_n"]} | ' + 
                 f'Компактность={result["metrics"]["compactness"]:.2f}', 
                 fontsize=12, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def run_full_research(
    distance_range=(10, 90, 5),
    radius=60,
    image_size=(400, 400),
    save_results=True
):
    """Запускает полное исследование с автоопределением N"""
    print("="*80)
    print(" ИССЛЕДОВАНИЕ ПОРОГА СЛИЯНИЯ ПОР")
    print("   С АВТОМАТИЧЕСКИМ ОПРЕДЕЛЕНИЕМ КОЛИЧЕСТВА ОКРУЖНОСТЕЙ")
    print("="*80)
    print(f"Диапазон расстояний: {distance_range[0]} - {distance_range[1]} px (шаг {distance_range[2]})")
    print(f"Радиус пор: {radius} px")
    print("="*80)
    
    approximator = CircleGeneticApproximator(
        population_size=120,
        generations=200,
        mutation_rate=0.15,
        crossover_rate=0.85
    )
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"overlap_research_auto_{timestamp}"
    if save_results:
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(f"{results_dir}/visualizations", exist_ok=True)
    
    results = []
    distances = list(range(distance_range[0], distance_range[1] + 1, distance_range[2]))
    total_tests = len(distances)
    
    print(f"\n Всего тестов: {total_tests}\n")
    
    for idx, distance in enumerate(distances, 1):
        print(f"[{idx}/{total_tests}] Расстояние: {distance:3d} px", end=" ... ")
        
        # Создаем маску
        binary_mask, _, _ = create_two_pores_with_distance(distance, radius, image_size)
        
        # Сохраняем тестовую маску
        if save_results:
            mask_path = f"{results_dir}/visualizations/mask_d{distance:03d}.png"
            io.imsave(mask_path, binary_mask)
        
        # Анализируем и аппроксимируем
        result = run_approximation_on_mask(binary_mask, approximator)
        result['distance'] = distance
        
        results.append(result)
        
        # Сохраняем визуализацию
        if save_results:
            viz_path = f"{results_dir}/visualizations/result_d{distance:03d}.png"
            create_visualization(binary_mask, result, distance, viz_path, radius)
            print(f" N={result['recommended_n']} (IoU={result['iou']:.3f})")
        else:
            print(f" N={result['recommended_n']}")
    
    # Создаем DataFrame
    df = pd.DataFrame([{
        'distance': r['distance'],
        'recommended_n': r['recommended_n'],
        'iou': r['iou'],
        'max_overlap': r['max_overlap'],
        'compactness': r['metrics']['compactness'],
        'eccentricity': r['metrics']['eccentricity'],
    } for r in results])
    
    if save_results:
        # CSV
        csv_path = f"{results_dir}/results.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"\n Результаты сохранены в {csv_path}")
        
        # График N от расстояния
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(df['distance'], df['recommended_n'], s=100, c=df['iou'], 
                   cmap='RdYlGn', vmin=0.7, vmax=1.0, edgecolors='black', linewidth=2)
        plt.xlabel('Расстояние между центрами (px)', fontsize=12)
        plt.ylabel('Рекомендованное количество окружностей', fontsize=12)
        plt.title('Автоопределение N в зависимости от расстояния', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(df['distance'], df['iou'], 'o-', linewidth=2, markersize=8, color='blue')
        plt.axhline(y=0.85, color='green', linestyle='--', alpha=0.7, label='Порог IoU ≥ 0.85')
        plt.xlabel('Расстояние между центрами (px)', fontsize=12)
        plt.ylabel('IoU', fontsize=12)
        plt.title('Качество аппроксимации (IoU)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        graph_path = f"{results_dir}/auto_n_analysis.png"
        plt.savefig(graph_path, dpi=300, bbox_inches='tight')
        print(f" График сохранен в {graph_path}")
        
        # Статистика переходов
        n1_count = len(df[df['recommended_n'] == 1])
        n2_count = len(df[df['recommended_n'] == 2])
        
        # Находим переходную зону
        transition_distances = df[(df['recommended_n'].shift(1) == 1) & (df['recommended_n'] == 2)]['distance']
        if len(transition_distances) > 0:
            transition_distance = transition_distances.iloc[0]
            print(f"\n ПЕРЕХОДНАЯ ЗОНА: {transition_distance} px")
            print(f"   При d < {transition_distance} px → 1 окружность")
            print(f"   При d ≥ {transition_distance} px → 2 окружности")
        
        print(f"\n СТАТИСТИКА:")
        print(f"   Всего тестов: {len(df)}")
        print(f"   N=1: {n1_count} ({n1_count/len(df)*100:.1f}%)")
        print(f"   N=2: {n2_count} ({n2_count/len(df)*100:.1f}%)")
        print(f"   Средний IoU: {df['iou'].mean():.3f}")
        print(f"   Min IoU: {df['iou'].min():.3f}")
        print(f"   Max IoU: {df['iou'].max():.3f}")
    
    # Сводная таблица
    print("\n" + "="*80)
    print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("="*80)
    print(df[['distance', 'recommended_n', 'iou', 'compactness']].to_string(index=False))
    print("="*80)
    
    return df, results_dir


if __name__ == "__main__":
    df, results_dir = run_full_research(
        distance_range=(10, 90, 5),
        radius=60,
        image_size=(400, 400),
        save_results=True
    )
    
    print(f"\n ВСЕ РЕЗУЛЬТАТЫ СОХРАНЕНЫ В ПАПКУ: {results_dir}/")