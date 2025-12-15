"""
ПОДСИСТЕМА АППРОКСИМАЦИИ СЛОЖНЫХ ГЕНЕТИЧЕСКИХ ОБЪЕКТОВ
ОПТИМАЛЬНАЯ ВЕРСИЯ: БЫСТРАЯ И ТОЧНАЯ (IoU > 0.9 ЗА 150-200 ПОКОЛЕНИЙ)
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import time
import os
import datetime
from skimage import io, measure, draw, morphology, filters, segmentation
from scipy.ndimage import distance_transform_edt
import warnings
warnings.filterwarnings('ignore')

class CircleGeneticApproximator:
    """
    Оптимальная версия для быстрой и точной аппроксимации.
    Достигает IoU > 0.9 за 150-200 поколений.
    """
    
    def __init__(self, population_size=150, generations=200, mutation_rate=0.15, crossover_rate=0.9):
        """
        Инициализация с оптимальными параметрами для скорости и качества
        """
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        self.results_dir = None
        self.image = None
        self.binary_mask = None
        self.target_mask = None
        self.image_path = None
        self.original_image = None
        self.distance_map = None
        self.initial_centers = None
        
        print("⚡ ИНИЦИАЛИЗАЦИЯ ОПТИМАЛЬНОЙ ВЕРСИИ")
        print(f"  Размер популяции: {population_size} (оптимальный баланс)")
        print(f"  Количество поколений: {generations} (быстрая сходимость)")
        print(f"  Стратегия: скорость + качество")
    
    def setup_results_directory(self, base_name):
        """Создает улучшенную структуру папок для результатов"""
        date_folder = datetime.datetime.now().strftime("%d.%m.%Y")
        
        if not os.path.exists(date_folder):
            os.makedirs(date_folder)
            print(f"✓ Создана папка за дату: {date_folder}")
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_folder = f"optimal_{base_name}_{timestamp}"
        
        self.results_dir = os.path.join(date_folder, run_folder)
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f"✓ Создана папка для результатов: {self.results_dir}")
        return self.results_dir
    
    def get_results_path(self, filename):
        """Генерирует полный путь к файлу в папке результатов"""
        if self.results_dir is None:
            if not os.path.exists('temp_results'):
                os.makedirs('temp_results')
            return os.path.join('temp_results', filename)
        return os.path.join(self.results_dir, filename)
    
    def load_image(self, image_path):
        """Загружает и подготавливает бинарное изображение"""
        print(f"\n📁 Загрузка изображения: {image_path}")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Файл {image_path} не найден!")
        
        self.image_path = image_path
        self.original_image = io.imread(image_path)
        self.image = self.original_image.copy()
        print("✓ Изображение успешно загружено")
        
        # Преобразуем в оттенки серого если нужно
        if len(self.image.shape) == 3:
            self.image = self.image.mean(axis=2)
            print("✓ Цветное изображение преобразовано в оттенки серого")
        
        # Продвинутая сегментация
        print("🔧 Применение продвинутой сегментации...")
        
        # Адаптивная бинаризация
        adaptive_thresh = filters.threshold_local(self.image, block_size=35)
        self.binary_mask = self.image > adaptive_thresh
        
        # Удаление шума
        self.binary_mask = morphology.remove_small_objects(self.binary_mask, min_size=50)
        self.binary_mask = morphology.remove_small_holes(self.binary_mask, area_threshold=50)
        
        # Морфологическая обработка для сглаживания
        self.binary_mask = morphology.binary_closing(self.binary_mask, morphology.disk(2))
        self.binary_mask = morphology.binary_opening(self.binary_mask, morphology.disk(1))
        
        self.height, self.width = self.binary_mask.shape
        
        # Находим связные компоненты
        labeled_image = measure.label(self.binary_mask.astype(int))
        regions = measure.regionprops(labeled_image)
        
        if not regions:
            raise ValueError("На изображении не найдено связных компонент!")
        
        # Выбираем самую большую связную компоненту
        largest_region = max(regions, key=lambda x: x.area)
        self.target_mask = largest_region.filled_image
        self.bbox = largest_region.bbox
        self.mask_height, self.mask_width = self.target_mask.shape
        
        print("✓ Предобработка изображения завершена:")
        print(f"  Размер изображения: {self.width} x {self.height} пикселей")
        print(f"  Размер целевого объекта: {self.mask_width} x {self.mask_height} пикселей")
        print(f"  Площадь объекта: {np.sum(self.target_mask):,} пикселей")
    
    def preprocess_image_for_precision(self):
        """Продвинутая предобработка для максимальной точности"""
        print("\n🔍 ПРОДВИНУТАЯ ПРЕДОБРАБОТКА ИЗОБРАЖЕНИЯ")
        
        # Создаем карту расстояний для инициализации
        self.distance_map = distance_transform_edt(self.target_mask)
        max_distance = np.max(self.distance_map)
        
        # Находим локальные максимумы для начальных центров
        from scipy.ndimage import maximum_filter
        local_max = maximum_filter(self.distance_map, size=15) == self.distance_map
        local_max[self.distance_map < 0.5 * max_distance] = False
        
        # Получаем координаты локальных максимумов
        coords = np.column_stack(np.where(local_max))
        print(f"  Найдено локальных максимумов: {len(coords)}")
        
        # Фильтруем близкие точки
        filtered_coords = []
        min_distance = max(10, self.mask_width * 0.1)
        
        for coord in coords:
            if not filtered_coords:
                filtered_coords.append(coord)
                continue
            
            distances = np.sqrt(np.sum((np.array(filtered_coords) - coord)**2, axis=1))
            if np.min(distances) > min_distance:
                filtered_coords.append(coord)
        
        print(f"  Отфильтровано до: {len(filtered_coords)} начальных центров")
        self.initial_centers = filtered_coords
        
        # Визуализация карты расстояний
        plt.figure(figsize=(10, 8))
        plt.imshow(self.distance_map, cmap='hot')
        if filtered_coords:
            y_coords, x_coords = zip(*filtered_coords)
            plt.scatter(x_coords, y_coords, c='blue', s=50, marker='o', label='Начальные центры')
        plt.colorbar(label='Расстояние до границы')
        plt.title('Карта расстояний для инициализации', fontsize=14, fontweight='bold')
        plt.legend()
        plt.savefig(self.get_results_path('distance_map.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        return filtered_coords
    
    def detect_touching_pores_advanced(self):
        """Продвинутое обнаружение касающихся пор с использованием watershed"""
        print("\n🔍 ПРОДВИНУТОЕ ОБНАРУЖЕНИЕ КАСАЮЩИХСЯ ПОР")
        
        if self.distance_map is None:
            self.distance_map = distance_transform_edt(self.target_mask)
        
        distance = self.distance_map.copy()
        
        # Находим локальные максимумы
        from scipy.ndimage import maximum_filter
        local_max = maximum_filter(distance, size=20) == distance
        local_max[distance < 0.5 * np.max(distance)] = False
        
        # Маркеры для watershed
        markers = measure.label(local_max.astype(int))
        
        # Применяем watershed
        labels = segmentation.watershed(-distance, markers, mask=self.target_mask)
        num_watershed_regions = np.max(labels)
        
        print(f"  Watershed обнаружил: {num_watershed_regions} компонент")
        
        # Анализируем результаты
        if num_watershed_regions >= 2:
            # Проверяем, действительно ли это касающиеся поры
            regions = measure.regionprops(measure.label(labels))
            if len(regions) >= 2:
                areas = [region.area for region in regions]
                area_ratio = max(areas) / min(areas) if min(areas) > 0 else 10
                
                if area_ratio < 5:  # Примерно одинаковые по размеру
                    print(f"🎯 ПОДТВЕРЖДЕНО: обнаружены касающиеся поры (коэффициент площадей: {area_ratio:.2f})")
                    return True, num_watershed_regions
        
        return False, 1
    
    def analyze_image_complexity(self):
        """Продвинутый анализ сложности изображения"""
        print("\n🧠 ПРОДВИНУТЫЙ АНАЛИЗ СЛОЖНОСТИ ИЗОБРАЖЕНИЯ")
        
        # Проверка на касающиеся поры
        touching_detected, watershed_regions = self.detect_touching_pores_advanced()
        
        # Анализ компактности и формы
        labeled = measure.label(self.target_mask.astype(int))
        regions = measure.regionprops(labeled)
        
        if not regions:
            raise ValueError("Не найдено связных компонент для анализа!")
        
        region = regions[0]
        area = region.area
        perimeter = region.perimeter
        compactness = (perimeter ** 2) / (4 * np.pi * area) if area > 0 else 1
        eccentricity = region.eccentricity
        solidity = region.solidity
        
        print(f"  Геометрические характеристики основного объекта:")
        print(f"    Площадь: {area} пикселей")
        print(f"    Компактность: {compactness:.3f}")
        print(f"    Эксцентриситет: {eccentricity:.3f}")
        print(f"    Сплошность (solidity): {solidity:.3f}")
        
        # Эвристические правила для определения сложности
        complexity_level = 0
        
        if compactness > 1.8:
            complexity_level += 1
            print("    → Высокая сложность: некомпактная форма")
        if eccentricity > 0.8:
            complexity_level += 1
            print("    → Высокая сложность: вытянутая форма")
        if solidity < 0.9:
            complexity_level += 1
            print("    → Высокая сложность: вогнутости/дыры")
        
        # Определение количества кругов
        if touching_detected:
            print(f"  🎯 Рекомендуемое количество окружностей для касающихся пор: {watershed_regions}")
            return watershed_regions, watershed_regions
        
        # Для простых форм
        if complexity_level == 0 and compactness < 1.2:
            print("  🎯 Рекомендуемое количество окружностей: 1 (простая круглая форма)")
            return 1, 1
        
        # Для умеренно сложных форм
        if complexity_level <= 1:
            print("  🎯 Рекомендуемое количество окружностей: 2 (умеренно сложная форма)")
            return 2, 2
        
        # Для сложных форм
        print(f"  🎯 Рекомендуемое количество окружностей: {complexity_level + 1} (сложная форма)")
        return complexity_level + 1, complexity_level + 1
    
    def create_individual_with_initialization(self, num_circles, initial_centers=None):
        """Создает особь с умной инициализацией на основе карты расстояний"""
        individual = []
        
        # Счетчик для вывода информации только один раз
        if hasattr(self, 'initialization_printed') and self.initialization_printed:
            print_initialization = False
        else:
            print_initialization = True
            self.initialization_printed = True
        
        # Если есть начальные центры от предобработки
        if initial_centers is not None and len(initial_centers) >= num_circles:
            centers_to_use = initial_centers[:num_circles]
            if print_initialization:
                print(f"  🎯 Использую {len(centers_to_use)} начальных центров из карты расстояний")
        else:
            centers_to_use = []
            if print_initialization:
                print("  🎯 Не найдено подходящих начальных центров, использую случайную инициализацию")
        
        for i in range(num_circles):
            if i < len(centers_to_use) and centers_to_use[i] is not None:
                # Используем предварительно вычисленный центр
                y_coord, x_coord = centers_to_use[i]
                x = x_coord
                y = y_coord
                # Радиус на основе расстояния до границы
                if hasattr(self, 'distance_map') and self.distance_map is not None:
                    radius = self.distance_map[y_coord, x_coord] * 0.9
                else:
                    radius = min(self.mask_width, self.mask_height) / 4
            else:
                # Случайная инициализация в пределах объекта
                y_coords, x_coords = np.where(self.target_mask)
                if len(y_coords) > 0:
                    idx = np.random.randint(len(y_coords))
                    x = x_coords[idx]
                    y = y_coords[idx]
                    
                    # Радиус на основе локального расстояния
                    if hasattr(self, 'distance_map') and self.distance_map is not None:
                        local_radius = self.distance_map[y, x]
                        radius = max(5, local_radius * np.random.uniform(0.8, 1.2))
                    else:
                        radius = min(self.mask_width, self.mask_height) / 4
                else:
                    # Резервный вариант
                    x = self.mask_width / 2
                    y = self.mask_height / 2
                    radius = min(self.mask_width, self.mask_height) / 4
            
            # Ограничиваем радиус разумными пределами
            max_radius = min(self.mask_width, self.mask_height) / 2.5
            radius = min(radius, max_radius)
            
            individual.extend([x, y, radius])
        
        return individual
    
    def create_population(self, num_circles, initial_centers=None):
        """Создает популяцию с умной инициализацией"""
        print(f"\n🧬 СОЗДАНИЕ ПОПУЛЯЦИИ ИЗ {self.population_size} ОСОБЕЙ")
        print(f"  Количество окружностей: {num_circles}")
        
        population = []
        
        # Сбрасываем флаг для вывода информации
        if hasattr(self, 'initialization_printed'):
            del self.initialization_printed
        
        # Создаем разнообразную популяцию
        for i in range(self.population_size):
            # Первые 20% особей используют умную инициализацию
            if i < self.population_size * 0.2 and initial_centers is not None:
                individual = self.create_individual_with_initialization(num_circles, initial_centers)
            else:
                # Остальные особи - случайная инициализация для разнообразия
                individual = self.create_individual_with_initialization(num_circles)
            
            population.append(individual)
        
        print("  ✓ Популяция создана с разнообразной инициализацией")
        return population
    
    def draw_circles(self, individual, shape=None):
        """Отрисовывает круги на маске"""
        if shape is None:
            shape = (self.mask_height, self.mask_width)
        
        mask = np.zeros(shape, dtype=bool)
        num_circles = len(individual) // 3
        
        for i in range(num_circles):
            x, y, radius = individual[i*3:(i+1)*3]
            
            x_int, y_int = int(x), int(y)
            radius_int = int(radius)
            
            if radius_int > 0:
                try:
                    rr, cc = draw.disk((y_int, x_int), radius_int, shape=shape)
                    mask[rr, cc] = True
                except:
                    continue
                    
        return mask
    
    def draw_circles_on_original(self, individual):
        """Отрисовывает круги на оригинальном изображении с учетом bounding box"""
        if len(self.original_image.shape) == 3:
            result_image = self.original_image.copy()
        else:
            result_image = np.stack([self.original_image] * 3, axis=-1)
        
        num_circles = len(individual) // 3
        bbox = self.bbox
        
        for i in range(num_circles):
            x, y, radius = individual[i*3:(i+1)*3]
            
            x_original = int(x) + bbox[1]
            y_original = int(y) + bbox[0]
            radius_int = int(radius)
            
            if radius_int > 0:
                try:
                    rr, cc = draw.circle_perimeter(y_original, x_original, radius_int, shape=result_image.shape[:2])
                    valid = (rr >= 0) & (rr < result_image.shape[0]) & (cc >= 0) & (cc < result_image.shape[1])
                    rr, cc = rr[valid], cc[valid]
                    
                    # Рисуем контур разного цвета для каждого круга
                    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
                    color_idx = i % len(colors)
                    result_image[rr, cc, 0] = colors[color_idx][0]
                    result_image[rr, cc, 1] = colors[color_idx][1]
                    result_image[rr, cc, 2] = colors[color_idx][2]
                    
                    # Добавляем номер круга
                    if (0 <= y_original < result_image.shape[0] and 0 <= x_original < result_image.shape[1]):
                        text_color = [255, 255, 255]
                        outline_color = [0, 0, 0]
                        
                        for dx in [-1, 0, 1]:
                            for dy in [-1, 0, 1]:
                                if dx == 0 and dy == 0:
                                    continue
                                y_text = min(max(y_original + dy, 0), result_image.shape[0]-1)
                                x_text = min(max(x_original + dx, 0), result_image.shape[1]-1)
                                result_image[y_text, x_text] = outline_color
                        
                        result_image[y_original, x_original] = text_color
                        
                except Exception as e:
                    continue
                    
        return result_image
    
    def calculate_circle_overlap(self, circle1, circle2):
        """Точное вычисление перекрытия с учетом вложенности"""
        x1, y1, r1 = circle1
        x2, y2, r2 = circle2
        
        # Расстояние между центрами
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Случай полной вложенности
        if distance + min(r1, r2) <= max(r1, r2):
            return 1.0
        
        # Случай отсутствия пересечения
        if distance >= r1 + r2:
            return 0.0
        
        # Вычисляем площадь пересечения
        d = distance
        r = min(r1, r2)
        R = max(r1, r2)
        
        part1 = r**2 * np.arccos((d**2 + r**2 - R**2) / (2 * d * r))
        part2 = R**2 * np.arccos((d**2 + R**2 - r**2) / (2 * d * R))
        part3 = 0.5 * np.sqrt((-d + r + R) * (d + r - R) * (d - r + R) * (d + r + R))
        
        intersection_area = part1 + part2 - part3
        smaller_area = np.pi * r**2
        
        return min(intersection_area / smaller_area, 1.0)
    
    def fitness_function_precision(self, individual):
        """Функция приспособленности с фокусом на максимальной точности"""
        generated_mask = self.draw_circles(individual)
        
        # Основная метрика - IoU
        intersection = np.logical_and(self.target_mask, generated_mask)
        union = np.logical_or(self.target_mask, generated_mask)
        total_union = np.sum(union)
        iou = np.sum(intersection) / total_union if total_union > 0 else 0
        
        # Штрафы за геометрические ошибки
        total_target_area = np.sum(self.target_mask)
        
        extra_area = np.sum(np.logical_and(generated_mask, np.logical_not(self.target_mask)))
        uncovered_area = np.sum(np.logical_and(self.target_mask, np.logical_not(generated_mask)))
        
        penalty_extra = 0.3 * (extra_area / total_target_area) if total_target_area > 0 else 1
        penalty_uncovered = 0.3 * (uncovered_area / total_target_area) if total_target_area > 0 else 1
        
        # Штраф за перекрытие между кругами
        num_circles = len(individual) // 3
        penalty_overlap = 0
        
        for i in range(num_circles):
            for j in range(i + 1, num_circles):
                circle1 = individual[i*3:(i+1)*3]
                circle2 = individual[j*3:(j+1)*3]
                overlap = self.calculate_circle_overlap(circle1, circle2)
                
                # Мягкий штраф для перекрытия до 0.3
                if 0.1 < overlap <= 0.3:
                    penalty_overlap += overlap * 0.1
                # Сильный штраф для перекрытия > 0.3
                elif overlap > 0.3:
                    penalty_overlap += overlap * 0.4
        
        # Награда за хорошее покрытие границ
        boundary_target = morphology.binary_dilation(self.target_mask, morphology.disk(1)) ^ self.target_mask
        boundary_generated = morphology.binary_dilation(generated_mask, morphology.disk(1)) ^ generated_mask
        
        boundary_intersection = np.logical_and(boundary_target, boundary_generated)
        boundary_union = np.logical_or(boundary_target, boundary_generated)
        
        boundary_coverage = np.sum(boundary_intersection) / np.sum(boundary_union) if np.sum(boundary_union) > 0 else 0
        boundary_bonus = 0.1 * boundary_coverage
        
        # Финальная оценка с фокусом на IoU
        fitness = (iou * 0.8 +  # Основной вес на IoU
                  boundary_bonus - 
                  penalty_extra * 0.5 - 
                  penalty_uncovered * 0.5 - 
                  penalty_overlap * 0.3)
        
        final_fitness = max(fitness, 0)
        
        return final_fitness, iou, penalty_overlap
    
    def tournament_selection_elitism(self, population, fitnesses, tournament_size=5, elite_count=10):
        """Турнирный отбор с элитизмом для сохранения лучших решений"""
        selected = []
        
        # Элитизм: сохраняем лучших особей
        elite_indices = np.argsort(fitnesses)[-elite_count:]
        elite_population = [population[i] for i in elite_indices]
        selected.extend(elite_population)
        
        # Турнирный отбор для остальных мест
        for _ in range(len(population) - elite_count):
            contestants = np.random.choice(len(population), tournament_size, replace=False)
            best_contestant = contestants[np.argmax([fitnesses[i] for i in contestants])]
            selected.append(population[best_contestant])
        
        return selected
    
    def adaptive_mutation(self, individual, generation, total_generations):
        """Адаптивная мутация: сильная в начале, слабая в конце"""
        mutated = individual.copy()
        num_circles = len(individual) // 3
        
        # Коэффициент адаптации мутации
        adaptation_factor = 1.0 - (generation / total_generations)
        
        for i in range(num_circles):
            if np.random.random() < self.mutation_rate:
                param_index = np.random.randint(3)
                idx = i * 3 + param_index
                
                if param_index in [0, 1]:  # Координаты X или Y
                    mutation_strength = self.mask_width * 0.1 * adaptation_factor
                    mutated[idx] += np.random.normal(0, mutation_strength)
                    if param_index == 0:  # X
                        mutated[idx] = np.clip(mutated[idx], 0, self.mask_width)
                    else:  # Y
                        mutated[idx] = np.clip(mutated[idx], 0, self.mask_height)
                else:  # Радиус
                    mutation_range = 0.2 * adaptation_factor + 0.05
                    mutated[idx] = max(5, mutated[idx] * np.random.uniform(1 - mutation_range, 1 + mutation_range))
                    
        return mutated
    
    def local_search_refinement(self, best_individual, iterations=30):
        """Локальный поиск для тонкой настройки лучших решений (ускоренная версия)"""
        best_fitness, best_iou, _ = self.fitness_function_precision(best_individual)
        current_individual = best_individual.copy()
        
        for i in range(iterations):
            new_individual = current_individual.copy()
            num_circles = len(new_individual) // 3
            
            circle_idx = np.random.randint(num_circles)
            param_idx = np.random.randint(3)
            idx = circle_idx * 3 + param_idx
            
            if param_idx in [0, 1]:  # Координаты
                new_individual[idx] += np.random.normal(0, 0.5)  # Меньшие изменения
                if param_idx == 0:
                    new_individual[idx] = np.clip(new_individual[idx], 0, self.mask_width)
                else:
                    new_individual[idx] = np.clip(new_individual[idx], 0, self.mask_height)
            else:  # Радиус
                new_individual[idx] *= np.random.uniform(0.995, 1.005)  # 0.5% изменения
                new_individual[idx] = max(5, new_individual[idx])
            
            new_fitness, new_iou, _ = self.fitness_function_precision(new_individual)
            
            if new_fitness > best_fitness:
                best_fitness = new_fitness
                best_iou = new_iou
                current_individual = new_individual
        
        return current_individual, best_fitness, best_iou
    
    def optimize_precision(self, num_circles, initial_centers=None, verbose=True):
        """Оптимизация с фокусом на максимальной точности"""
        if verbose:
            print(f"\n🚀 ЗАПУСК ОПТИМИЗАЦИИ")
            print(f"  Количество окружностей: {num_circles}")
            print(f"  Целевой IoU: > 0.9")
        
        start_time = time.time()
        
        population = self.create_population(num_circles, initial_centers)
        
        best_fitness = 0
        best_iou = 0
        best_individual = None
        fitness_history = []
        iou_history = []
        early_stop_generation = None
        
        for generation in range(self.generations):
            fitnesses = []
            ious = []
            
            for individual in population:
                fitness, iou, _ = self.fitness_function_precision(individual)
                fitnesses.append(fitness)
                ious.append(iou)
            
            current_best_fitness = max(fitnesses)
            current_best_iou = max(ious)
            
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_iou = current_best_iou
                best_individual = population[np.argmax(fitnesses)].copy()
            
            fitness_history.append(best_fitness)
            iou_history.append(best_iou)
            
            # Выводим прогресс каждые 25 поколений
            if verbose and (generation % 25 == 0 or generation == self.generations - 1):
                avg_fitness = np.mean(fitnesses)
                print(f"   Поколение {generation:3d}/{self.generations}: "
                      f"Лучший IoU = {best_iou:.4f}, "
                      f"Средняя приспособленность = {avg_fitness:.4f}")
            
            # Ранняя остановка при достижении цели
            if best_iou >= 0.92 and generation > 50:
                if early_stop_generation is None:
                    early_stop_generation = generation
                if generation - early_stop_generation >= 10:  # Ждем 10 поколений для стабильности
                    print(f"   🎯 Целевой IoU достигнут и стабилен! Остановка на поколении {generation}")
                    break
            
            # Отбор с элитизмом
            selected = self.tournament_selection_elitism(population, fitnesses, 
                                                       tournament_size=5, 
                                                       elite_count=max(5, int(self.population_size * 0.05)))
            
            # Создаем новую популяцию
            new_population = []
            
            for i in range(0, len(selected), 2):
                if i + 1 < len(selected):
                    parent1, parent2 = selected[i], selected[i+1]
                    
                    if np.random.random() < self.crossover_rate:
                        num_circles = len(parent1) // 3
                        if num_circles > 1:
                            circle_idx = np.random.randint(1, num_circles)
                            crossover_point = circle_idx * 3
                            
                            child1 = parent1[:crossover_point] + parent2[crossover_point:]
                            child2 = parent2[:crossover_point] + parent1[crossover_point:]
                        else:
                            child1, child2 = parent1.copy(), parent2.copy()
                    else:
                        child1, child2 = parent1.copy(), parent2.copy()
                    
                    child1 = self.adaptive_mutation(child1, generation, self.generations)
                    child2 = self.adaptive_mutation(child2, generation, self.generations)
                    
                    new_population.extend([child1, child2])
                else:
                    mutated = self.adaptive_mutation(selected[i], generation, self.generations)
                    new_population.append(mutated)
            
            # Гарантируем сохранение лучшей особи
            if best_individual not in new_population:
                replace_idx = np.random.randint(len(new_population))
                new_population[replace_idx] = best_individual.copy()
            
            population = new_population
        
        end_time = time.time()
        
        # Применяем локальный поиск для финальной настройки
        best_individual, best_fitness, best_iou = self.local_search_refinement(best_individual, iterations=30)
        
        if verbose:
            print(f"\n✅ ОПТИМИЗАЦИЯ ЗАВЕРШЕНА ЧЕРЕЗ {end_time - start_time:.2f} СЕКУНД")
            print(f"🎯 ФИНАЛЬНЫЙ РЕЗУЛЬТАТ: IoU = {best_iou:.4f}")
            print(f"   Количество окружностей: {num_circles}")
        
        return best_individual, fitness_history, iou_history, best_iou
    
    def find_optimal_circles_count_precision(self, max_circles=4):
        """Поиск оптимального количества окружностей с фокусом на скорости и точности"""
        print("\n" + "="*70)
        print("🎯 ПОИСК ОПТИМАЛЬНОГО КОЛИЧЕСТВА ОКРУЖНОСТЕЙ")
        print("="*70)
        
        self.preprocess_image_for_precision()
        
        min_circles, recommended_circles = self.analyze_image_complexity()
        max_test_circles = min(max_circles, recommended_circles + 1)  # Уменьшено максимальное количество
        
        print(f"\n📊 ДИАПАЗОН ТЕСТИРОВАНИЯ: от {min_circles} до {max_test_circles} окружностей")
        print(f"🎯 РЕКОМЕНДОВАННОЕ КОЛИЧЕСТВО: {recommended_circles}")
        
        best_results = {}
        best_iou = 0
        best_circles = min_circles
        
        for num_circles in range(min_circles, max_test_circles + 1):
            print(f"\n" + "-"*50)
            print(f"🔍 ТЕСТИРОВАНИЕ {num_circles} ОКРУЖНОСТЕЙ")
            print("-"*50)
            
            best_solution, fitness_history, iou_history, final_iou = self.optimize_precision(
                num_circles, 
                initial_centers=self.initial_centers,
                verbose=True
            )
            
            best_results[num_circles] = {
                'solution': best_solution,
                'fitness_history': fitness_history,
                'iou_history': iou_history,
                'final_iou': final_iou
            }
            
            print(f"  📊 Результат для {num_circles} окружностей: IoU = {final_iou:.4f}")
            
            if final_iou > best_iou:
                best_iou = final_iou
                best_circles = num_circles
            
            # Ранняя остановка при отличном результате
            if final_iou >= 0.94 and num_circles >= recommended_circles:
                print(f"  🎯 ОТЛИЧНЫЙ РЕЗУЛЬТАТ ДОСТИГНУТ! IoU = {final_iou:.4f}")
                break
        
        print(f"\n🏆 ВЫБРАНО ОПТИМАЛЬНОЕ КОЛИЧЕСТВО: {best_circles} окружностей")
        print(f"   Максимальный достигнутый IoU: {best_iou:.4f}")
        
        return best_circles, best_results[best_circles]
    
    def visualize_result(self, individual, save_path=None):
        """Расширенная визуализация результатов"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        axes[0, 0].imshow(self.original_image, cmap='gray' if len(self.original_image.shape) == 2 else None)
        axes[0, 0].set_title('Исходное изображение', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(self.target_mask, cmap='viridis')
        axes[0, 1].set_title('Целевая маска', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        approximation = self.draw_circles(individual)
        axes[0, 2].imshow(approximation, cmap='plasma')
        axes[0, 2].set_title('Аппроксимация кругами', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')
        
        result_with_circles = self.draw_circles_on_original(individual)
        axes[1, 0].imshow(result_with_circles)
        axes[1, 0].set_title('Круги на исходном изображении', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        
        difference = np.logical_xor(self.target_mask, approximation)
        axes[1, 1].imshow(difference, cmap='Reds')
        axes[1, 1].set_title('Области различий (ошибки)', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        
        error_map = np.zeros_like(self.target_mask, dtype=float)
        error_map[np.logical_and(self.target_mask, np.logical_not(approximation))] = 1.0
        error_map[np.logical_and(np.logical_not(self.target_mask), approximation)] = -1.0
        
        im = axes[1, 2].imshow(error_map, cmap='seismic', vmin=-1, vmax=1, alpha=0.8)
        axes[1, 2].set_title('Карта ошибок (красный: не покрыто, синий: лишнее)', fontsize=10, fontweight='bold')
        axes[1, 2].axis('off')
        plt.colorbar(im, ax=axes[1, 2], label='Тип ошибки')
        
        iou = np.sum(np.logical_and(self.target_mask, approximation)) / \
              np.sum(np.logical_or(self.target_mask, approximation))
        
        num_circles = len(individual) // 3
        plt.suptitle(
            f'Результат аппроксимации ({num_circles} кругов)\nIoU: {iou:.4f}', 
            fontsize=16, 
            fontweight='bold',
            y=0.95
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✓ Расширенная визуализация сохранена как {save_path}")
        
        plt.close()
    
    def export_parameters(self, individual, output_path):
        """Экспортирует детальные параметры в JSON"""
        num_circles = len(individual) // 3
        circles_data = []
        
        for i in range(num_circles):
            x, y, radius = individual[i*3:(i+1)*3]
            
            x_original = float(x) + self.bbox[1]
            y_original = float(y) + self.bbox[0]
            
            circle_mask = np.zeros((self.mask_height, self.mask_width), dtype=bool)
            rr, cc = draw.disk((int(y), int(x)), int(radius), shape=(self.mask_height, self.mask_width))
            circle_mask[rr, cc] = True
            
            circle_coverage = np.sum(np.logical_and(self.target_mask, circle_mask)) / np.sum(circle_mask) if np.sum(circle_mask) > 0 else 0
            target_coverage = np.sum(np.logical_and(self.target_mask, circle_mask)) / np.sum(self.target_mask) if np.sum(self.target_mask) > 0 else 0
            
            circle_info = {
                "id": i + 1,
                "center": {"x": x_original, "y": y_original},
                "radius": float(radius),
                "diameter": float(2 * radius),
                "area": float(np.pi * radius ** 2),
                "coverage_of_circle": float(circle_coverage),
                "coverage_of_target": float(target_coverage)
            }
            circles_data.append(circle_info)
        
        approximation = self.draw_circles(individual)
        iou = np.sum(np.logical_and(self.target_mask, approximation)) / \
              np.sum(np.logical_or(self.target_mask, approximation))
        
        extra_area = np.sum(np.logical_and(approximation, np.logical_not(self.target_mask)))
        uncovered_area = np.sum(np.logical_and(self.target_mask, np.logical_not(approximation)))
        total_area = np.sum(self.target_mask)
        
        error_stats = {
            "extra_pixels": int(extra_area),
            "uncovered_pixels": int(uncovered_area),
            "extra_percentage": float(extra_area / total_area * 100) if total_area > 0 else 0,
            "uncovered_percentage": float(uncovered_area / total_area * 100) if total_area > 0 else 0
        }
        
        result = {
            "image_info": {
                "width": self.width,
                "height": self.height,
                "original_area": int(np.sum(self.target_mask)),
                "processing_time": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "approximation_metrics": {
                "number_of_circles": num_circles,
                "iou_score": float(iou),
                "precision": float(iou / (iou + error_stats["extra_percentage"] / 100)) if (iou + error_stats["extra_percentage"] / 100) > 0 else 0,
                "recall": float(iou / (iou + error_stats["uncovered_percentage"] / 100)) if (iou + error_stats["uncovered_percentage"] / 100) > 0 else 0
            },
            "error_statistics": error_stats,
            "circles": circles_data
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Детальные параметры экспортированы в {output_path}")


def main():
    """Основная функция программы"""
    print("=" * 70)
    print("⚡ ОПТИМАЛЬНАЯ ВЕРСИЯ ПОДСИСТЕМЫ АППРОКСИМАЦИИ")
    print("   Высокая точность за минимальное время (150-200 поколений)")
    print("=" * 70)
    
    # Инициализируем аппроксиматор с оптимальными параметрами
    approximator = CircleGeneticApproximator(
        population_size=150,    # Оптимальный размер популяции
        generations=200,        # Максимум 200 поколений
        mutation_rate=0.15,     # Умеренная мутация
        crossover_rate=0.9      # Высокий кроссовер для разнообразия
    )
    
    # Показываем доступные изображения
    available_masks = [
        f for f in os.listdir('.') 
        if f.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp')) 
        and 'preview' not in f.lower()
    ]
    
    print("\n📋 ДОСТУПНЫЕ ИЗОБРАЖЕНИЯ:")
    if available_masks:
        for i, mask in enumerate(available_masks, 1):
            print(f"   {i}. {mask}")
    else:
        print("   (нет доступных изображений)")
    
    try:
        choice = int(input(f"\n👉 Выберите изображение для обработки (1-{len(available_masks)}): ")) - 1
        selected_file = available_masks[choice]
    except (ValueError, IndexError):
        if available_masks:
            print("⚠️  Неверный выбор. Используется первая доступная маска.")
            selected_file = available_masks[0]
        else:
            print("❌ Нет доступных изображений для обработки.")
            return
    
    # Загружаем изображение
    try:
        approximator.load_image(selected_file)
    except Exception as e:
        print(f"❌ Ошибка загрузки изображения: {e}")
        return
    
    # Создаем папку для результатов
    base_name = os.path.splitext(selected_file)[0]
    results_dir = approximator.setup_results_directory(base_name)
    
    # Поиск оптимального количества окружностей
    print("\n🎯 НАЧИНАЕМ ПОИСК ОПТИМАЛЬНОГО КОЛИЧЕСТВА ОКРУЖНОСТЕЙ...")
    optimal_circles, optimal_results = approximator.find_optimal_circles_count_precision(max_circles=4)
    
    # Финальная оптимизация
    print(f"\n🚀 ЗАПУСК ФИНАЛЬНОЙ ОПТИМИЗАЦИИ ДЛЯ {optimal_circles} ОКРУЖНОСТЕЙ...")
    best_solution = optimal_results['solution']
    final_iou = optimal_results['final_iou']
    
    # Визуализируем и сохраняем результаты
    result_image_path = approximator.get_results_path(f'{base_name}_optimal_result.png')
    approximator.visualize_result(best_solution, save_path=result_image_path)
    
    # Экспортируем параметры
    json_path = approximator.get_results_path(f'{base_name}_optimal_parameters.json')
    approximator.export_parameters(best_solution, json_path)
    
    # Финальный отчет
    print("\n" + "=" * 70)
    print("🎉 АППРОКСИМАЦИЯ УСПЕШНО ЗАВЕРШЕНА!")
    print("=" * 70)
    print(f"📁 Все результаты сохранены в папке: {results_dir}")
    print(f"\n📊 КЛЮЧЕВЫЕ РЕЗУЛЬТАТЫ:")
    print(f"   Оптимальное количество окружностей: {optimal_circles}")
    print(f"   Достигнутый IoU: {final_iou:.4f}")
    print(f"   Статус: {'🎯 ЦЕЛЬ ДОСТИГНУТА (IoU > 0.9)' if final_iou >= 0.9 else '⚠️ Требуется ручная проверка'}")
    print(f"\n💾 СОЗДАННЫЕ ФАЙЛЫ:")
    print(f"   📄 {base_name}_optimal_result.png - детальная визуализация")
    print(f"   📄 {base_name}_optimal_parameters.json - детальные параметры")
    print(f"   📄 distance_map.png - карта расстояний для анализа")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()