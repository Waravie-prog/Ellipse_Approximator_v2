"""
ПОДСИСТЕМА АППРОКСИМАЦИИ СЛОЖНЫХ МОРФОЛОГИЧЕСКИХ ОБЪЕКТОВ
ВЕРСИЯ ДЛЯ ВКР С ПОЛНЫМ ИССЛЕДОВАНИЕМ АЛГОРИТМА
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
    Модуль аппроксимации поровых кластеров на основе генетического алгоритма
    Для ВКР: исследование влияния параметров, устойчивость, визуализация
    """
    
    def __init__(self, population_size=120, generations=200, 
                 mutation_rate=0.15, crossover_rate=0.85):
        """
        Инициализация с параметрами ГА
        
        Args:
            population_size: размер популяции (оптимизировано для 1-4 пор)
            generations: количество поколений
            mutation_rate: вероятность мутации
            crossover_rate: вероятность кроссовера
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
        self.num_pores = None
        self.bbox = None
        self.mask_height = 0
        self.mask_width = 0
        self.height = 0
        self.width = 0
        
        print("⚡ ИНИЦИАЛИЗАЦИЯ МОДУЛЯ АППРОКСИМАЦИИ")
        print(f"  Размер популяции: {population_size}")
        print(f"  Количество поколений: {generations}")
        print(f"  Вероятность мутации: {mutation_rate}")
        print(f"  Вероятность кроссовера: {crossover_rate}")
    
    # =========================================================================
    # БЛОК 1: ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
    # =========================================================================
    
    def setup_results_directory(self, base_name):
        """Создаёт структуру папок для результатов"""
        date_folder = datetime.datetime.now().strftime("%d.%m.%Y")
        
        if not os.path.exists(date_folder):
            os.makedirs(date_folder)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_folder = f"final_{base_name}_{timestamp}"
        
        self.results_dir = os.path.join(date_folder, run_folder)
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f"✓ Папка результатов: {self.results_dir}")
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
        
        # Преобразуем в оттенки серого если нужно
        if len(self.image.shape) == 3:
            self.image = self.image.mean(axis=2)
        
        # Продвинутая сегментация
        adaptive_thresh = filters.threshold_local(self.image, block_size=25)
        self.binary_mask = self.image > adaptive_thresh
        
        # Удаление шума
        self.binary_mask = morphology.remove_small_objects(
            self.binary_mask, min_size=20)
        self.binary_mask = morphology.remove_small_holes(
            self.binary_mask, area_threshold=20)
        
        # Морфологическая обработка
        self.binary_mask = morphology.binary_closing(
            self.binary_mask, morphology.disk(1))
        self.binary_mask = morphology.binary_opening(
            self.binary_mask, morphology.disk(1))
        
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
        
        print("✓ Предобработка завершена:")
        print(f"  Размер изображения: {self.width} x {self.height}")
        print(f"  Размер объекта: {self.mask_width} x {self.mask_height}")
        print(f"  Площадь объекта: {np.sum(self.target_mask):,} пикселей")
    
    def preprocess_image_for_precision(self):
        """Продвинутая предобработка для максимальной точности"""
        print("\n🔍 ПРЕДОБРАБОТКА ИЗОБРАЖЕНИЯ")
        
        # Создаем карту расстояний
        self.distance_map = distance_transform_edt(self.target_mask)
        max_distance = np.max(self.distance_map)
        
        # Находим локальные максимумы
        from scipy.ndimage import maximum_filter
        local_max = maximum_filter(self.distance_map, size=15) == self.distance_map
        local_max[self.distance_map < 0.45 * max_distance] = False
        
        coords = np.column_stack(np.where(local_max))
        print(f"  Найдено локальных максимумов: {len(coords)}")
        
        # Фильтруем близкие точки
        filtered_coords = []
        min_distance = max(15, self.mask_width * 0.15)
        
        for coord in coords:
            if not filtered_coords:
                filtered_coords.append(coord)
                continue
            distances = np.sqrt(np.sum(
                (np.array(filtered_coords) - coord)**2, axis=1))
            if np.min(distances) > min_distance:
                filtered_coords.append(coord)
        
        self.num_pores = min(len(filtered_coords), 4)
        print(f"  Определено количество пор: {self.num_pores}")
        
        self.initial_centers = filtered_coords[:4]
        
        # Визуализация карты расстояний
        plt.figure(figsize=(10, 8))
        plt.imshow(self.distance_map, cmap='hot')
        if filtered_coords:
            y_coords, x_coords = zip(*filtered_coords)
            plt.scatter(x_coords, y_coords, c='blue', s=50, 
                       marker='o', label='Начальные центры')
        
        plt.colorbar(label='Расстояние до границы')
        plt.title('Карта расстояний для инициализации', 
                 fontsize=14, fontweight='bold')
        plt.legend()
        plt.savefig(self.get_results_path('distance_map.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        return filtered_coords
    
    # =========================================================================
    # БЛОК 2: ЕДИНЫЙ ИНТЕРФЕЙС МОДУЛЯ (ДЕНЬ 1-2 ПЛАНА)
    # =========================================================================
    
    def approximate_blob(self, local_mask, N, offset_x=0, offset_y=0, 
                        ga_params=None, verbose=True):
        """
        ЕДИНЫЙ ИНТЕРФЕЙС МОДУЛЯ ДЛЯ ВКР
        
        Аппроксимация одного сложного порового кластера набором окружностей.
        Модуль применяется не ко всем порам, а только к тем, которые не 
        проходят тест на автономность по морфологическим критериям.
        
        Args:
            local_mask: локальная бинарная маска одного кластера (numpy.ndarray)
            N: количество аппроксимирующих окружностей
            offset_x: смещение по X в глобальной системе координат
            offset_y: смещение по Y в глобальной системе координат
            ga_params: параметры ГА (опционально)
            verbose: режим подробного вывода
            
        Returns:
            dict: {
                'circles': список окружностей в глобальных координатах,
                'metrics': метрики качества (IoU, fitness, время)
            }
        
        Примечание:
            Выбор кластеров, подлежащих обработке, осуществляется на 
            предыдущих этапах анализа и не входит в задачи данного исследования.
        """
        if verbose:
            print("="*60)
            print(f"🔬 АППРОКСИМАЦИЯ КЛАСТЕРА (N={N})")
            print("="*60)
        
        # Сохраняем текущие параметры
        saved_target_mask = self.target_mask
        saved_mask_height = self.mask_height
        saved_mask_width = self.mask_width
        saved_bbox = self.bbox
        saved_distance_map = self.distance_map
        saved_initial_centers = self.initial_centers
        
        # Устанавливаем локальную маску
        self.target_mask = local_mask.astype(bool)
        self.mask_height, self.mask_width = local_mask.shape
        self.height, self.width = local_mask.shape
        self.bbox = (0, 0, local_mask.shape[0], local_mask.shape[1])
        
        # Пересчитываем карту расстояний для локальной маски
        self.distance_map = distance_transform_edt(self.target_mask)
        
        # Находим начальные центры для локальной маски
        from scipy.ndimage import maximum_filter
        max_distance = np.max(self.distance_map)
        local_max = maximum_filter(self.distance_map, size=15) == self.distance_map
        local_max[self.distance_map < 0.45 * max_distance] = False
        coords = np.column_stack(np.where(local_max))
        
        filtered_coords = []
        min_distance = max(15, self.mask_width * 0.15)
        
        for coord in coords:
            if not filtered_coords:
                filtered_coords.append(coord)
                continue
            distances = np.sqrt(np.sum(
                (np.array(filtered_coords) - coord)**2, axis=1))
            if np.min(distances) > min_distance:
                filtered_coords.append(coord)
        
        self.initial_centers = filtered_coords[:N] if len(filtered_coords) >= N else filtered_coords
        
        # Создаём аппроксиматор с параметрами
        pop_size = ga_params['population_size'] if ga_params and 'population_size' in ga_params else self.population_size
        gen_count = ga_params['generations'] if ga_params and 'generations' in ga_params else self.generations
        mut_rate = ga_params['mutation_rate'] if ga_params and 'mutation_rate' in ga_params else self.mutation_rate
        cross_rate = ga_params['crossover_rate'] if ga_params and 'crossover_rate' in ga_params else self.crossover_rate
        
        approximator = CircleGeneticApproximator(
            population_size=pop_size,
            generations=gen_count,
            mutation_rate=mut_rate,
            crossover_rate=cross_rate
        )
        
        # Копируем текущие данные в аппроксиматор
        approximator.target_mask = self.target_mask
        approximator.mask_height = self.mask_height
        approximator.mask_width = self.mask_width
        approximator.distance_map = self.distance_map
        approximator.initial_centers = self.initial_centers
        approximator.results_dir = self.results_dir
        
        # Запуск оптимизации
        import time
        start_time = time.time()
        
        best_individual, fitness_history, iou_history, overlap_history, final_iou, extra_history, uncovered_history = \
            approximator.optimize_precision(
                num_circles=N,
                initial_centers=approximator.initial_centers,
                verbose=verbose
            )
        
        end_time = time.time()
        
        # Преобразование координат в глобальные
        circles_global = []
        num_circles = len(best_individual) // 3
        
        for i in range(num_circles):
            x_local, y_local, radius = best_individual[i*3:(i+1)*3]
            circles_global.append({
                'x': float(x_local + offset_x),
                'y': float(y_local + offset_y),
                'r': float(radius)
            })
        
        # Восстанавливаем сохранённые параметры
        self.target_mask = saved_target_mask
        self.mask_height = saved_mask_height
        self.mask_width = saved_mask_width
        self.bbox = saved_bbox
        self.distance_map = saved_distance_map
        self.initial_centers = saved_initial_centers
        
        # Формирование результата
        result = {
            'circles': circles_global,
            'metrics': {
                'iou': float(final_iou),
                'fitness': float(fitness_history[-1]) if fitness_history else 0,
                'runtime_seconds': float(end_time - start_time),
                'generations': len(fitness_history),
                'num_circles': num_circles,
                'final_overlap': float(overlap_history[-1]) if overlap_history else 0
            },
            'individual': best_individual,
            'histories': {
                'fitness': fitness_history,
                'iou': iou_history,
                'overlap': overlap_history,
                'extra_area': extra_history,
                'uncovered_area': uncovered_history
            }
        }
        
        if verbose:
            print(f"\n✅ РЕЗУЛЬТАТ:")
            print(f"   IoU = {final_iou:.4f}")
            print(f"   Время = {end_time - start_time:.2f} сек")
            print(f"   Окружностей = {num_circles}")
            print("="*60)
        
        return result
    
    def extract_blob_with_padding(self, binary_mask, region_label=1, 
                                  padding_ratio=0.15):
        """
        Выделение региона с bounding box и padding
        
        Args:
            binary_mask: бинарная маска изображения
            region_label: номер региона для выделения
            padding_ratio: коэффициент padding (0.1-0.2 рекомендуется)
            
        Returns:
            local_mask: локальная маска региона
            offset_x: смещение по X
            offset_y: смещение по Y
            bbox: оригинальный bounding box
        """
        labeled = measure.label(binary_mask.astype(int))
        regions = measure.regionprops(labeled)
        
        if not regions:
            raise ValueError("Не найдено связных компонент!")
        
        region = regions[region_label - 1] if region_label <= len(regions) else regions[0]
        
        min_row, min_col, max_row, max_col = region.bbox
        
        height = max_row - min_row
        width = max_col - min_col
        
        # Вычисляем padding (10-20% от размера bbox)
        padding = max(5, int(padding_ratio * max(height, width)))
        
        # Расширяем bounding box с учётом границ изображения
        padded_min_row = max(0, min_row - padding)
        padded_min_col = max(0, min_col - padding)
        padded_max_row = min(binary_mask.shape[0], max_row + padding)
        padded_max_col = min(binary_mask.shape[1], max_col + padding)
        
        # Вырезаем локальную маску
        local_mask = binary_mask[padded_min_row:padded_max_row, 
                                 padded_min_col:padded_max_col].copy()
        
        # Смещения в глобальной системе
        offset_x = padded_min_col
        offset_y = padded_min_row
        
        bbox = {
            'original': (min_row, min_col, max_row, max_col),
            'padded': (padded_min_row, padded_min_col, padded_max_row, padded_max_col),
            'padding': padding
        }
        
        return local_mask, offset_x, offset_y, bbox
    
    # =========================================================================
    # БЛОК 3: ГЕНЕТИЧЕСКИЙ АЛГОРИТМ
    # =========================================================================
    
    def create_individual_with_initialization(self, num_circles, 
                                              initial_centers=None):
        """Создает особь с умной инициализацией"""
        individual = []
        
        if num_circles == 4 and self.initial_centers is not None and len(self.initial_centers) >= 4:
            centers_to_use = self.initial_centers[:4]
        elif initial_centers is not None and len(initial_centers) >= num_circles:
            centers_to_use = initial_centers[:num_circles]
        else:
            centers_to_use = []
        
        for i in range(num_circles):
            if i < len(centers_to_use) and centers_to_use[i] is not None:
                y_coord, x_coord = centers_to_use[i]
                y_coord = int(max(0, min(y_coord, self.mask_height - 1)))
                x_coord = int(max(0, min(x_coord, self.mask_width - 1)))
                
                x = x_coord
                y = y_coord
                
                if self.distance_map is not None:
                    radius = self.distance_map[y_coord, x_coord] * 0.85
                else:
                    radius = min(self.mask_width, self.mask_height) / 4
            else:
                y_coords, x_coords = np.where(self.target_mask)
                if len(y_coords) > 0:
                    idx = np.random.randint(len(y_coords))
                    x = int(x_coords[idx])
                    y = int(y_coords[idx])
                    
                    if self.distance_map is not None:
                        local_radius = self.distance_map[y, x]
                        radius = max(5, local_radius * np.random.uniform(0.7, 1.0))
                    else:
                        radius = min(self.mask_width, self.mask_height) / 4
                else:
                    x = self.mask_width // 2
                    y = self.mask_height // 2
                    radius = min(self.mask_width, self.mask_height) / 4
            
            max_radius = min(self.mask_width, self.mask_height) / 2.5
            radius = min(radius, max_radius)
            min_radius = 25
            radius = max(radius, min_radius)
            
            individual.extend([x, y, radius])
        
        return individual
    
    def create_population(self, num_circles, initial_centers=None):
        """Создает популяцию с умной инициализацией"""
        print(f"\n🧬 СОЗДАНИЕ ПОПУЛЯЦИИ ИЗ {self.population_size} ОСОБЕЙ")
        print(f"  Количество окружностей: {num_circles}")
        
        population = []
        
        for i in range(self.population_size):
            if i < self.population_size * 0.3 and initial_centers is not None:
                individual = self.create_individual_with_initialization(
                    num_circles, initial_centers)
            else:
                individual = self.create_individual_with_initialization(num_circles)
            
            population.append(individual)
        
        print("  ✓ Популяция создана")
        return population
    
    def draw_circles(self, individual, shape=None):
        """Отрисовывает круги на маске"""
        if individual is None:
            if shape is None:
                shape = (self.mask_height, self.mask_width)
            return np.zeros(shape, dtype=bool)
        
        if shape is None:
            shape = (self.mask_height, self.mask_width)
        
        mask = np.zeros(shape, dtype=bool)
        num_circles = len(individual) // 3
        
        for i in range(num_circles):
            x, y, radius = individual[i*3:(i+1)*3]
            
            x_int = int(x)
            y_int = int(y)
            radius_int = int(radius)
            
            if radius_int > 0:
                try:
                    rr, cc = draw.disk((y_int, x_int), radius_int, shape=shape)
                    mask[rr, cc] = True
                except:
                    continue
        
        return mask
    
    def calculate_circle_overlap(self, circle1, circle2):
        """Вычисляет степень перекрытия двух кругов"""
        x1, y1, r1 = circle1
        x2, y2, r2 = circle2
        
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        if distance >= r1 + r2:
            return 0.0
        
        if distance <= abs(r1 - r2):
            smaller_radius = min(r1, r2)
            larger_radius = max(r1, r2)
            return (np.pi * smaller_radius**2) / (np.pi * larger_radius**2)
        
        d = distance
        r = r1
        R = r2
        if r > R:
            r, R = R, r
        
        part1 = r**2 * np.arccos((d**2 + r**2 - R**2) / (2 * d * r))
        part2 = R**2 * np.arccos((d**2 + R**2 - r**2) / (2 * d * R))
        part3 = 0.5 * np.sqrt((-d + r + R) * (d + r - R) * 
                              (d - r + R) * (d + r + R))
        
        intersection_area = part1 + part2 - part3
        
        smaller_area = np.pi * min(r1, r2)**2
        overlap_ratio = intersection_area / smaller_area if smaller_area > 0 else 0
        
        return overlap_ratio
    
    def fitness_function_precision(self, individual):
        """
        Функция приспособленности с штрафом за перекрытие
        
        F(C) = (1 - IoU) + α × ExtraArea + β × OverlapPenalty
        """
        if individual is None:
            return 0, 0, 0
        
        generated_mask = self.draw_circles(individual)
        
        # Основные метрики
        intersection = np.logical_and(self.target_mask, generated_mask)
        union = np.logical_or(self.target_mask, generated_mask)
        total_union = np.sum(union)
        iou = np.sum(intersection) / total_union if total_union > 0 else 0
        
        total_target_area = np.sum(self.target_mask)
        
        extra_area = np.sum(np.logical_and(generated_mask, 
                                           np.logical_not(self.target_mask)))
        uncovered_area = np.sum(np.logical_and(self.target_mask, 
                                               np.logical_not(generated_mask)))
        
        penalty_extra = 0.5 * (extra_area / total_target_area) if total_target_area > 0 else 1
        penalty_uncovered = 0.5 * (uncovered_area / total_target_area) if total_target_area > 0 else 1
        
        # Штраф за перекрытие между кругами
        num_circles = len(individual) // 3
        penalty_overlap = 0
        significant_overlaps = 0
        
        for i in range(num_circles):
            for j in range(i + 1, num_circles):
                circle1 = individual[i*3:(i+1)*3]
                circle2 = individual[j*3:(j+1)*3]
                overlap = self.calculate_circle_overlap(circle1, circle2)
                
                if overlap > 0.2:
                    penalty_overlap += overlap * 1.0
                    significant_overlaps += 1
                elif overlap > 0.1:
                    penalty_overlap += overlap * 0.5
        
        if significant_overlaps > 0:
            penalty_overlap = min(penalty_overlap, 0.8)
        
        radius_penalty = 0
        for i in range(num_circles):
            x, y, radius = individual[i*3:(i+1)*3]
            if self.distance_map is not None:
                try:
                    local_distance = self.distance_map[int(y), int(x)]
                    if radius > local_distance * 0.8:
                        radius_penalty += 0.2 * (radius / (local_distance * 0.8) - 1)
                except:
                    pass
        
        small_radius_penalty = 0
        for i in range(num_circles):
            x, y, radius = individual[i*3:(i+1)*3]
            if radius < 25:
                small_radius_penalty += (25 - radius) * 0.07
        
        triple_overlap_penalty = 0
        if num_circles >= 3:
            for i in range(num_circles):
                for j in range(i + 1, num_circles):
                    for k in range(j + 1, num_circles):
                        circle1 = individual[i*3:(i+1)*3]
                        circle2 = individual[j*3:(j+1)*3]
                        circle3 = individual[k*3:(k+1)*3]
                        
                        intersection12 = self.calculate_circle_overlap(circle1, circle2)
                        intersection13 = self.calculate_circle_overlap(circle1, circle3)
                        intersection23 = self.calculate_circle_overlap(circle2, circle3)
                        
                        if intersection12 > 0.1 and intersection13 > 0.1 and intersection23 > 0.1:
                            triple_overlap_penalty += 0.8
        
        triple_overlap_penalty *= 2.5
        
        circles_bonus = 0
        if num_circles <= 4 and iou > 0.85:
            circles_bonus = 0.15
        
        fitness = (iou * 0.85 - 
                  penalty_extra * 0.6 - 
                  penalty_uncovered * 0.6 - 
                  penalty_overlap * 0.7 - 
                  radius_penalty * 0.4 - 
                  small_radius_penalty * 1.2 - 
                  triple_overlap_penalty * 1.5 + 
                  circles_bonus)
        
        final_fitness = max(fitness, 0)
        
        return final_fitness, iou, penalty_overlap
    
    def tournament_selection_elitism(self, population, fitnesses, 
                                     tournament_size=5, elite_count=5):
        """Турнирный отбор с элитизмом"""
        selected = []
        
        elite_indices = np.argsort(fitnesses)[-elite_count:]
        elite_population = [population[i] for i in elite_indices]
        selected.extend(elite_population)
        
        for _ in range(len(population) - elite_count):
            contestants = np.random.choice(len(population), tournament_size, 
                                          replace=False)
            best_contestant = contestants[np.argmax(
                [fitnesses[i] for i in contestants])]
            selected.append(population[best_contestant])
        
        return selected
    
    def adaptive_mutation(self, individual, generation, total_generations):
        """Адаптивная мутация"""
        mutated = individual.copy()
        num_circles = len(individual) // 3
        
        adaptation_factor = 1.0 - (generation / total_generations)
        
        for i in range(num_circles):
            if np.random.random() < self.mutation_rate:
                param_index = np.random.randint(3)
                idx = i * 3 + param_index
                
                if param_index in [0, 1]:
                    mutation_strength = self.mask_width * 0.1 * adaptation_factor
                    mutated[idx] += np.random.normal(0, mutation_strength)
                    if param_index == 0:
                        mutated[idx] = np.clip(mutated[idx], 0, self.mask_width)
                    else:
                        mutated[idx] = np.clip(mutated[idx], 0, self.mask_height)
                else:
                    mutation_range = 0.2 * adaptation_factor + 0.05
                    mutated[idx] = max(25, mutated[idx] * 
                                      np.random.uniform(1 - mutation_range, 
                                                       1 + mutation_range))
        
        return mutated
    
    def crossover(self, parent1, parent2):
        """Скрещивание двух особей"""
        if np.random.random() < self.crossover_rate:
            num_circles = len(parent1) // 3
            
            if num_circles <= 1:
                return parent1.copy(), parent2.copy()
            
            circle_idx = np.random.randint(1, num_circles)
            crossover_point = circle_idx * 3
            
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
            
            return child1, child2
        
        return parent1.copy(), parent2.copy()
    
    def local_search_refinement(self, best_individual, iterations=30):
        """Локальный поиск для тонкой настройки"""
        if best_individual is None:
            return None, 0, 0
        
        best_fitness, best_iou, _ = self.fitness_function_precision(best_individual)
        current_individual = best_individual.copy()
        
        for i in range(iterations):
            new_individual = current_individual.copy()
            num_circles = len(new_individual) // 3
            
            circle_idx = np.random.randint(num_circles)
            param_idx = np.random.randint(3)
            idx = circle_idx * 3 + param_idx
            
            if param_idx in [0, 1]:
                new_individual[idx] += np.random.normal(0, 0.5)
                if param_idx == 0:
                    new_individual[idx] = np.clip(new_individual[idx], 
                                                  0, self.mask_width)
                else:
                    new_individual[idx] = np.clip(new_individual[idx], 
                                                  0, self.mask_height)
            else:
                new_radius = new_individual[idx] * np.random.uniform(0.995, 1.005)
                new_radius = max(25, new_radius)
                new_individual[idx] = new_radius
            
            new_fitness, new_iou, _ = self.fitness_function_precision(new_individual)
            
            if new_fitness > best_fitness:
                best_fitness = new_fitness
                best_iou = new_iou
                current_individual = new_individual
        
        return current_individual, best_fitness, best_iou
    
    def optimize_precision(self, num_circles, initial_centers=None, 
                          verbose=True):
        """Оптимизация с фокусом на максимальной точности"""
        if verbose:
            print(f"\n🚀 ЗАПУСК ОПТИМИЗАЦИИ")
            print(f"  Количество окружностей: {num_circles}")
            print(f"  Целевой IoU: > 0.85")
        
        start_time = time.time()
        
        population = self.create_population(num_circles, initial_centers)
        
        best_fitness = 0
        best_iou = 0
        best_individual = None
        fitness_history = []
        iou_history = []
        overlap_history = []
        extra_area_history = []
        uncovered_area_history = []
        best_overlap = 0
        
        for generation in range(self.generations):
            fitnesses = []
            ious = []
            overlaps = []
            extra_areas = []
            uncovered_areas = []
            
            for individual in population:
                fitness, iou, overlap = self.fitness_function_precision(individual)
                fitnesses.append(fitness)
                ious.append(iou)
                overlaps.append(overlap)
                # Вычисляем площади ошибок для графика
                gen_mask = self.draw_circles(individual)
                extra = np.sum(np.logical_and(gen_mask, np.logical_not(self.target_mask)))
                uncovered = np.sum(np.logical_and(self.target_mask, np.logical_not(gen_mask)))
                extra_areas.append(extra)
                uncovered_areas.append(uncovered)
            
            current_best_fitness = max(fitnesses)
            current_best_iou = max(ious)
            current_best_overlap = min(overlaps)
            current_best_extra = min(extra_areas)
            current_best_uncovered = min(uncovered_areas)
            
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_iou = current_best_iou
                best_overlap = current_best_overlap
                best_individual = population[np.argmax(fitnesses)].copy()
            
            fitness_history.append(best_fitness)
            iou_history.append(best_iou)
            overlap_history.append(best_overlap)
            extra_area_history.append(current_best_extra)
            uncovered_area_history.append(current_best_uncovered)
            
            if verbose and (generation % 25 == 0 or generation == self.generations - 1):
                avg_fitness = np.mean(fitnesses)
                print(f"   Поколение {generation:3d}/{self.generations}:  "
                      f"Лучший IoU = {best_iou:.4f},  "
                      f"Средняя приспособленность = {avg_fitness:.4f}")
            
            selected = self.tournament_selection_elitism(population, fitnesses)
            new_population = []
            
            for i in range(0, len(selected), 2):
                if i + 1 < len(selected):
                    child1, child2 = self.crossover(selected[i], selected[i+1])
                    new_population.extend([
                        self.adaptive_mutation(child1, generation, self.generations),
                        self.adaptive_mutation(child2, generation, self.generations)
                    ])
                else:
                    new_population.append(
                        self.adaptive_mutation(selected[i], generation, 
                                              self.generations))
            
            population = new_population
        
        end_time = time.time()
        
        if best_individual is not None:
            best_individual, best_fitness, best_iou = self.local_search_refinement(
                best_individual, iterations=30)
        
        if verbose:
            print(f"\n✅ ОПТИМИЗАЦИЯ ЗАВЕРШЕНА ЧЕРЕЗ {end_time - start_time:.2f} СЕКУНД")
            print(f"🎯 ФИНАЛЬНЫЙ РЕЗУЛЬТАТ: IoU = {best_iou:.4f}")
            print(f"   Количество окружностей: {num_circles}")
        
        return best_individual, fitness_history, iou_history, overlap_history, best_iou, extra_area_history, uncovered_area_history
    
    # =========================================================================
    # БЛОК 4: ИССЛЕДОВАНИЕ ПАРАМЕТРОВ (ДНИ 6-8 ПЛАНА)
    # =========================================================================
    
    def research_n_circles(self, local_mask, n_range=[2, 3, 4, 5], 
                          offset_x=0, offset_y=0):
        """
        ИССЛЕДОВАНИЕ ВЛИЯНИЯ ЧИСЛА ОКРУЖНОСТЕЙ N (ДЕНЬ 7)
        
        Args:
            local_mask: локальная бинарная маска
            n_range: диапазон исследования N
            offset_x, offset_y: смещения для глобальных координат
            
        Returns:
            dict: результаты исследования
        """
        print("="*60)
        print("🔬 ИССЛЕДОВАНИЕ ВЛИЯНИЯ ЧИСЛА ОКРУЖНОСТЕЙ (N)")
        print("="*60)
        
        results = {}
        
        for n in n_range:
            print(f"\n  🔹 N = {n}...", end=" ")
            result = self.approximate_blob(
                local_mask, n, offset_x, offset_y, verbose=False)
            results[n] = result
            print(f"IoU = {result['metrics']['iou']:.4f}")
        
        return results
    
    def test_stability(self, local_mask, N, offset_x=0, offset_y=0, 
                      num_runs=5):
        """
        ПРОВЕРКА УСТОЙЧИВОСТИ АЛГОРИТМА (ДЕНЬ 8)
        
        Args:
            local_mask: локальная бинарная маска
            N: количество окружностей
            offset_x, offset_y: смещения
            num_runs: количество повторных запусков
            
        Returns:
            dict: статистика устойчивости
        """
        print("="*60)
        print(f"🔬 ПРОВЕРКА УСТОЙЧИВОСТИ (N={N}, {num_runs} запусков)")
        print("="*60)
        
        iou_values = []
        fitness_values = []
        runtime_values = []
        
        for run in range(num_runs):
            np.random.seed(42 + run * 100)
            result = self.approximate_blob(
                local_mask, N, offset_x, offset_y, verbose=False)
            
            iou_values.append(result['metrics']['iou'])
            fitness_values.append(result['metrics']['fitness'])
            runtime_values.append(result['metrics']['runtime_seconds'])
            
            print(f"  Запуск {run+1}: IoU = {result['metrics']['iou']:.4f}")
        
        stats = {
            'iou_mean': float(np.mean(iou_values)),
            'iou_std': float(np.std(iou_values)),
            'iou_min': float(np.min(iou_values)),
            'iou_max': float(np.max(iou_values)),
            'fitness_mean': float(np.mean(fitness_values)),
            'fitness_std': float(np.std(fitness_values)),
            'runtime_mean': float(np.mean(runtime_values)),
            'all_iou_values': [float(v) for v in iou_values]
        }
        
        print(f"\n  Средний IoU: {stats['iou_mean']:.4f} ± {stats['iou_std']:.4f}")
        print(f"  Разброс: [{stats['iou_min']:.4f}, {stats['iou_max']:.4f}]")
        
        return stats
    
    def find_optimal_circles_count_precision(self, max_circles=4):
        """Поиск оптимального количества окружностей с поддержкой 1-4 пор"""
        print("\n" + "="*80)
        print("🎯 ПОИСК ОПТИМАЛЬНОГО КОЛИЧЕСТВА ОКРУЖНОСТЕЙ")
        print("="*80)
        
        self.preprocess_image_for_precision()
        
        # ИЗМЕНЕНИЕ: Начинаем с 1 окружности
        min_circles = 1
        recommended_circles = self.num_pores if self.num_pores else 2
        
        # Определяем диапазон тестирования
        max_test_circles = min(max_circles, recommended_circles + 1)
        
        print(f"\n📊 ДИАПАЗОН ТЕСТИРОВАНИЯ: от {min_circles} до {max_test_circles} окружностей")
        print(f"🎯 РЕКОМЕНДОВАННОЕ КОЛИЧЕСТВО: {recommended_circles}")
        
        best_results = {}
        best_iou = 0
        best_circles = min_circles
        
        # Явно проверяем рекомендованное количество
        if recommended_circles <= max_circles:
            print(f"\n🔍 ПРИОРИТЕТНАЯ ПРОВЕРКА: {recommended_circles} окружности")
            best_solution, fitness_history, iou_history, overlap_history, final_iou, extra_history, uncovered_history = self.optimize_precision(
                recommended_circles, 
                initial_centers=self.initial_centers,
                verbose=True
            )
            
            best_results[recommended_circles] = {
                'solution': best_solution,
                'fitness_history': fitness_history,
                'iou_history': iou_history,
                'overlap_history': overlap_history,
                'final_iou': final_iou,
                'extra_area_history': extra_history,
                'uncovered_area_history': uncovered_history
            }
            
            print(f"  📊 Результат для {recommended_circles} окружностей: IoU = {final_iou:.4f}")
            
            if final_iou > best_iou:
                best_iou = final_iou
                best_circles = recommended_circles
        
        # Тестируем остальные количества окружностей
        for num_circles in range(min_circles, max_test_circles + 1):
            if num_circles == recommended_circles:
                continue  # Уже проверили приоритетно
            
            print(f"\n" + "-"*50)
            print(f"🔍 ТЕСТИРОВАНИЕ {num_circles} ОКРУЖНОСТЕЙ")
            print("-"*50)
            
            best_solution, fitness_history, iou_history, overlap_history, final_iou, extra_history, uncovered_history = self.optimize_precision(
                num_circles, 
                initial_centers=self.initial_centers,
                verbose=True
            )
            
            best_results[num_circles] = {
                'solution': best_solution,
                'fitness_history': fitness_history,
                'iou_history': iou_history,
                'overlap_history': overlap_history,
                'final_iou': final_iou,
                'extra_area_history': extra_history,
                'uncovered_area_history': uncovered_history
            }
            
            print(f"  📊 Результат для {num_circles} окружностей: IoU = {final_iou:.4f}")
            
            if final_iou > best_iou:
                best_iou = final_iou
                best_circles = num_circles
            
            # Ранняя остановка при отличном результате
            if final_iou >= 0.9 and num_circles >= recommended_circles:
                print(f"  🎯 ОТЛИЧНЫЙ РЕЗУЛЬТАТ ДОСТИГНУТ! IoU = {final_iou:.4f}")
                break
        
        # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Удаление лишних окружностей
        best_solution = best_results[best_circles]['solution']
        num_circles = len(best_solution) // 3
        
        # Удаляем избыточные окружности
        improved_solution, was_modified = self.remove_extra_circles(best_solution, num_circles)
        
        if was_modified:
            print("🔍 ПОСТ-ОБРАБОТКА: Удалены избыточные окружности")
            # Пересчитываем IoU для улучшенного решения
            improved_mask = self.draw_circles(improved_solution)
            intersection = np.logical_and(self.target_mask, improved_mask)
            union = np.logical_or(self.target_mask, improved_mask)
            total_union = np.sum(union)
            improved_iou = np.sum(intersection) / total_union if total_union > 0 else 0
            
            print(f"  📊 Улучшенный IoU после удаления лишних окружностей: {improved_iou:.4f}")
            
            # Сохраняем улучшенное решение
            best_results[best_circles]['solution'] = improved_solution
            best_results[best_circles]['final_iou'] = improved_iou 
            
            # Если количество окружностей изменилось, обновляем best_circles
            new_num_circles = len(improved_solution) // 3
            if new_num_circles != best_circles: 
                best_circles = new_num_circles
                print(f"  🎯 Новое оптимальное количество окружностей: {best_circles}")
        
        print(f"\n🏆 ВЫБРАНО ОПТИМАЛЬНОЕ КОЛИЧЕСТВО: {best_circles} окружностей")
        print(f"   Максимальный достигнутый IoU: {best_iou:.4f}")
        
        return best_circles, best_results[best_circles]
    
    def remove_extra_circles(self, individual, num_circles):
        """
        Удаляет лишние окружности, которые полностью покрываются другими окружностями
        """
        # Создаем маску для проверки
        mask = self.draw_circles(individual)
        
        # Проверяем каждую окружность на избыточность
        circles_to_remove = []
        for i in range(num_circles):
            # Создаем маску без текущей окружности
            temp_individual = individual.copy()
            temp_individual[i*3:i*3+3] = [0, 0, 0]  # Удаляем текущую окружность
            
            # Рисуем маску без текущей окружности
            temp_mask = self.draw_circles(temp_individual)
            
            # Проверяем, покрывает ли оставшаяся маска исходную маску
            uncovered = np.logical_and(self.target_mask, np.logical_not(temp_mask))
            if np.sum(uncovered) < 0.01 * np.sum(self.target_mask):  # Если покрытие почти полное
                circles_to_remove.append(i)
        
        # Удаляем избыточные окружности
        if circles_to_remove:
            new_individual = []
            for i in range(num_circles):
                if i not in circles_to_remove:
                    new_individual.extend(individual[i*3:i*3+3])
            return new_individual, True
        
        return individual, False
    
    # =========================================================================
    # БЛОК 5: ВИЗУАЛИЗАЦИЯ
    # =========================================================================
    
    def visualize_result(self, individual, save_path=None, local_mask=None, padding=20):
        """
        Основная визуализация результатов (3 изображения)
        
        1. Исходная бинарная маска
        2. Результат аппроксимации
        3. Карта ошибки (XOR)
        
        Args:
            individual: лучшая особь
            save_path: путь для сохранения
            local_mask: локальная маска (по умолчанию self.target_mask)
            padding: отступ в пикселях вокруг изображения (по умолчанию 20)
        """
        if local_mask is None:
            local_mask = self.target_mask
        
        # Добавляем padding к маскам для красивого отображения
        padded_mask = np.pad(local_mask, pad_width=padding, mode='constant', constant_values=0)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. Исходная бинарная маска (с padding)
        axes[0].imshow(padded_mask, cmap='gray')
        axes[0].set_title('1. Исходная бинарная маска', 
                         fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # 2. Результат аппроксимации (с padding)
        approx_mask = self.draw_circles(individual, shape=local_mask.shape)
        padded_approx = np.pad(approx_mask, pad_width=padding, mode='constant', constant_values=0)
        axes[1].imshow(padded_approx, cmap='plasma')
        axes[1].set_title('2. Аппроксимация окружностями', 
                         fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # 3. Карта ошибки (XOR) (с padding)
        error_map = np.logical_xor(local_mask, approx_mask)
        padded_error = np.pad(error_map, pad_width=padding, mode='constant', constant_values=0)
        axes[2].imshow(padded_error, cmap='Reds')
        axes[2].set_title('3. Карта ошибки (XOR)', 
                         fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        iou = np.sum(np.logical_and(local_mask, approx_mask)) / \
              np.sum(np.logical_or(local_mask, approx_mask))
        
        num_circles = len(individual) // 3
        plt.suptitle(f'Результат аппроксимации ({num_circles} кругов)\nIoU: {iou:.3f}', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white')
            print(f"✓ Визуализация сохранена: {save_path}")
        
        plt.show()
        return fig
    
    def visualize_convergence(self, fitness_history, iou_history, 
                             overlap_history, extra_area_history=None, 
                             uncovered_area_history=None, save_path=None):
        """Визуализирует графики сходимости"""
        plt.figure(figsize=(14, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(fitness_history, linewidth=2.5, color='blue', alpha=0.8)
        plt.title('Сходимость функции приспособленности', 
                 fontsize=12, fontweight='bold')
        plt.xlabel('Номер поколения', fontsize=10)
        plt.ylabel('Значение приспособленности', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.plot(iou_history, linewidth=2.5, color='green', alpha=0.8)
        plt.title('Сходимость метрики IoU', 
                 fontsize=12, fontweight='bold')
        plt.xlabel('Номер поколения', fontsize=10)
        plt.ylabel('Значение IoU', fontsize=10)
        
        plt.ylim(0.65, 1.0)  # Фиксированные пределы оси Y
        plt.yticks([0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00])  # Явные метки


        plt.grid(True, alpha=0.3)
        
        # ИЗМЕНЕНИЕ: Вместо динамики перекрытия показываем непокрытую и лишнюю площадь
        plt.subplot(2, 2, 3)
        if extra_area_history is not None and uncovered_area_history is not None:
            plt.plot(extra_area_history, linewidth=2.5, color='red', alpha=0.8, label='Лишняя площадь')
            plt.plot(uncovered_area_history, linewidth=2.5, color='orange', alpha=0.8, label='Непокрытая площадь')
            plt.legend(fontsize=9)
        else:
            plt.plot(overlap_history, linewidth=2.5, color='red', alpha=0.8)
        plt.title('Динамика ошибок аппроксимации', fontsize=12, fontweight='bold')
        plt.xlabel('Номер поколения', fontsize=10)
        plt.ylabel('Нормализованная площадь', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        final_iou = iou_history[-1] if iou_history else 0

        
        # ИСПРАВЛЕНИЕ: Берём последнее значение из overlap_history, а не 0
        final_overlap = overlap_history[-1] if overlap_history and len(overlap_history) > 0 else 0
        metrics_text = f"Итоговые метрики:\nIoU: {final_iou:.4f}\nПерекрытие: {final_overlap:.4f}"
        plt.text(0.5, 0.5, metrics_text, fontsize=12, ha='center', va='center', 
                 transform=plt.gca().transAxes, 
                 bbox=dict(boxstyle="round", facecolor='lightgray'))
        plt.axis('off')
        
        plt.suptitle('Динамика обучения генетического алгоритма', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Графики сходимости сохранены: {save_path}")
        
        plt.show()
    
    def plot_iou_vs_n(self, research_results, save_path=None):
        """
        Построение графика IoU от N
        """
        plt.figure(figsize=(10, 6))
        
        n_values = sorted(research_results.keys())
        iou_values = [research_results[n]['metrics']['iou'] for n in n_values]
        
        plt.plot(n_values, iou_values, 'o-', linewidth=2, markersize=10, 
                color='blue')
        
        for n, iou in zip(n_values, iou_values):
            plt.annotate(f'{iou:.3f}', xy=(n, iou), xytext=(0, 5),
                        textcoords='offset points', fontsize=10, ha='center')
        
        plt.xlabel('Количество окружностей (N)', fontsize=12, fontweight='bold')
        plt.ylabel('IoU (Intersection over Union)', fontsize=12, fontweight='bold')
        plt.title('Зависимость качества аппроксимации от числа окружностей', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.xticks(n_values)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white')
            print(f"✓ График IoU от N сохранён: {save_path}")
        
        plt.show()
    
    # =========================================================================
    # БЛОК 6: ЭКСПОРТ ДАННЫХ
    # =========================================================================
    
    def export_parameters(self, individual, output_path, local_mask=None):
        """Экспортирует детальные параметры в JSON"""
        if local_mask is None:
            local_mask = self.target_mask
        
        num_circles = len(individual) // 3
        circles_data = []
        
        for i in range(num_circles):
            x, y, radius = individual[i*3:(i+1)*3]
            
            x_original = float(x) + self.bbox[1] if self.bbox else x
            y_original = float(y) + self.bbox[0] if self.bbox else y
            
            circle_mask = np.zeros((self.mask_height, self.mask_width), 
                                  dtype=bool)
            rr, cc = draw.disk((int(y), int(x)), int(radius), 
                              shape=(self.mask_height, self.mask_width))
            circle_mask[rr, cc] = True
            
            circle_coverage = np.sum(np.logical_and(self.target_mask, 
                                                   circle_mask)) / \
                             np.sum(circle_mask) if np.sum(circle_mask) > 0 else 0
            
            circle_info = {
                "id": i + 1,
                "center": {"x": x_original, "y": y_original},
                "radius": float(radius),
                "diameter": float(2 * radius),
                "area": float(np.pi * radius ** 2),
                "coverage_of_circle": float(circle_coverage)
            }
            circles_data.append(circle_info)
        
        approximation = self.draw_circles(individual)
        iou = np.sum(np.logical_and(self.target_mask, approximation)) / \
              np.sum(np.logical_or(self.target_mask, approximation))
        
        extra_area = np.sum(np.logical_and(approximation, 
                                          np.logical_not(self.target_mask)))
        uncovered_area = np.sum(np.logical_and(self.target_mask, 
                                              np.logical_not(approximation)))
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
        
        print(f"✓ Параметры экспортированы: {output_path}")


# =========================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# =========================================================================

def main():
    """Основная функция программы"""
    print("=" * 80)
    print("🎯 ПОДСИСТЕМА АППРОКСИМАЦИИ МОРФОЛОГИЧЕСКИХ ОБЪЕКТОВ")
    print("   Версия для ВКР с полным исследованием алгоритма")
    print("=" * 80)
    
    approximator = CircleGeneticApproximator(
        population_size=120,
        generations=200,
        mutation_rate=0.15,
        crossover_rate=0.85
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
        return
    
    try:
        choice = int(input(f"\n👉 Выберите изображение (1-{len(available_masks)}): ")) - 1
        selected_file = available_masks[choice]
    except (ValueError, IndexError):
        if available_masks:
            print("⚠️ Неверный выбор. Используется первая маска.")
            selected_file = available_masks[0]
        else:
            print("❌ Нет доступных изображений.")
            return
    
    # Загружаем изображение
    try:
        approximator.load_image(selected_file)
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        return
    
    # Создаем папку для результатов
    base_name = os.path.splitext(selected_file)[0]
    results_dir = approximator.setup_results_directory(base_name)
    
    # ИЗМЕНЕНИЕ: Запрашиваем у пользователя диапазон тестирования N
    print("\n" + "="*80)
    print("📊 НАСТРОЙКА ДИАПАЗОНА ТЕСТИРОВАНИЯ")
    print("="*80)
    
    try:
        min_n = int(input("👉 Минимальное количество окружностей для тестирования: "))
        max_n = int(input("👉 Максимальное количество окружностей для тестирования: "))
        
        # Валидация ввода
        if min_n < 1:
            print("⚠️ Минимальное значение не может быть меньше 1. Установлено 1.")
            min_n = 1
        if max_n < min_n:
            print(f"⚠️ Максимальное значение не может быть меньше минимального. Установлено {min_n}.")
            max_n = min_n
        if max_n > 10:
            print("⚠️ Максимальное значение ограничено 10 для производительности.")
            max_n = 10
            
        print(f"\n✅ Диапазон тестирования: от {min_n} до {max_n} окружностей")
    except (ValueError, KeyboardInterrupt):
        print("\n⚠️ Неверный ввод. Используем диапазон по умолчанию: 1-4")
        min_n = 1
        max_n = 4
    
    # Поиск оптимального количества окружностей
    print("\n🎯 ПОИСК ОПТИМАЛЬНОГО КОЛИЧЕСТВА ОКРУЖНОСТЕЙ...")
    
    best_iou = 0
    best_n = min_n
    best_result = None
    
    # ИЗМЕНЕНИЕ: Тестируем в указанном пользователем диапазоне
    for n in range(min_n, max_n + 1):
        result = approximator.approximate_blob(
            approximator.target_mask, n, verbose=True)
        
        if result['metrics']['iou'] > best_iou:
            best_iou = result['metrics']['iou']
            best_n = n
            best_result = result
    
    print(f"\n🏆 ОПТИМАЛЬНОЕ N = {best_n} (IoU = {best_iou:.4f})")
    
    # Визуализация
    result_image_path = approximator.get_results_path(f'{base_name}_main_result.png')
    approximator.visualize_result(
        best_result['individual'], 
        save_path=result_image_path,
        local_mask=approximator.target_mask,
        padding=20  # ИЗМЕНЕНИЕ: Добавлен padding 20 пикселей
    )
    
    # Графики сходимости
    convergence_path = approximator.get_results_path(f'{base_name}_convergence.png')
    approximator.visualize_convergence(
        best_result['histories']['fitness'],
        best_result['histories']['iou'],
        best_result['histories']['overlap'],
        best_result['histories'].get('extra_area', None),
        best_result['histories'].get('uncovered_area', None),
        save_path=convergence_path
    )
    
    # Экспорт параметров
    json_path = approximator.get_results_path(f'{base_name}_final_parameters.json')
    approximator.export_parameters(best_result['individual'], json_path)
    
    # Финальный отчет
    print("\n" + "=" * 80)
    print("🎉 АППРОКСИМАЦИЯ УСПЕШНО ЗАВЕРШЕНА!")
    print("=" * 80)
    print(f"📁 Результаты: {results_dir}")
    print(f"\n📊 КЛЮЧЕВЫЕ РЕЗУЛЬТАТЫ:")
    print(f"   Оптимальное количество окружностей: {best_n}")
    print(f"   Достигнутый IoU: {best_iou:.4f}")
    print(f"   Статус: {'🎯 IoU > 0.85' if best_iou >= 0.85 else '⚠️ Требуется проверка'}")
    print(f"\n💾 СОЗДАННЫЕ ФАЙЛЫ:")
    print(f"   📄 {base_name}_main_result.png - визуализация")
    print(f"   📄 {base_name}_convergence.png - графики сходимости")
    print(f"   📄 {base_name}_final_parameters.json - параметры")
    print("=" * 80)


if __name__ == "__main__":
    main()