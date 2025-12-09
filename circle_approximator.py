"""
ПОДСИСТЕМА АППРОКСИМАЦИИ КРУГЛЫХ ПОР
Умный генетический алгоритм с улучшенной системой организации результатов
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import time
import os
import datetime
from skimage import io, measure, draw
import warnings
warnings.filterwarnings('ignore')

class CircleGeneticApproximator:
    """
    Умный алгоритм для аппроксимации круглых пор с автоматическим определением минимального количества окружностей.
    """
    
    def __init__(self, population_size=100, generations=200, mutation_rate=0.1, crossover_rate=0.8):
        """
        Инициализация генетического алгоритма.
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
        self.original_image = None  # Сохраняем оригинальное изображение
        
        print("✓ Инициализация умного генетического алгоритма")
        print(f"  Размер популяции: {population_size}")
        print(f"  Количество поколений: {generations}")
    
    def setup_results_directory(self, base_name):
        """Создает улучшенную структуру папок для результатов"""
        # Основная папка с датой в формате "25.11.2025"
        date_folder = datetime.datetime.now().strftime("%d.%m.%Y")
        
        # Если папка с датой не существует, создаем ее
        if not os.path.exists(date_folder):
            os.makedirs(date_folder)
            print(f"✓ Создана папка за дату: {date_folder}")
        
        # Папка для конкретного запуска с временной меткой
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_folder = f"results_{base_name}_{timestamp}"
        
        # Полный путь к папке результатов
        self.results_dir = os.path.join(date_folder, run_folder)
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f"✓ Создана папка для результатов: {self.results_dir}")
        return self.results_dir
    
    def get_results_path(self, filename):
        """Генерирует полный путь к файлу в папке результатов"""
        if self.results_dir is None:
            raise ValueError("Ошибка: папка результатов не создана!")
        return os.path.join(self.results_dir, filename)
    
    def load_image(self, image_path):
        """Загружает и подготавливает бинарное изображение"""
        print(f"\n📁 Загрузка изображения: {image_path}")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Файл {image_path} не найден!")
        
        self.image_path = image_path
        self.original_image = io.imread(image_path)  # Сохраняем оригинал
        self.image = self.original_image.copy()
        print("✓ Изображение успешно загружено")
        
        # Преобразуем в оттенки серого если нужно
        if len(self.image.shape) == 3:
            self.image = self.image.mean(axis=2)
            print("✓ Цветное изображение преобразовано в оттенки серого")
        
        # Создаем бинарную маску
        threshold = 0.5 * np.max(self.image)
        self.binary_mask = self.image > threshold
        self.height, self.width = self.binary_mask.shape
        
        # Находим связные компоненты
        labeled_image = measure.label(self.binary_mask)
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
    
    def analyze_image_complexity(self):
        """Анализирует изображение для определения начальных параметров"""
        # Сначала проверяем имя файла для эвристической оценки
        filename = os.path.basename(self.image_path).lower()
        
        # Эвристики на основе имени файла
        if any(keyword in filename for keyword in ['single', 'one', '1', 'один', 'одна']):
            print("✓ По имени файла определено: 1 пора")
            return 1, 2
        elif any(keyword in filename for keyword in ['two', '2', 'две', 'два', 'touching']):
            print("✓ По имени файла определено: 2 поры")
            return 2, 3
        elif any(keyword in filename for keyword in ['three', '3', 'три', 'multiple']):
            print("✓ По имени файла определено: 3+ поры")
            return 3, 5
        elif any(keyword in filename for keyword in ['complex', 'many', 'multiple', 'сложн']):
            print("✓ По имени файла определено: сложная структура")
            return 3, 6
        
        # Если по имени не определили, анализируем геометрические свойства текущего изображения
        labeled = measure.label(self.target_mask)
        regions = measure.regionprops(labeled)
        
        if len(regions) == 1:
            region = regions[0]
            area = region.area
            equivalent_diameter = region.equivalent_diameter
            
            # Оцениваем сложность на основе компактности и формы
            compactness = (region.perimeter ** 2) / (4 * np.pi * area) if area > 0 else 1
            eccentricity = region.eccentricity
            
            # Анализируем соотношение сторон bounding box
            bbox = region.bbox
            bbox_height = bbox[2] - bbox[0]
            bbox_width = bbox[3] - bbox[1]
            aspect_ratio = max(bbox_width / bbox_height, bbox_height / bbox_width)
            
            print(f"✓ Геометрический анализ текущего изображения:")
            print(f"  Обнаружена 1 связная компонента")
            print(f"  Площадь: {area} пикселей")
            print(f"  Эквивалентный диаметр: {equivalent_diameter:.1f} пикселей")
            print(f"  Компактность: {compactness:.3f}")
            print(f"  Эксцентриситет: {eccentricity:.3f}")
            print(f"  Соотношение сторон: {aspect_ratio:.3f}")
            
            # Определяем начальное количество кругов на основе сложности
            if compactness < 1.2 and eccentricity < 0.3:
                # Близко к кругу - начинаем с 1
                return 1, 2
            elif aspect_ratio > 1.5 or eccentricity > 0.7:
                # Вытянутая форма - начинаем с 2
                return 2, 3
            elif compactness > 1.5:
                # Сложная форма - начинаем с 3
                return 3, 4
            else:
                # По умолчанию начинаем с 2
                return 2, 3
        else:
            # Несколько компонент - начинаем с количества компонент
            num_regions = len(regions)
            print(f"✓ Геометрический анализ:")
            print(f"  Обнаружено {num_regions} связных компонент")
            return num_regions, num_regions + 2
    
    def create_individual(self, num_circles):
        """Создает одну особь (набор кругов)"""
        individual = []
        
        for _ in range(num_circles):
            # Параметры круга: x, y, radius
            x = np.random.uniform(0, self.mask_width)
            y = np.random.uniform(0, self.mask_height)
            
            # Радиус в разумных пределах
            max_radius = min(self.mask_width, self.mask_height) / 3
            radius = np.random.uniform(5, max_radius)
            
            individual.extend([x, y, radius])
        
        return individual
    
    def create_population(self, num_circles):
        """Создает начальную популяцию"""
        return [self.create_individual(num_circles) for _ in range(self.population_size)]
    
    def draw_circles(self, individual, shape=None):
        """Отрисовывает круги на маске"""
        if shape is None:
            shape = (self.mask_height, self.mask_width)
        
        mask = np.zeros(shape, dtype=bool)
        num_circles = len(individual) // 3
        
        for i in range(num_circles):
            x, y, radius = individual[i*3:(i+1)*3]
            
            # Преобразуем в целые координаты
            x_int, y_int = int(x), int(y)
            radius_int = int(radius)
            
            if radius_int > 0:
                try:
                    # Рисуем круг
                    rr, cc = draw.disk((y_int, x_int), radius_int, shape=shape)
                    mask[rr, cc] = True
                except:
                    continue
                    
        return mask
    
    def draw_circles_on_original(self, individual):
        """Отрисовывает круги на оригинальном изображении с учетом bounding box"""
        # Создаем копию оригинального изображения
        if len(self.original_image.shape) == 3:
            result_image = self.original_image.copy()
        else:
            result_image = np.stack([self.original_image] * 3, axis=-1)
        
        num_circles = len(individual) // 3
        bbox = self.bbox  # (min_row, min_col, max_row, max_col)
        
        for i in range(num_circles):
            x, y, radius = individual[i*3:(i+1)*3]
            
            # Преобразуем координаты обратно в систему координат оригинального изображения
            x_original = int(x) + bbox[1]  # min_col
            y_original = int(y) + bbox[0]  # min_row
            radius_int = int(radius)
            
            if radius_int > 0:
                try:
                    # Рисуем красный контур круга
                    rr, cc = draw.circle_perimeter(y_original, x_original, radius_int, 
                                                 shape=result_image.shape[:2])
                    # Убедимся, что координаты в пределах изображения
                    valid = (rr >= 0) & (rr < result_image.shape[0]) & (cc >= 0) & (cc < result_image.shape[1])
                    rr, cc = rr[valid], cc[valid]
                    
                    # Рисуем красный контур
                    result_image[rr, cc, 0] = 255  # Красный канал
                    result_image[rr, cc, 1] = 0    # Зеленый канал
                    result_image[rr, cc, 2] = 0    # Синий канал
                    
                    # Добавляем номер круга
                    if (0 <= y_original < result_image.shape[0] and 
                        0 <= x_original < result_image.shape[1]):
                        # Белый текст с черной обводкой для лучшей видимости
                        text_color = [255, 255, 255]  # Белый
                        outline_color = [0, 0, 0]     # Черный
                        
                        # Рисуем обводку
                        for dx in [-1, 0, 1]:
                            for dy in [-1, 0, 1]:
                                if dx == 0 and dy == 0:
                                    continue
                                y_text = min(max(y_original + dy, 0), result_image.shape[0]-1)
                                x_text = min(max(x_original + dx, 0), result_image.shape[1]-1)
                                result_image[y_text, x_text] = outline_color
                        
                        # Рисуем текст
                        result_image[y_original, x_original] = text_color
                        
                except Exception as e:
                    print(f"⚠️ Ошибка при рисовании круга {i+1}: {e}")
                    continue
                    
        return result_image
    
    def calculate_circle_overlap(self, circle1, circle2):
        """Вычисляет степень перекрытия двух кругов"""
        x1, y1, r1 = circle1
        x2, y2, r2 = circle2
        
        # Расстояние между центрами
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Если круги не пересекаются
        if distance >= r1 + r2:
            return 0.0
        
        # Если один круг полностью внутри другого
        if distance <= abs(r1 - r2):
            smaller_radius = min(r1, r2)
            larger_radius = max(r1, r2)
            return (np.pi * smaller_radius**2) / (np.pi * larger_radius**2)
        
        # Вычисляем площадь пересечения
        d = distance
        r = r1
        R = r2
        if r > R:
            r, R = R, r
        
        part1 = r**2 * np.arccos((d**2 + r**2 - R**2) / (2 * d * r))
        part2 = R**2 * np.arccos((d**2 + R**2 - r**2) / (2 * d * R))
        part3 = 0.5 * np.sqrt((-d + r + R) * (d + r - R) * (d - r + R) * (d + r + R))
        
        intersection_area = part1 + part2 - part3
        
        # Нормализуем относительно площади меньшего круга
        smaller_area = np.pi * min(r1, r2)**2
        overlap_ratio = intersection_area / smaller_area if smaller_area > 0 else 0
        
        return overlap_ratio
    
    def fitness_function(self, individual):
        """Вычисляет приспособленность особи с учетом перекрытия кругов"""
        generated_mask = self.draw_circles(individual)
        
        # Вычисляем Intersection over Union (IoU)
        intersection = np.logical_and(self.target_mask, generated_mask)
        union = np.logical_or(self.target_mask, generated_mask)
        
        total_union = np.sum(union)
        iou = np.sum(intersection) / total_union if total_union > 0 else 0
        
        # Штрафы за ошибки
        total_target_area = np.sum(self.target_mask)
        
        # Штраф за площадь вне целевого объекта
        extra_area = np.sum(np.logical_and(generated_mask, np.logical_not(self.target_mask)))
        penalty_extra = 0.7 * (extra_area / total_target_area) if total_target_area > 0 else 1
        
        # Штраф за непокрытую площадь
        uncovered_area = np.sum(np.logical_and(self.target_mask, np.logical_not(generated_mask)))
        penalty_uncovered = 0.3 * (uncovered_area / total_target_area) if total_target_area > 0 else 1
        
        # Штраф за перекрытие между кругами
        num_circles = len(individual) // 3
        penalty_overlap = 0
        overlap_count = 0
        
        for i in range(num_circles):
            for j in range(i + 1, num_circles):
                circle1 = individual[i*3:(i+1)*3]
                circle2 = individual[j*3:(j+1)*3]
                overlap = self.calculate_circle_overlap(circle1, circle2)
                if overlap > 0.1:  # Порог значительного перекрытия
                    penalty_overlap += overlap
                    overlap_count += 1
        
        # Нормализуем штраф за перекрытие
        if overlap_count > 0:
            penalty_overlap = penalty_overlap / overlap_count * 0.5
        
        # Бонус за использование меньшего количества кругов
        circles_bonus = 0
        if num_circles == 1 and iou > 0.8:
            circles_bonus = 0.1
        elif num_circles == 2 and iou > 0.85:
            circles_bonus = 0.05
        
        # Итоговая оценка
        fitness = iou - penalty_extra - penalty_uncovered - penalty_overlap + circles_bonus
        final_fitness = max(fitness, 0)
        
        return final_fitness, iou, penalty_overlap
    
    def tournament_selection(self, population, fitnesses, tournament_size=3):
        """Турнирный отбор"""
        selected = []
        
        for _ in range(len(population)):
            contestants = np.random.choice(len(population), tournament_size, replace=False)
            best_contestant = contestants[np.argmax([fitnesses[i] for i in contestants])]
            selected.append(population[best_contestant])
        
        return selected
    
    def crossover(self, parent1, parent2):
        """Скрещивание двух особей"""
        if np.random.random() < self.crossover_rate:
            num_circles = len(parent1) // 3
            
            # Проверяем, что есть хотя бы 2 круга для скрещивания
            if num_circles <= 1:
                return parent1.copy(), parent2.copy()
            
            circle_idx = np.random.randint(1, num_circles)
            crossover_point = circle_idx * 3
            
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
            
            return child1, child2
        
        return parent1.copy(), parent2.copy()
    
    def mutate(self, individual):
        """Мутация особи"""
        mutated = individual.copy()
        num_circles = len(individual) // 3
        
        for i in range(num_circles):
            if np.random.random() < self.mutation_rate:
                param_index = np.random.randint(3)
                idx = i * 3 + param_index
                
                if param_index in [0, 1]:  # Координаты X или Y
                    mutated[idx] += np.random.normal(0, self.mask_width * 0.1)
                    if param_index == 0:  # X
                        mutated[idx] = np.clip(mutated[idx], 0, self.mask_width)
                    else:  # Y
                        mutated[idx] = np.clip(mutated[idx], 0, self.mask_height)
                else:  # Радиус
                    mutated[idx] = max(5, mutated[idx] * np.random.uniform(0.8, 1.2))
                    
        return mutated
    
    def optimize(self, num_circles, verbose=True):
        """Запускает генетический алгоритм"""
        if verbose:
            print(f"\n🔧 Запуск оптимизации для {num_circles} кругов...")
        
        start_time = time.time()
        
        # Создаем начальную популяцию
        population = self.create_population(num_circles)
        
        best_fitness = 0
        best_iou = 0
        best_overlap = 0
        best_individual = None
        fitness_history = []
        iou_history = []
        overlap_history = []
        
        # Основной цикл генетического алгоритма
        for generation in range(self.generations):
            # Оцениваем приспособленность
            fitnesses = []
            ious = []
            overlaps = []
            
            for individual in population:
                fitness, iou, overlap = self.fitness_function(individual)
                fitnesses.append(fitness)
                ious.append(iou)
                overlaps.append(overlap)
            
            # Находим лучшую особь
            current_best_fitness = max(fitnesses)
            current_best_iou = max(ious)
            current_best_overlap = min(overlaps)
            
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_iou = current_best_iou
                best_overlap = current_best_overlap
                best_individual = population[np.argmax(fitnesses)].copy()
            
            fitness_history.append(best_fitness)
            iou_history.append(best_iou)
            overlap_history.append(best_overlap)
            
            if verbose and generation % 20 == 0:
                avg_fitness = np.mean(fitnesses)
                print(f"   Поколение {generation:3d}: "
                      f"Лучшая приспособленность = {best_fitness:.4f}, "
                      f"Лучший IoU = {best_iou:.4f}, "
                      f"Перекрытие = {best_overlap:.4f}")
            
            # Отбор, скрещивание и мутация
            selected = self.tournament_selection(population, fitnesses)
            new_population = []
            
            for i in range(0, len(selected), 2):
                if i + 1 < len(selected):
                    child1, child2 = self.crossover(selected[i], selected[i+1])
                    new_population.extend([self.mutate(child1), self.mutate(child2)])
                else:
                    new_population.append(self.mutate(selected[i]))
            
            population = new_population
        
        end_time = time.time()
        
        if verbose:
            print(f"\n✓ Оптимизация завершена за {end_time - start_time:.2f} секунд")
            print(f"✓ Лучшая приспособленность: {best_fitness:.4f}")
            print(f"✓ Лучший IoU: {best_iou:.4f}")
            print(f"✓ Перекрытие кругов: {best_overlap:.4f}")
        
        return best_individual, fitness_history, iou_history, overlap_history
    
    def evaluate_solution_quality(self, individual):
        """Оценивает качество решения и определяет, нужно ли больше кругов"""
        num_circles = len(individual) // 3
        fitness, iou, overlap = self.fitness_function(individual)
        
        # Анализируем покрытие
        generated_mask = self.draw_circles(individual)
        uncovered_area = np.sum(np.logical_and(self.target_mask, np.logical_not(generated_mask)))
        total_target_area = np.sum(self.target_mask)
        coverage = 1 - (uncovered_area / total_target_area) if total_target_area > 0 else 0
        
        print(f"  Анализ решения с {num_circles} кругами:")
        print(f"  - IoU: {iou:.4f}")
        print(f"  - Покрытие: {coverage:.4f}")
        print(f"  - Перекрытие кругов: {overlap:.4f}")
        
        # Определяем, достаточно ли кругов
        if iou >= 0.95 and overlap < 0.1:
            # Отличное решение - скорее всего, кругов достаточно
            return "excellent", iou
        elif iou >= 0.90 and overlap < 0.2:
            # Хорошее решение - возможно, кругов достаточно
            return "good", iou
        elif iou >= 0.85 and overlap < 0.3:
            # Приемлемое решение - возможно, нужно больше кругов
            return "acceptable", iou
        else:
            # Плохое решение - определенно нужно больше кругов
            return "poor", iou
    
    def find_optimal_circles_count(self, max_circles=5):
        """Умный поиск оптимального количества кругов"""
        print("\n" + "="*70)
        print("🔍 УМНЫЙ ПОИСК ОПТИМАЛЬНОГО КОЛИЧЕСТВА КРУГОВ")
        print("="*70)
        
        # Анализируем изображение для определения начального количества
        initial_circles, max_test_circles = self.analyze_image_complexity()
        max_test_circles = min(max_test_circles, max_circles)
        
        print(f"  Начинаем поиск с {initial_circles} круга(ов)")
        print(f"  Максимальное количество для тестирования: {max_test_circles}")
        
        best_results = {}
        tested_counts = []
        
        # Тестируем разное количество кругов
        for num_circles in range(initial_circles, max_test_circles + 1):
            print(f"\n--- 📊 Тестируем {num_circles} круг(ов) ---")
            
            best_solution, fitness_history, iou_history, overlap_history = self.optimize(
                num_circles, verbose=False
            )
            
            quality, iou = self.evaluate_solution_quality(best_solution)
            
            best_results[num_circles] = {
                'solution': best_solution,
                'fitness_history': fitness_history,
                'iou_history': iou_history,
                'overlap_history': overlap_history,
                'quality': quality,
                'iou': iou
            }
            
            tested_counts.append(num_circles)
            
            # Если достигли отличного качества, останавливаемся
            if quality == "excellent":
                print(f"  🎯 Достигнуто отличное качество с {num_circles} кругами!")
                optimal_circles = num_circles
                break
            
            # Если уже тестировали несколько вариантов и качество ухудшается, останавливаемся
            if len(tested_counts) >= 2:
                prev_quality = best_results[tested_counts[-2]]['quality']
                if quality == "poor" and prev_quality in ["good", "excellent"]:
                    print(f"  ⚠️  Качество ухудшилось, останавливаем поиск")
                    optimal_circles = tested_counts[-2]  # Берем предыдущий хороший результат
                    break
        else:
            # Если дошли до конца цикла, выбираем лучший по IoU
            if best_results:
                best_circles = max(best_results.keys(), key=lambda k: best_results[k]['iou'])
                optimal_circles = best_circles
                best_iou = best_results[best_circles]['iou']
                print(f"\nℹ️  Используем лучшее количество кругов: {optimal_circles} (IoU = {best_iou:.4f})")
            else:
                optimal_circles = initial_circles
                print(f"\nℹ️  Используем начальное количество кругов: {optimal_circles}")
        
        optimal_results = best_results[optimal_circles]
        
        print(f"\n✅ ОПТИМАЛЬНОЕ КОЛИЧЕСТВО КРУГОВ: {optimal_circles}")
        print(f"✅ Качество аппроксимации: {optimal_results['quality']}")
        print(f"✅ IoU: {optimal_results['iou']:.4f}")
        
        return optimal_circles, optimal_results
    
    def visualize_result(self, individual, save_path=None):
        """Визуализирует результаты аппроксимации"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. Исходное изображение
        axes[0, 0].imshow(self.original_image, cmap='gray' if len(self.original_image.shape) == 2 else None)
        axes[0, 0].set_title('Исходное изображение', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # 2. Аппроксимация кругами
        approximation = self.draw_circles(individual)
        axes[0, 1].imshow(approximation, cmap='gray')
        axes[0, 1].set_title('Аппроксимация кругами', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        # 3. Наложение на оригинальное изображение
        result_with_circles = self.draw_circles_on_original(individual)
        axes[1, 0].imshow(result_with_circles)
        axes[1, 0].set_title('Круги на исходном изображении', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        
        # 4. Области различий
        difference = np.logical_xor(self.target_mask, approximation)
        axes[1, 1].imshow(difference, cmap='Reds')
        axes[1, 1].set_title('Области различий (ошибки)', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        
        # Вычисляем метрики качества
        iou = np.sum(np.logical_and(self.target_mask, approximation)) / \
              np.sum(np.logical_or(self.target_mask, approximation))
        
        num_circles = len(individual) // 3
        plt.suptitle(
            f'Результат аппроксимации ({num_circles} кругов)\nIoU: {iou:.3f}', 
            fontsize=16, 
            fontweight='bold',
            y=0.95
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✓ Визуализация сохранена как {save_path}")
        
        plt.show()
    
    def export_parameters(self, individual, output_path):
        """Экспортирует параметры кругов в JSON файл"""
        num_circles = len(individual) // 3
        circles_data = []
        
        for i in range(num_circles):
            x, y, radius = individual[i*3:(i+1)*3]
            
            # Преобразуем координаты обратно в систему координат оригинального изображения
            x_original = float(x) + self.bbox[1]  # min_col
            y_original = float(y) + self.bbox[0]  # min_row
            
            circle_info = {
                "id": i + 1,
                "center": {"x": x_original, "y": y_original},
                "radius": float(radius),
                "diameter": float(2 * radius),
                "area": float(np.pi * radius ** 2)
            }
            circles_data.append(circle_info)
        
        # Вычисляем общие метрики
        approximation = self.draw_circles(individual)
        iou = np.sum(np.logical_and(self.target_mask, approximation)) / \
              np.sum(np.logical_or(self.target_mask, approximation))
        
        result = {
            "image_info": {
                "width": self.width,
                "height": self.height,
                "original_area": int(np.sum(self.target_mask))
            },
            "approximation_metrics": {
                "number_of_circles": num_circles,
                "iou_score": float(iou),
                "fitness_score": float(self.fitness_function(individual)[0])
            },
            "circles": circles_data
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Параметры кругов экспортированы в {output_path}")


def show_recent_results():
    """Показывает недавние папки с результатами, сгруппированные по датам"""
    print("\n📂 СТРУКТУРА РЕЗУЛЬТАТОВ:")
    
    # Ищем папки с датами (в формате DD.MM.YYYY)
    date_folders = [f for f in os.listdir('.') 
                   if os.path.isdir(f) and 
                   len(f.split('.')) == 3 and 
                   all(part.isdigit() for part in f.split('.'))]
    
    if not date_folders:
        print("   (пока нет сохраненных результатов)")
        return
    
    # Сортируем папки по дате (новые сначала)
    date_folders.sort(key=lambda x: datetime.datetime.strptime(x, "%d.%m.%Y"), reverse=True)
    
    for date_folder in date_folders[:3]:  # Показываем только последние 3 даты
        print(f"\n📅 {date_folder}:")
        
        # Ищем папки с результатами внутри папки с датой
        results_in_date = [f for f in os.listdir(date_folder) 
                          if os.path.isdir(os.path.join(date_folder, f)) and 
                          f.startswith('results_')]
        
        if results_in_date:
            # Сортируем по времени создания (новые сначала)
            results_in_date.sort(key=lambda x: os.path.getctime(os.path.join(date_folder, x)), reverse=True)
            
            for result_folder in results_in_date[:5]:  # Показываем до 5 последних запусков
                full_path = os.path.join(date_folder, result_folder)
                creation_time = datetime.datetime.fromtimestamp(os.path.getctime(full_path))
                time_str = creation_time.strftime("%H:%M:%S")
                
                # Извлекаем базовое имя из названия папки
                base_name = result_folder.replace('results_', '').split('_')[0]
                print(f"   • {base_name} ({time_str}) - {result_folder}")
        else:
            print("   (нет результатов за эту дату)")


def show_available_images():
    """Показывает доступные изображения и предлагает добавить новые"""
    available_masks = [
        f for f in os.listdir('.') 
        if f.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp')) 
        and 'preview' not in f
    ]
    
    print("\n📋 ДОСТУПНЫЕ ИЗОБРАЖЕНИЯ:")
    if available_masks:
        for i, mask in enumerate(available_masks, 1):
            print(f"   {i}. {mask}")
    else:
        print("   (нет доступных изображений)")
    
    print(f"   {len(available_masks) + 1}. 📁 ЗАГРУЗИТЬ НОВЫЙ ФАЙЛ")
    
    return available_masks


def load_custom_image():
    """Позволяет пользователю загрузить произвольный файл"""
    print("\n📁 ЗАГРУЗКА НОВОГО ФАЙЛА")
    print("Доступные файлы в текущей директории:")
    
    all_files = [f for f in os.listdir('.') if os.path.isfile(f)]
    image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'))]
    
    for i, file in enumerate(image_files, 1):
        print(f"   {i}. {file}")
    
    if not image_files:
        print("   (нет файлов изображений)")
        return None
    
    try:
        choice = int(input(f"\n👉 Выберите файл для загрузки (1-{len(image_files)}): ")) - 1
        selected_file = image_files[choice]
        return selected_file
    except (ValueError, IndexError):
        print("⚠️  Неверный выбор.")
        return None


def main():
    """Основная функция программы"""
    print("=" * 70)
    print("🎯 УМНАЯ ПОДСИСТЕМА АППРОКСИМАЦИИ КРУГЛЫХ ПОР")
    print("   Улучшенная система организации результатов")
    print("=" * 70)
    
    # Показываем недавние результаты
    show_recent_results()
    
    # Инициализируем аппроксиматор
    approximator = CircleGeneticApproximator(
        population_size=80,
        generations=150,
        mutation_rate=0.15,
        crossover_rate=0.7
    )
    
    # Показываем доступные изображения и предлагаем выбор
    available_masks = show_available_images()
    
    try:
        choice = int(input(f"\n👉 Выберите действие (1-{len(available_masks) + 1}): "))
        
        if choice == len(available_masks) + 1:
            # Загрузка нового файла
            selected_file = load_custom_image()
            if selected_file is None:
                print("❌ Не удалось загрузить файл.")
                return
        else:
            # Выбор существующего файла
            selected_file = available_masks[choice - 1]
            
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
    
    # Создаем папку для результатов с новой структурой
    base_name = os.path.splitext(selected_file)[0]
    results_dir = approximator.setup_results_directory(base_name)
    
    # Умный поиск оптимального количества кругов
    print("\n🎯 Начинаем умный поиск оптимального количества кругов...")
    optimal_circles, optimal_results = approximator.find_optimal_circles_count(max_circles=5)
    
    # Запускаем финальную оптимизацию
    print(f"\n🚀 Запуск финальной оптимизации для {optimal_circles} кругов...")
    best_solution, fitness_history, iou_history, overlap_history = approximator.optimize(optimal_circles)
    
    # Визуализируем и сохраняем результаты
    result_image_path = approximator.get_results_path(f'{base_name}_result.png')
    approximator.visualize_result(best_solution, save_path=result_image_path)
    
    # Экспортируем параметры
    json_path = approximator.get_results_path(f'{base_name}_parameters.json')
    approximator.export_parameters(best_solution, json_path)
    
    # Создаем графики сходимости
    convergence_path = approximator.get_results_path(f'{base_name}_convergence.png')
    
    plt.figure(figsize=(14, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(fitness_history, linewidth=2.5, color='blue', alpha=0.8)
    plt.title('Сходимость функции приспособленности', fontsize=12, fontweight='bold')
    plt.xlabel('Номер поколения', fontsize=10)
    plt.ylabel('Значение приспособленности', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(iou_history, linewidth=2.5, color='green', alpha=0.8)
    plt.title('Сходимость метрики IoU', fontsize=12, fontweight='bold')
    plt.xlabel('Номер поколения', fontsize=10)
    plt.ylabel('Значение IoU', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.plot(overlap_history, linewidth=2.5, color='red', alpha=0.8)
    plt.title('Динамика перекрытия кругов', fontsize=12, fontweight='bold')
    plt.xlabel('Номер поколения', fontsize=10)
    plt.ylabel('Степень перекрытия', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    # Показываем итоговые метрики
    final_iou = iou_history[-1] if iou_history else 0
    final_overlap = overlap_history[-1] if overlap_history else 0
    metrics_text = f"Итоговые метрики:\nIoU: {final_iou:.4f}\nПерекрытие: {final_overlap:.4f}\nКругов: {optimal_circles}"
    plt.text(0.5, 0.5, metrics_text, fontsize=12, ha='center', va='center', 
             transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='lightgray'))
    plt.axis('off')
    
    plt.suptitle('Динамика обучения генетического алгоритма', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(convergence_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # Финальный отчет
    print("\n" + "=" * 70)
    print("🎉 АППРОКСИМАЦИЯ УСПЕШНО ЗАВЕРШЕНА!")
    print("=" * 70)
    print(f"📁 Все результаты сохранены в папке: {results_dir}")
    print(f"\n📊 ОСНОВНЫЕ РЕЗУЛЬТАТЫ:")
    print(f"   Оптимальное количество кругов: {optimal_circles}")
    print(f"   Качество аппроксимации (IoU): {optimal_results['iou']:.4f}")
    print(f"   Качество решения: {optimal_results['quality']}")
    print(f"\n💾 СОЗДАННЫЕ ФАЙЛЫ:")
    print(f"   📄 {base_name}_result.png - визуализация результатов")
    print(f"   📄 {base_name}_parameters.json - параметры кругов")
    print(f"   📄 {base_name}_convergence.png - графики сходимости")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()