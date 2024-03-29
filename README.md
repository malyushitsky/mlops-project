# Предсказание выживаемости на титанике

**Задачей проекта** является построение воспроизводимого и структурно адекватного ML проекта

В качестве датасета взят датасет по выживаемости на титанике

**ML задача** - классификация факта смерти/выживаемости на титанике

## Инициализационные параметры
Перед использованием пакета необходимо:
1. Создать виртуальную среду с версией python=3.9
2. Активировать ранее созданную среду

Команды запуска:
1. poetry install
2. pre-commit install
3. cd mlops_project
4. python train.py
5. python infer.py

## Этапы проекта
1. Обучение модели с ее сохранением
2. Инференс ранее обученной модели

### Использованные технологии
1. **Git** для контроля хранения версий проекта
2. **DVC** для версионирования датасетов, моделей и инструментов предобработки данных
3. **Hydra** для параметризации и логгирования моделей (использование **gdrive** в качестве удаленного хранилища данных)
4. **Poetry + Conda env(python=3.9)** для пакетирования и управления зависимостями в проекте
5. **Pre-commit** для автопроверки и исправления кода перед коммитами в git

### Дополнительная информация
* В проекте используется модель CatboostClassifier
* Предсказания после инференса сохраняются в predictions/test_preds.csv
* Метрики качества можно отслеживать как после обучения, так и после инференса в папке outputs
* Параметры модели можно менять из командной строки (например python train.py params.n_estimators=500)
* Проект является легко масштабируемым
