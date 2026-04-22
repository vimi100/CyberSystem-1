# Лабораторная работа №1: Классификация стадий болезни Альцгеймера по МРТ

## Описание работы

В работе исследуются модели классификации изображений для определения стадии болезни Альцгеймера по MRI-снимкам головного мозга. Цель — автоматически относить изображение к одной из стадий когнитивных нарушений и сравнить качество готовых предобученных моделей с собственной реализацией нейросети.

**Датасет:** [Augmented Alzheimer MRI Dataset (Kaggle)](https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset)

В исследовании использовались 4 класса:
- `NonDemented`
- `VeryMildDemented`
- `MildDemented`
- `ModerateDemented`

Для честного baseline использовался `OriginalDataset`, а дополнительные техники балансировки и аугментации проверялись отдельными экспериментами.

## Выполненные задачи

- Подключение датасета через `Kaggle API` в `Google Colab`
- Анализ структуры датасета и распределения классов
- Разделение данных на `train / validation / test` со стратификацией
- Подготовка `PyTorch Dataset` и `DataLoader`
- Обучение baseline-моделей из `torchvision`:
  - `ResNet18`
  - `EfficientNet_B0`
  - `ViT_B_16`
- Проверка гипотез улучшения бейзлайна:
  - аугментации + `class weights`
  - `WeightedRandomSampler`
  - обучение только классификационной головы
- Самостоятельная реализация модели `CustomCNN`
- Обучение собственной модели в базовой и улучшенной конфигурациях
- Сравнение моделей по метрикам `Accuracy`, `Macro F1-score`, `Confusion Matrix`

## Установка и запуск в Google Colab

1. Открыть ноутбук `classification_lab.ipynb` в Google Colab
2. Включить GPU: `Среда выполнения` → `Сменить среду выполнения` → `T4 GPU`
3. Выполнить ячейки последовательно
4. При подключении Kaggle API загрузить файл `kaggle.json`

## Установка и запуск на локальной машине

1. Перейти в папку проекта:
```bash
cd ml(к примеру)
```

2. Подготовить виртуальное окружение. Для Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

Для Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Установить зависимости:
```bash
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install kaggle scikit-learn pandas matplotlib seaborn pillow jupyter
```

4. Запустить Jupyter Notebook:
```bash
jupyter notebook
```

5. Открыть `classification_lab.ipynb` и выполнить все ячейки последовательно

## Основные результаты

| Модель | Конфигурация | Accuracy | Macro F1 |
|--------|-------------|----------|----------|
| ResNet18 | Baseline | **0.9500** | **0.9593** |
| EfficientNet_B0 | Baseline | 0.9260 | 0.9346 |
| ResNet18 | WeightedRandomSampler | 0.8823 | 0.9134 |
| ResNet18 | Augmentations + Class Weights | 0.8323 | 0.8540 |
| ViT_B_16 | Baseline | 0.8448 | 0.7592 |
| ResNet18 | Head Only | 0.6156 | 0.3764 |
| CustomCNN | Improved (`WeightedRandomSampler`) | 0.4885 | 0.5672 |
| CustomCNN | Baseline | 0.5698 | 0.3235 |

## Выводы

Лучший результат показала модель `ResNet18` в базовой конфигурации (`Accuracy = 0.9500`, `Macro F1 = 0.9593`). Предобученные модели из `torchvision` значительно превзошли самостоятельно реализованную `CustomCNN`, что объясняется преимуществами transfer learning и более сильных архитектур.

Проверка гипотез показала, что не каждое усложнение пайплайна приводит к улучшению качества. В частности, `WeightedRandomSampler`, `class weights` и обучение только головы модели не смогли превзойти исходный baseline `ResNet18`. Для собственной `CustomCNN` применение техник балансировки улучшило `Macro F1`, но не позволило приблизиться к качеству лучших библиотечных моделей.
