# Специальные виды свёрток

Учебный эксперимент по сравнению четырёх пространственных операторов в одной
VGG-подобной CNN на CIFAR-10:

- обычная свёртка 3×3;
- dilated convolution, `d=2`;
- depthwise separable convolution;
- deformable convolution, DCNv1.

Презентация: [`deliverables/special_convolutions_cifar10_v2.pptx`](deliverables/special_convolutions_cifar10_v2.pptx).

## Результаты

| Оператор | Test accuracy | Параметры | MACs на изображение |
|---|---:|---:|---:|
| Обычная 3×3 | 83,69% | 98 970 | 14,60 млн |
| Dilated, `d=2` | 83,57% | 98 970 | 14,60 млн |
| Depthwise separable | 74,71% | 13 962 | 2,27 млн |
| Deformable, DCNv1 | 85,06% | 117 392 | 19,24 млн |

Все модели обучались 50 эпох на одинаковом split: 45 000 train, 5 000
validation, 10 000 test. Для каждого варианта выполнен один запуск с `seed=42`.
Поэтому результаты показывают конкретный запуск и не содержат доверительных
интервалов.

## Структура

```text
src/            обучение и измерение latency
scripts/        создание иллюстрации операторов на MNIST
assets/         иллюстрации для презентации
outputs_full/   таблицы, графики и визуализации полного запуска
research/       источники
deliverables/   итоговая презентация
```

## Повторный запуск

Требуется Python 3.12.

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python src/experiment.py \
  --output-dir outputs_full \
  --epochs 50 \
  --val-size 5000 \
  --batch-size 128 \
  --seed 42
```

Для создания иллюстрации на MNIST:

```bash
.venv/bin/python scripts/make_operator_demo.py
```

Исходные публикации и документация перечислены в
[`research/SOURCES.md`](research/SOURCES.md).
