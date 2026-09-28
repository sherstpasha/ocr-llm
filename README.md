# OCR LLM

## Требования

- Python 3.11 или новее
- LM Studio с запущенным Local Server и загруженными моделями

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Для Windows PowerShell активация окружения:

```powershell
.venv\Scripts\Activate.ps1
```

Если сервер требует API-ключ, укажи его в `.env` в переменной `LLM_API_KEY`. Для локального LM Studio обычно подходит значение `dummy`.

Перед запуском проверь `configs/config.json`: адрес LM Studio, пути к данным и имена моделей.
Повторы сохраняются в `experiments/exp/repeat_1`, `repeat_2` и т. д. Параметр `experiment.reuse_ocr_results` включает копирование готовых OCR-результатов из предыдущего повтора; отсутствующие результаты распознаются заново.

## Запуск

Полный эксперимент:

```bash
python src/experiment.py --config configs/config.json
```

Остановить все запущенные процессы эксперимента:

```bash
pkill -f '[s]rc/experiment.py'
```

Отдельные этапы:

```bash
python src/ocr.py --config configs/config.json
python src/vlm.py --config configs/config.json
python src/ocr_vlm.py --config configs/config.json
python src/ocr_llm.py --config configs/config.json
python src/metrics_table.py --config configs/config.json
```
