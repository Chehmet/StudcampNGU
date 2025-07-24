import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import torch
import os

# --- КОНФИГУРАЦИЯ ---
# 1. Создаем словарь с путями к вашим трем моделям.
#    Ключ - это название, которое увидит пользователь.
#    Значение - это путь к папке с файлами модели.
#    !!! УБЕДИТЕСЬ, ЧТО ЭТИ ПУТИ ВЕРНЫЕ !!!
MODELS = {
    "Бейс (Base)": "models/best_model",
    "ИРМ (IRM)": "models/trained_model_irm",
    "БИРМ (BIRM)": "models/trained_model_birm/trained_model_birm",
}
# Определяем устройство: GPU, если доступно, иначе CPU
DEVICE = 0 if torch.cuda.is_available() else -1

# --- CSS Стили для эстетики (без изменений) ---
page_style = """
<style>
/* Основной фон и шрифт */
body, .main {
    background-color: #2c2f38;
    color: #e0e0e0;
}
h1, h2, h3 {
    color: #6c73ff;
    font-family: 'Segoe UI', sans-serif;
}
/* Стилизация радио-кнопок на боковой панели */
.stRadio [role=radiogroup]{
    align-items: stretch;
    border-radius: 10px;
    padding: 10px;
    background-color: #34374f;
}
.stRadio [role=radio] {
    margin: 5px;
}

/* Стилизация текстового поля для ввода */
.stTextArea textarea {
    background-color: #34374f;
    color: #e0e0e0;
    border: 1px solid #4a4e69;
    border-radius: 10px;
    font-family: 'Consolas', 'Menlo', 'monospace';
    font-size: 16px;
    padding: 15px;
}

/* Стилизация кнопок */
.stButton>button {
    background-color: #6c73ff;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
    font-weight: bold;
    width: 100%; /* Кнопка на всю ширину */
}
.stButton>button:hover {
    background-color: #5a61e0;
    color: white;
}

/* Стилизация вывода */
.result-box {
    background-color: #34374f;
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid #6c73ff;
    margin-top: 20px;
    font-family: 'Consolas', 'Menlo', 'monospace';
    font-size: 16px;
    line-height: 1.6;
    white-space: pre-wrap; /* Сохраняет переносы строк */
    word-wrap: break-word; /* Переносит длинные слова */
}

/* Стилизация найденных сущностей */
.entity-table {
    margin-top: 25px;
}
.entity-table th {
    background-color: #4a4e69;
    color: #e0e0e0;
    text-align: left;
}
.entity-table td {
    border-bottom: 1px solid #4a4e69;
}
</style>
"""

# --- Кэширование модели для быстрой загрузки ---
# Streamlit будет кэшировать результат для каждого уникального `path`
@st.cache_resource
def load_model(path):
    """Загружает модель и токенизатор по указанному пути и кэширует их."""
    st.info(f"Загрузка модели из папки: '{path}'...")
    # Проверяем, существует ли путь
    if not os.path.isdir(path):
        st.error(f"Ошибка: Папка с моделью не найдена по пути '{path}'.")
        st.warning("Пожалуйста, проверьте пути в словаре MODELS в файле app.py.")
        return None
    try:
        model = AutoModelForTokenClassification.from_pretrained(path)
        tokenizer = AutoTokenizer.from_pretrained(path)
        pii_pipeline = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=DEVICE,
        )
        st.success(f"Модель из '{path}' успешно загружена.")
        return pii_pipeline
    except Exception as e:
        st.error(f"Ошибка загрузки модели из '{path}': {e}")
        st.warning("Убедитесь, что в папке есть все необходимые файлы (config.json, model.safetensors, tokenizer.json и т.д.)")
        return None

# --- Основная функция для маскирования ---
def mask_log(pii_pipeline, text):
    """Принимает текст, находит PII и возвращает замаскированный текст и список найденных сущностей."""
    if not text:
        return "", []

    entities = pii_pipeline(text)
    sorted_entities = sorted(entities, key=lambda e: e["start"], reverse=True)
    
    masked_text = text
    found_pii = []
    
    for entity in sorted_entities:
        label = entity["entity_group"]
        value = entity["word"]
        start = entity["start"]
        end = entity["end"]
        
        # Создаем маску в верхнем регистре, например [PASSWORD]
        mask = f"[{label.upper()}]"
        
        masked_text = masked_text[:start] + mask + masked_text[end:]
        found_pii.append({"Тип данных": label.upper(), "Найденное значение": value})
        
    return masked_text, sorted(found_pii, key=lambda x: text.find(x["Найденное значение"]))


# --- ИНТЕРФЕЙС STREAMLIT ---

# Устанавливаем конфигурацию страницы
st.set_page_config(layout="wide", page_title="PII Детектор")

# Применяем наши стили
st.markdown(page_style, unsafe_allow_html=True)

# Заголовок
st.title("Детектор персональных данных в логах")

# --- Боковая панель для настроек ---
st.sidebar.header("⚙️ Настройки")
selected_model_name = st.sidebar.radio(
    "Выберите модель для анализа:",
    options=list(MODELS.keys()), # Опции - это названия моделей
    index=0 # Модель "Бейс" будет выбрана по умолчанию
)

# Получаем путь к выбранной модели
model_path = MODELS[selected_model_name]

# Загружаем выбранную модель (результат будет взят из кэша, если модель уже загружалась)
pii_pipeline = load_model(model_path)


# --- Основная рабочая область ---
col1, col2 = st.columns([2, 3]) # Делим экран на две колонки

with col1:
    st.subheader(f"Выбрана модель: **{selected_model_name}**")
    
    log_input = st.text_area(
        "📝 Вставьте ваш лог сюда:",
        height=400,
        placeholder="Например: 2025-05-22 13:42:14 ERROR User request processed, email franklinjames@example.net, card 4851961064628792322..."
    )

    # Кнопка для запуска анализа
    if st.button("🔍 Проанализировать и замаскировать"):
        if pii_pipeline and log_input:
            # Если кнопка нажата, результат будет в правой колонке
            pass
        elif not log_input:
            st.warning("Пожалуйста, введите текст для анализа.")
        else:
            # Если модель не загрузилась
            st.error("Модель не загружена. Проверьте сообщения об ошибках выше.")

with col2:
    st.subheader("🔒 Результат анализа")
    # Эта часть будет обновляться только после нажатия кнопки
    if pii_pipeline and log_input and 'masked_text' not in st.session_state:
        st.session_state.masked_text = "Здесь появится замаскированный лог..."
        st.session_state.found_pii = []
        
    if st.button("обработать", key="process_button_hidden_trigger"): # Хитрый способ для обновления
        with st.spinner(f"Анализирую с помощью модели '{selected_model_name}'..."):
            masked_text, found_pii = mask_log(pii_pipeline, log_input)
            st.session_state.masked_text = masked_text
            st.session_state.found_pii = found_pii
    
    # Всегда отображаем результат из st.session_state
    if 'masked_text' in st.session_state:
        st.markdown(f'<div class="result-box">{st.session_state.masked_text}</div>', unsafe_allow_html=True)
        
        if st.session_state.found_pii:
            st.subheader("🔎 Найденные и скрытые данные")
            st.dataframe(st.session_state.found_pii, use_container_width=True, hide_index=True)
        # Убираем сообщение об успехе, если ничего не найдено, для чистоты интерфейса

st.sidebar.markdown("---")
st.sidebar.info("Это приложение позволяет сравнивать разные модели для распознавания PII.")