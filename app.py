"""
Streamlit-приложение: Рекомендательная система модных товаров
Fashion Product Recommendation System

Запуск:
    streamlit run app.py

Требования:
    pip install streamlit numpy pandas scikit-learn Pillow tensorflow
"""

import os
import io
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import normalize

# ─────────────────────────────────────────────────────────────
# Конфигурация страницы
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fashion Recommender",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Кастомный CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #888;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .product-card {
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        padding: 10px;
        text-align: center;
        background: #fafafa;
        transition: box-shadow 0.2s;
    }
    .product-card:hover {
        box-shadow: 0 4px 16px rgba(102,126,234,0.15);
    }
    .query-card {
        border-radius: 12px;
        border: 2px solid #667eea;
        padding: 10px;
        text-align: center;
        background: #f0f0ff;
    }
    .tag {
        display: inline-block;
        background: #667eea22;
        color: #667eea;
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 0.8rem;
        margin: 2px;
    }
    .metric-box {
        background: linear-gradient(135deg, #667eea22, #764ba222);
        border-radius: 10px;
        padding: 12px 16px;
        text-align: center;
    }
    .stSelectbox label, .stSlider label { font-weight: 600; }
    div[data-testid="stSidebar"] { background: #f8f9ff; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Загрузка данных (кэшируется)
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_styles(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, on_bad_lines='skip')
    if 'image_path' not in df.columns:
        # Пробуем восстановить путь к изображениям
        base = os.path.dirname(csv_path)
        imgs = os.path.join(base, "images")
        df['image_path'] = df['id'].apply(lambda x: os.path.join(imgs, f"{x}.jpg"))
    df['image_exists'] = df['image_path'].apply(os.path.exists)
    return df[df['image_exists']].reset_index(drop=True)


@st.cache_data
def load_features(features_path: str) -> np.ndarray:
    return np.load(features_path)


@st.cache_data
def load_model_info(info_path: str) -> dict:
    with open(info_path, 'rb') as f:
        return pickle.load(f)


def load_image_safe(path: str, size=(200, 200)) -> Image.Image:
    try:
        img = Image.open(path).convert('RGB')
        img.thumbnail(size, Image.LANCZOS)
        return img
    except Exception:
        placeholder = Image.new('RGB', size, color=(220, 220, 230))
        return placeholder


# ─────────────────────────────────────────────────────────────
# Рекомендательные функции
# ─────────────────────────────────────────────────────────────
def get_recommendations_cosine(query_idx: int, features: np.ndarray, n: int = 5):
    query_vec = features[query_idx].reshape(1, -1)
    sims = cosine_similarity(query_vec, features)[0]
    sims[query_idx] = -1
    top_indices = np.argsort(sims)[::-1][:n]
    scores = sims[top_indices]
    return top_indices.tolist(), scores.tolist()


def get_recommendations_color(query_idx: int, features: np.ndarray, n: int = 5):
    query_vec = features[query_idx].reshape(1, -1)
    dists = euclidean_distances(query_vec, features)[0]
    dists[query_idx] = np.inf
    top_indices = np.argsort(dists)[:n]
    scores = 1 / (1 + dists[top_indices])
    return top_indices.tolist(), scores.tolist()


def extract_features_from_upload(uploaded_image, model_name: str):
    """Извлекает признаки из загруженного изображения."""
    try:
        import tensorflow as tf
        from tensorflow.keras.applications import VGG16, ResNet50, EfficientNetB0
        from tensorflow.keras.preprocessing.image import img_to_array
        from tensorflow.keras.models import Model

        img = Image.open(uploaded_image).convert('RGB').resize((224, 224))
        arr = img_to_array(img)
        arr = np.expand_dims(arr, axis=0)

        if model_name == 'VGG16':
            from tensorflow.keras.applications.vgg16 import preprocess_input
            base = VGG16(weights='imagenet', include_top=False, pooling='avg')
        elif model_name == 'ResNet50':
            from tensorflow.keras.applications.resnet50 import preprocess_input
            base = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        elif model_name == 'EfficientNetB0':
            from tensorflow.keras.applications.efficientnet import preprocess_input
            base = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
        else:
            return None

        model = Model(inputs=base.input, outputs=base.output)
        arr = preprocess_input(arr)
        features = model.predict(arr, verbose=0)
        return normalize(features)[0]
    except Exception as e:
        st.warning(f"Не удалось извлечь признаки: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# Sidebar — настройки
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Настройки")
    st.divider()

    # Пути к файлам
    st.markdown("### Файлы данных")
    styles_csv = st.text_input(
        "Путь к styles_sample.csv",
        value="styles_sample.csv",
        help="CSV-файл с метаданными товаров (из ноутбука)"
    )
    features_npy = st.text_input(
        "Путь к best_features.npy",
        value="best_features.npy",
        help="Признаки товаров (из ноутбука)"
    )
    model_info_pkl = st.text_input(
        "Путь к model_info.pkl",
        value="model_info.pkl",
        help="Информация о лучшей модели"
    )

    st.divider()

    # Параметры поиска
    st.markdown("### Параметры поиска")
    n_recommendations = st.slider("Количество рекомендаций", 3, 12, 6)

    st.divider()

    # Фильтры
    st.markdown("### Фильтры каталога")
    show_filters = st.checkbox("Применить фильтры к рекомендациям", value=False)

    st.divider()
    st.markdown("""
    <small style='color:#999'>
    Рекомендательная система на основе Transfer Learning.<br>
    Лабораторная работа №4
    </small>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Загрузка данных
# ─────────────────────────────────────────────────────────────
data_loaded = False
df = None
features = None
model_info = {}

if os.path.exists(styles_csv) and os.path.exists(features_npy):
    try:
        df = load_styles(styles_csv)
        features = load_features(features_npy)

        if os.path.exists(model_info_pkl):
            model_info = load_model_info(model_info_pkl)

        data_loaded = True
    except Exception as e:
        st.error(f"Ошибка загрузки данных: {e}")


# ─────────────────────────────────────────────────────────────
# Главный интерфейс
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">Fashion Recommender</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Рекомендательная система модных товаров на основе Computer Vision & Transfer Learning</div>',
    unsafe_allow_html=True
)

if not data_loaded:
    st.warning("Файлы данных не найдены. Сначала запустите ноутбук `LR4_Fashion_RecSys.ipynb`, чтобы сгенерировать `styles_sample.csv`, `best_features.npy` и `model_info.pkl`.")
    st.info("После запуска ноутбука укажите корректные пути в боковой панели слева.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# Вкладки
# ─────────────────────────────────────────────────────────────
tab_catalog, tab_upload, tab_stats = st.tabs([
    "Поиск по каталогу",
    "Загрузить изображение",
    "Статистика датасета"
])


# ═══════════════════════════════════════════════════════════
# ВКЛАДКА 1: Поиск по каталогу
# ═══════════════════════════════════════════════════════════
with tab_catalog:
    st.markdown("### Выберите товар для поиска похожих")

    col_f1, col_f2, col_f3, col_f4 = st.columns(4)

    with col_f1:
        categories = ['Все'] + sorted(df['masterCategory'].dropna().unique().tolist())
        selected_cat = st.selectbox("Категория", categories)

    with col_f2:
        genders = ['Все'] + sorted(df['gender'].dropna().unique().tolist())
        selected_gender = st.selectbox("Пол", genders)

    with col_f3:
        seasons = ['Все'] + sorted(df['season'].dropna().unique().tolist())
        selected_season = st.selectbox("Сезон", seasons)

    with col_f4:
        article_types = ['Все'] + sorted(df['articleType'].dropna().unique().tolist())
        selected_article = st.selectbox("Тип товара", article_types)

    # Применяем фильтры к каталогу
    filtered_df = df.copy()
    if selected_cat != 'Все':
        filtered_df = filtered_df[filtered_df['masterCategory'] == selected_cat]
    if selected_gender != 'Все':
        filtered_df = filtered_df[filtered_df['gender'] == selected_gender]
    if selected_season != 'Все':
        filtered_df = filtered_df[filtered_df['season'] == selected_season]
    if selected_article != 'Все':
        filtered_df = filtered_df[filtered_df['articleType'] == selected_article]

    st.caption(f"Найдено товаров по фильтру: **{len(filtered_df)}**")

    if len(filtered_df) == 0:
        st.warning("По выбранным фильтрам нет товаров. Попробуйте другие параметры.")
        st.stop()

    # Выбор товара из отфильтрованного списка
    st.divider()

    # Показываем сетку товаров для выбора
    st.markdown("#### 🖼️ Выберите товар из каталога")
    preview_n = st.slider("Показать товаров для выбора", 6, 24, 12, step=6)

    preview_df = filtered_df.sample(min(preview_n, len(filtered_df)), random_state=42).reset_index(drop=True)
    preview_cols = st.columns(6)

    selected_product_id = st.session_state.get('selected_product_id', None)
    selected_df_idx = None

    for i, (_, row) in enumerate(preview_df.iterrows()):
        col = preview_cols[i % 6]
        with col:
            img = load_image_safe(row['image_path'], size=(150, 150))
            is_selected = selected_product_id == row['id']

            border_style = "border: 3px solid #667eea;" if is_selected else "border: 1px solid #ddd;"
            st.markdown(f'<div style="border-radius:10px;{border_style}padding:4px;margin-bottom:8px;">', unsafe_allow_html=True)
            st.image(img, use_container_width=True)
            label = f"{row.get('articleType', '')} | {row.get('gender', '')}"
            if st.button(f"Выбрать", key=f"btn_{row['id']}_{i}", use_container_width=True):
                st.session_state['selected_product_id'] = row['id']
                st.rerun()
            st.caption(f"*{label[:25]}*")
            st.markdown('</div>', unsafe_allow_html=True)

    # Также позволяем ввести ID товара вручную
    st.divider()
    col_input, col_btn = st.columns([3, 1])
    with col_input:
        manual_id = st.number_input(
            "Или введите ID товара вручную",
            min_value=int(df['id'].min()),
            max_value=int(df['id'].max()),
            step=1,
            value=int(df['id'].iloc[0])
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔍 Найти по ID", use_container_width=True):
            st.session_state['selected_product_id'] = manual_id
            st.rerun()

    # Получаем выбранный товар
    current_id = st.session_state.get('selected_product_id', None)
    if current_id is None:
        current_id = df['id'].iloc[0]

    product_rows = df[df['id'] == current_id]
    if len(product_rows) == 0:
        st.warning(f"Товар с ID={current_id} не найден в выборке.")
        st.stop()

    query_row = product_rows.iloc[0]
    query_idx = product_rows.index[0]

    # ─── Блок: Выбранный товар + Рекомендации ───
    st.divider()
    st.markdown("## 🎯 Результаты рекомендации")

    col_query, col_recs = st.columns([1, 3])

    with col_query:
        st.markdown("**Выбранный товар:**")
        img_query = load_image_safe(query_row['image_path'], size=(250, 250))

        st.markdown('<div class="query-card">', unsafe_allow_html=True)
        st.image(img_query, use_container_width=True)

        name = str(query_row.get('productDisplayName', ''))[:50]
        st.markdown(f"**{name}**" if name else "")
        st.markdown(f"""
        <span class="tag">{query_row.get('articleType','—')}</span>
        <span class="tag">{query_row.get('gender','—')}</span>
        <span class="tag">{query_row.get('season','—')}</span>
        <span class="tag">{query_row.get('baseColour','—')}</span>
        <span class="tag">{query_row.get('masterCategory','—')}</span>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_recs:
        st.markdown("**Похожие товары:**")

        # Получаем рекомендации
        best_model = model_info.get('best_model', 'VGG16')

        if best_model == 'ColorHistogram':
            rec_indices, rec_scores = get_recommendations_color(query_idx, features, n=n_recommendations * 3)
        else:
            rec_indices, rec_scores = get_recommendations_cosine(query_idx, features, n=n_recommendations * 3)

        # Фильтруем рекомендации по выбранным параметрам (если включено)
        if show_filters:
            valid_indices = []
            valid_scores = []
            for idx, score in zip(rec_indices, rec_scores):
                row = df.iloc[idx]
                ok = True
                if selected_cat != 'Все' and row.get('masterCategory') != selected_cat:
                    ok = False
                if selected_gender != 'Все' and row.get('gender') != selected_gender:
                    ok = False
                if selected_season != 'Все' and row.get('season') != selected_season:
                    ok = False
                if ok:
                    valid_indices.append(idx)
                    valid_scores.append(score)
                if len(valid_indices) >= n_recommendations:
                    break
            rec_indices = valid_indices
            rec_scores = valid_scores
        else:
            rec_indices = rec_indices[:n_recommendations]
            rec_scores = rec_scores[:n_recommendations]

        # Отображаем рекомендации сеткой
        n_cols = min(n_recommendations, 4)
        cols_rec = st.columns(n_cols)

        for i, (idx, score) in enumerate(zip(rec_indices, rec_scores)):
            col = cols_rec[i % n_cols]
            rec_row = df.iloc[idx]
            with col:
                rec_img = load_image_safe(rec_row['image_path'], size=(180, 180))
                st.image(rec_img, use_container_width=True)
                st.markdown(f"""
                <div style="text-align:center;font-size:0.75rem;color:#555;">
                    {rec_row.get('articleType','')}<br>
                    {rec_row.get('gender','')} | {rec_row.get('season','')}<br>
                    <span style="color:#667eea;font-weight:600;">Сходство: {score:.3f}</span>
                </div>
                """, unsafe_allow_html=True)

    # Метрики
    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Модель", model_info.get('best_model', '—'))
    with m2:
        st.metric("Товаров в базе", model_info.get('n_items', len(df)))
    with m3:
        st.metric("Размерность вектора", model_info.get('feature_dim', features.shape[1] if features is not None else '—'))
    with m4:
        prec = model_info.get('precision', None)
        st.metric("Precision@K", f"{prec:.3f}" if prec else "—")


# ═══════════════════════════════════════════════════════════
# ВКЛАДКА 2: Загрузка своего изображения
# ═══════════════════════════════════════════════════════════
with tab_upload:
    st.markdown("### 📷 Найти похожие товары по своему изображению")
    st.info("Загрузите фотографию товара, и система найдёт похожие товары из каталога.")

    col_up1, col_up2 = st.columns([1, 2])

    with col_up1:
        uploaded_file = st.file_uploader(
            "Загрузите изображение товара",
            type=['jpg', 'jpeg', 'png', 'webp'],
            help="Поддерживаются форматы JPG, PNG, WEBP"
        )

        if uploaded_file:
            img_preview = Image.open(uploaded_file).convert('RGB')
            st.image(img_preview, caption="Ваше изображение", use_container_width=True)

            model_choice = st.selectbox(
                "Выберите модель для сравнения",
                ['VGG16', 'ResNet50', 'EfficientNetB0'],
                index=0
            )
            n_recs_upload = st.slider("Количество рекомендаций", 3, 10, 5, key="upload_n")

            find_btn = st.button("🔍 Найти похожие товары", type="primary", use_container_width=True)

    with col_up2:
        if uploaded_file and find_btn:
            with st.spinner(f"Извлекаю признаки ({model_choice})..."):
                query_features = extract_features_from_upload(uploaded_file, model_choice)

            if query_features is not None and features is not None:
                # Сравниваем с базой
                query_vec = query_features.reshape(1, -1)
                base_norm = normalize(features) if model_info.get('best_model') != 'ColorHistogram' else features
                sims = cosine_similarity(query_vec, base_norm)[0]
                top_indices = np.argsort(sims)[::-1][:n_recs_upload]
                top_scores = sims[top_indices]

                st.markdown(f"#### ✅ Найдено {n_recs_upload} похожих товаров:")

                n_cols = min(n_recs_upload, 5)
                cols_up = st.columns(n_cols)

                for i, (idx, score) in enumerate(zip(top_indices, top_scores)):
                    col = cols_up[i % n_cols]
                    rec_row = df.iloc[idx]
                    with col:
                        rec_img = load_image_safe(rec_row['image_path'])
                        st.image(rec_img, use_container_width=True)
                        st.markdown(f"""
                        <div style="text-align:center;font-size:0.75rem;">
                            <b>{rec_row.get('articleType','')}</b><br>
                            {rec_row.get('gender','')} | {rec_row.get('season','')}<br>
                            <span style="color:#667eea;">{score:.3f}</span>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.error("Не удалось извлечь признаки. Убедитесь, что TensorFlow установлен.")

        elif not uploaded_file:
            st.markdown("""
            <div style="text-align:center;padding:60px 20px;color:#aaa;">
                <div style="font-size:3rem;"></div>
                <div style="font-size:1.1rem;">Загрузите изображение слева<br>для поиска похожих товаров</div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# ВКЛАДКА 3: Статистика датасета
# ═══════════════════════════════════════════════════════════
with tab_stats:
    st.markdown("### Анализ датасета")

    # Сводка
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric("Всего товаров", len(df))
    col_s2.metric("Категорий", df['masterCategory'].nunique())
    col_s3.metric("Типов товаров", df['articleType'].nunique())
    col_s4.metric("Цветов", df['baseColour'].nunique())

    st.divider()

    import plotly.express as px
    import plotly.graph_objects as go

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        # Распределение по полу
        gender_counts = df['gender'].value_counts()
        fig1 = px.pie(
            values=gender_counts.values,
            names=gender_counts.index,
            title="Распределение по полу",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig1.update_layout(height=350)
        st.plotly_chart(fig1, use_container_width=True)

    with col_g2:
        # Распределение по сезону
        season_counts = df['season'].value_counts()
        fig2 = px.bar(
            x=season_counts.index,
            y=season_counts.values,
            title="Распределение по сезону",
            color=season_counts.values,
            color_continuous_scale='Blues',
            labels={'x': 'Сезон', 'y': 'Количество'}
        )
        fig2.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    col_g3, col_g4 = st.columns(2)

    with col_g3:
        # Топ категории
        cat_counts = df['masterCategory'].value_counts()
        fig3 = px.bar(
            x=cat_counts.values,
            y=cat_counts.index,
            orientation='h',
            title="Главные категории",
            color=cat_counts.values,
            color_continuous_scale='Purples',
            labels={'x': 'Количество', 'y': 'Категория'}
        )
        fig3.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    with col_g4:
        # Топ-12 типов товаров
        article_counts = df['articleType'].value_counts()[:12]
        fig4 = px.bar(
            x=article_counts.values,
            y=article_counts.index,
            orientation='h',
            title="Топ-12 типов товаров",
            color=article_counts.values,
            color_continuous_scale='Greens',
            labels={'x': 'Количество', 'y': 'Тип'}
        )
        fig4.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

    # Тепловая карта
    st.markdown("#### Матрица: Пол × Категория")
    pivot = df.pivot_table(
        index='gender',
        columns='masterCategory',
        values='id',
        aggfunc='count',
        fill_value=0
    )
    fig5 = px.imshow(
        pivot,
        color_continuous_scale='Blues',
        aspect='auto',
        text_auto=True,
        title="Количество товаров: Пол vs Категория"
    )
    fig5.update_layout(height=300)
    st.plotly_chart(fig5, use_container_width=True)

    # Топ цветов
    st.markdown("#### Распределение цветов")
    color_counts = df['baseColour'].value_counts()[:15]
    fig6 = px.pie(
        values=color_counts.values,
        names=color_counts.index,
        title="Топ-15 цветов",
        color_discrete_sequence=px.colors.qualitative.Light24
    )
    fig6.update_layout(height=400)
    st.plotly_chart(fig6, use_container_width=True)
