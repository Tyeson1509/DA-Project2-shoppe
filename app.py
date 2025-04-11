import streamlit as st
import pandas as pd
import math
from gensim import corpora, models, similarities
import ast
import random
import pickle
from surprise import Dataset, Reader

# 🔒 Kiểm tra trạng thái đăng nhập từ session
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login_page():
    st.set_page_config(page_title="Đăng nhập", layout="centered")
    st.title("🔐 Đăng nhập")
    username_input = st.text_input("Tên đăng nhập")
    password_input = st.text_input("Mật khẩu", type="password")
    login_button = st.button("Đăng nhập")

    if login_button:
        try:
            df_users = pd.read_csv("data/Products_ThoiTrangNam_rating_raw.csv", delimiter="\t")
            valid_users = df_users["user"].unique().tolist()
        except Exception as e:
            st.error(f"Không đọc được file đăng nhập: {e}")
            st.stop()

        if username_input in valid_users and password_input == "123":
            st.session_state.logged_in = True
            st.session_state.username = username_input
            st.rerun()
        else:
            st.error("Tên đăng nhập hoặc mật khẩu không đúng!")
    
        # 🔽 Phần chú thích
    st.markdown("""
    ---
    📝 **Hướng dẫn đăng nhập:**
    
    Lấy các username trong file `Products_ThoiTrangNam_rating_raw.csv` để đăng nhập với password mặc định là `123`.

    **Ví dụ:**
    - Username: `karmakyun2nd`  
    - Password: `123`
    """)


# ✅ Nếu chưa đăng nhập thì hiển thị form login và STOP
if not st.session_state.logged_in:
    login_page()
    st.stop()

# --- Sau khi đăng nhập ---
st.set_page_config(page_title="Shopee", layout="wide")

query_params = st.query_params
menu_options = ["Trang chủ", "Sản phẩm", "About Us"]
menu = query_params.get("menu", "Trang chủ")
if menu not in menu_options:
    menu = "Trang chủ"

selected_menu = st.sidebar.radio("📌 Điều hướng", menu_options, index=menu_options.index(menu))
if selected_menu != menu:
    st.query_params["menu"] = selected_menu
    if "product_id" in st.query_params:
        del st.query_params["product_id"]
    st.rerun()

# Load dữ liệu đánh giá
@st.cache_data
def load_user_rating_data():
    df_ratings = pd.read_csv("data/Products_ThoiTrangNam_rating_raw.csv", delimiter="\t")
    return df_ratings

ratings_df = load_user_rating_data()

# Load model collaborative filtering
@st.cache_resource
def load_collaborative_model():
    with open("models/surprise_model.pkl", "rb") as f:
        return pickle.load(f)

collab_model = load_collaborative_model()

def recommend_for_user(user_id, top_n=10):
    rated_products = ratings_df[ratings_df["user_id"] == user_id]["product_id"].unique()
    all_product_ids = products_df["product_id"].unique()

    # Lọc sản phẩm chưa đánh giá
    products_to_predict = [pid for pid in all_product_ids if pid not in rated_products]

    predictions = []
    for pid in products_to_predict:
        try:
            pred = collab_model.predict(user_id, pid)
            predictions.append((pid, pred.est))
        except:
            continue

    top_predictions = sorted(predictions, key=lambda x: -x[1])[:top_n]
    top_products = products_df[products_df["product_id"].isin([pid for pid, _ in top_predictions])]
    return top_products

# --- Load dữ liệu ---
@st.cache_data
def load_data():
    df = pd.read_csv("data/Products_ThoiTrangNam_cleaned.csv")
    valid_images = df["image"].dropna()
    valid_images = valid_images[valid_images.apply(lambda x: isinstance(x, str) and x.startswith("http"))].tolist()
    random.seed(42)

    def replace_invalid_image(img):
        return img if isinstance(img, str) and img.startswith("http") else random.choice(valid_images)

    df["image"] = df["image"].apply(replace_invalid_image)
    df["final_tokens"] = df["final_tokens"].apply(lambda x: tuple(ast.literal_eval(x)))
    return df

products_df = load_data().reset_index(drop=True)

# --- Load TF-IDF models ---
@st.cache_resource
def load_model_components():
    dictionary = corpora.Dictionary.load("models/product_dictionary.dict")
    tfidf_model = models.TfidfModel.load("models/product_tfidf.model")
    index = similarities.MatrixSimilarity.load("models/product_similarity.index")
    return dictionary, tfidf_model, index

dictionary, tfidf_model, index = load_model_components()

# --- TF-IDF corpus ---
@st.cache_data
def get_tfidf_corpus(df):
    corpus = [dictionary.doc2bow(tokens) for tokens in df['final_tokens']]
    return tfidf_model[corpus]

tfidf_corpus = get_tfidf_corpus(products_df)

def recommend_similar_products(product_id, top_n=5):
    try:
        product_idx = products_df[products_df["product_id"] == product_id].index[0]
    except IndexError:
        return pd.DataFrame()

    query_doc = tfidf_corpus[product_idx]
    sims = index[query_doc]
    sims = sorted(enumerate(sims), key=lambda x: -x[1])
    sims = [s for s in sims if s[0] != product_idx][:top_n]

    recommendations = []
    for idx, score in sims:
        product_info = products_df.iloc[idx]
        recommendations.append({
            "product_id": product_info["product_id"],
            "product_name": product_info["product_name"],
            "description": product_info.get("description", ""),
            "price": product_info.get("price", ""),
            "category": product_info.get("category", ""),
            "similarity_score": score
        })

    return pd.DataFrame(recommendations)

# --- Trang chủ ---
if selected_menu == "Trang chủ":
    st.title("🛒 Shopee")
    st.header("🏠 Trang chủ")
    st.image("images/shopee-banner.png", width=600)
    st.markdown("<h4>Chào mừng bạn đến với Shopee – nơi mua sắm thú vị theo từng khung giờ!</h4>", unsafe_allow_html=True)

    username = st.session_state.get("username")
    user_row = ratings_df[ratings_df["user"] == username]
    if not user_row.empty:
        user_id = user_row["user_id"].iloc[0]
        st.subheader("🎯 Gợi ý dành riêng cho bạn")
        recommended = recommend_for_user(user_id, top_n=10)

        rec_cols = st.columns(5)
        for i, (_, row) in enumerate(recommended.iterrows()):
            with rec_cols[i % 5]:
                try:
                    st.image(row["image"], use_container_width=True)
                except:
                    st.warning("Không có ảnh")
                link = f'<a href="?menu=Sản%20phẩm&product_id={row["product_id"]}"><b>{row["product_name"][:30]}</b></a>'
                st.markdown(link, unsafe_allow_html=True)
                st.markdown(f"💰 **{row['price']:,}₫**")
    else:
        st.info("Bạn chưa có dữ liệu đánh giá để gợi ý sản phẩm.")

# --- Sản phẩm ---
elif selected_menu == "Sản phẩm":
    st.header("🛍️ Sản phẩm")
    sub_categories = ["All"] + sorted(products_df["sub_category"].dropna().unique())

    if "selected_sub_category" not in st.session_state:
        st.session_state.selected_sub_category = "All"

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        search_term = st.text_input("🔍 Tìm kiếm sản phẩm", value="")
    with col2:
        selected_sub_category = st.selectbox(
            "📂 Lọc theo danh mục con",
            sub_categories,
            index=sub_categories.index(st.session_state.selected_sub_category)
        )

    if selected_sub_category != st.session_state.selected_sub_category:
        st.session_state.selected_sub_category = selected_sub_category
        st.session_state.page = 1
        st.rerun()

    filtered_df = products_df.copy()
    if search_term:
        filtered_df = filtered_df[filtered_df["product_name"].str.contains(search_term, case=False, na=False)]
    if selected_sub_category != "All":
        filtered_df = filtered_df[filtered_df["sub_category"] == selected_sub_category]

    if "product_id" in st.query_params:
        try:
            product_id = int(st.query_params["product_id"])
            product = products_df[products_df["product_id"] == product_id].iloc[0]
        except:
            product = None

        if product is not None:
            st.image(product["image"], width=300)
            st.subheader(f"🔎 {product['product_name']}")
            st.markdown(f"**Mã sản phẩm:** {product['product_id']}")
            st.markdown(f"**Danh mục:** {product['category']} / {product['sub_category']}")
            st.markdown(f"**Giá:** {product['price']:,}₫")
            st.markdown(f"**Đánh giá:** ⭐ {product['rating']}")
            st.markdown("**Mô tả:**")
            st.write(product["description"])

            st.markdown("---")
            st.subheader("🧠 Có thể bạn cũng thích:")

            recommended_df = recommend_similar_products(product['product_id'], top_n=5)
            rec_cols = st.columns(5)
            for i, (_, row) in enumerate(recommended_df.iterrows()):
                with rec_cols[i]:
                    try:
                        img_url = products_df[products_df["product_id"] == row["product_id"]]["image"].values[0]
                        st.image(img_url, use_container_width=True)
                    except:
                        st.warning("Không có ảnh")

                    link = f'<a href="?menu=Sản%20phẩm&product_id={row["product_id"]}"><b>{row["product_name"][:30]}</b></a>'
                    st.markdown(link, unsafe_allow_html=True)
                    st.markdown(f"💰 **{row['price']:,}₫**")

            st.markdown("---")
            if st.button("⬅️ Quay lại danh sách"):
                st.query_params.clear()
                st.rerun()
        else:
            st.error("Không tìm thấy sản phẩm.")
    else:
        items_per_page = 30
        total_items = len(filtered_df)
        total_pages = math.ceil(total_items / items_per_page)

        if "page" not in st.session_state:
            st.session_state.page = 1
        if st.session_state.page > total_pages:
            st.session_state.page = max(total_pages, 1)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("⬅️ Trang trước") and st.session_state.page > 1:
                st.session_state.page -= 1
        with col3:
            if st.button("Trang sau ➡️") and st.session_state.page < total_pages:
                st.session_state.page += 1
        with col2:
            st.markdown(f"<div style='text-align: center;'>Trang {st.session_state.page} / {total_pages}</div>", unsafe_allow_html=True)

        start = (st.session_state.page - 1) * items_per_page
        end = start + items_per_page
        page_data = filtered_df.iloc[start:end]

        cols = st.columns(5)
        for index, row in page_data.iterrows():
            with cols[index % 5]:
                try:
                    st.image(row["image"], width=120)
                except:
                    st.warning("Không thể hiển thị ảnh.")
                st.markdown(
                    f"""
                    <div style="height: 80px; overflow: hidden; text-overflow: ellipsis;">
                        <a href="?menu=Sản phẩm&product_id={row["product_id"]}" style="font-weight:bold; text-decoration:none; color:white;">
                            {row["product_name"]}
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"💰 <b>{row['price']:,}₫</b> &nbsp;&nbsp; ⭐ <b>{row['rating']}</b>",
                    unsafe_allow_html=True
                )

# --- About Us ---
elif selected_menu == "About Us":
    #st.title("🛒 Shopee")
    st.header("ℹ️ About Us")
    st.write("Shopee là nền tảng bán hàng theo khung giờ giúp bạn tiết kiệm và săn deal hấp dẫn.")

    st.markdown("""
        ---
        **📌 Sản phẩm được thực hiện bởi nhóm:**

        - 👨‍💻 Mạch Cảnh Toàn  
        - 👩‍💻 Hoàng Thị Diệp

        **🎓 Giảng viên hướng dẫn:**  
        Cô Khuất Thùy Phương

        ---
        **🔍 Các mô hình sử dụng trong project:**

        - **Content-Based Filtering:**  
        Sử dụng thư viện `Gensim` để trích xuất đặc trưng văn bản từ mô tả sản phẩm thông qua TF-IDF.  
        Mô hình này xây dựng ma trận tương đồng giữa các sản phẩm, từ đó đưa ra gợi ý dựa trên nội dung mô tả của sản phẩm.

        - **Collaborative Filtering:**  
        Sử dụng thư viện `Surprise` với thuật toán `BaselineOnly` để mô hình hóa hành vi người dùng dựa trên đánh giá trước đó.  
        Mô hình có khả năng dự đoán sản phẩm mà người dùng có thể quan tâm, kể cả khi chưa từng tương tác với sản phẩm đó.

        """)

    st.image("images/recommend-model.jpg", caption="Hình minh họa hệ thống gợi ý sản phẩm", use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Một số kết quả phân tích dữ liệu (EDA)")

    col1, col2 = st.columns(2)
    with col1:
        st.image("images/rating-distribution.PNG", caption="Phân phối mất cân đối, phần lớn sản phẩm ít được đánh giá", use_container_width=True)
    with col2:
        st.image("images/rating-distribution-product.PNG", caption="Phân phối mất cân đối, phần lớn sản phẩm ít được đánh giá", use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.image("images/rating-quantity.PNG", caption="Theo dữ liệu product", use_container_width=True)
    with col4:
        st.image("images/user-rating-distribution.PNG", caption="Theo dữ liệu người dùng rating", use_container_width=True)

    st.markdown("""
    **📌 Nhận xét:**  
    Phần lớn rating score ở mức 5 (75%) , Rất ít rating score ở mức 1-3.
    """)
