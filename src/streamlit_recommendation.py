import streamlit as st
from PIL import Image
import pickle
import pandas as pd
import numpy as np
import scipy.sparse as sp
from surprise import Reader, Dataset, SVDpp
from utils import helpers,filters



st.set_page_config(page_title="Recommender System", page_icon=":shopping_cart:", layout="wide")
button_style = """
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 12px;
    }
    </style>
    """
menu = ["Project Summary", "Recommender System"]
st.sidebar.title("Đồ Án Tốt Nghiệp")
st.sidebar.markdown(
    """
    <h2 style="display: flex; align-items: center; font-size: 18px;">
        <span style="margin-left: 8px;">Recommender System Project</span>
    </h2>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.selectbox("Chức Năng", menu)

# Subheader with an icon for "Giảng viên hướng dẫn"
st.sidebar.markdown(
    """
    <h3 style="display: flex; align-items: center; font-size: 18px;">
        <span style="margin-left: 8px;">Giảng viên hướng dẫn</span>
    </h3>
    """,
    unsafe_allow_html=True,
)
st.sidebar.markdown("👩‍🏫 [Thạc Sĩ Khuất Thùy Phương](https://csc.edu.vn/giao-vien~37#)")

# Subheader with an icon for "Thành Viên"
st.sidebar.markdown(
    """
    <style>
        .footer {
            text-align: center;
            font-size: 12px;
        }
        hr {
            border: 0.5px solid gray;
        }
    </style>
    <div class="footer">
        <hr>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    """
    <h3 style="display: flex; align-items: center; font-size: 18px;">
        <span style="margin-left: 8px;">Thành Viên</span>
    </h3>
    """,
    unsafe_allow_html=True,
)
st.sidebar.markdown(" :boy: Mã Thế Nhựt")
st.sidebar.markdown(" :girl: Từ Thị Thanh Xuân")

# Subheader with an icon for "Ngày báo cáo tốt nghiệp"
st.sidebar.markdown(
    """
    <style>
        .footer {
            text-align: center;
            font-size: 12px;
        }
        hr {
            border: 0.5px solid gray;
        }
    </style>
    <div class="footer">
        <hr>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    """
    <h3 style="display: flex; align-items: center; font-size: 18px;">
        <span style="margin-left: 8px;">Ngày báo cáo tốt nghiệp:</span>
    </h3>
    """,
    unsafe_allow_html=True,
)
st.sidebar.write('📅 16/12/2024')

# Add spacer for footer positioning
st.sidebar.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)

# Sidebar footer
st.sidebar.write("© 2024 Hasaki Recommender System")

# Load Data
@st.cache_data
def load_data():
    df = helpers.read_csv('src/data/df_file.csv')
    df_products = helpers.read_csv('src/data/San_pham.csv')
    return df, df_products

df, df_products = load_data()


# # Load or Train Model
@st.cache_resource
def load_model(df):
    reader = Reader()
    data_sample = Dataset.load_from_df(df[['ma_khach_hang', 'ma_san_pham', 'so_sao']].sample(5000, random_state=42), reader)
    trainset = data_sample.build_full_trainset()
    algorithm = SVDpp()
    algorithm.fit(trainset)
    return algorithm
# Main page logic
if page == "Project Summary":
    # Banner Image
    image = Image.open("src/images/hasaki_banner.jpg")
    st.image(image)

    tab_containers = st.tabs(['Hasaki Project', 'Thực Hiện Dự Án'])

    with tab_containers[0]:
        # Title Section
        st.title("Dự án HASAKI: Xây dựng hệ thống recommendation 🛍️")

        # Introduction
        st.subheader("1. Giới thiệu về Hasaki")
        st.write("""
        **HASAKI.VN** là hệ thống cửa hàng mỹ phẩm chính hãng và dịch vụ chăm sóc sắc đẹp trải dài khắp Việt Nam.  
        Là **đối tác chiến lược** của nhiều thương hiệu lớn, Hasaki cung cấp nền tảng trực tuyến tiện lợi giúp khách hàng:  
        - 🛍️ Mua sắm sản phẩm chất lượng  
        - 🌟 Xem đánh giá thực tế  
        - 🚚 Đặt hàng nhanh chóng  
        """)

        # Divider
        st.divider()

        # Problem Section
        st.subheader("2. Vấn đề 🚩")
        st.markdown("""
        - 🔍 **Khách hàng gặp khó khăn** trong tìm kiếm sản phẩm phù hợp do danh mục quá lớn.  
        - 😟 **Trải nghiệm chưa tối ưu**, giảm sự hài lòng và bỏ lỡ cơ hội doanh thu.  
        - 🤖 Thiếu **hệ thống gợi ý cá nhân hóa** để hỗ trợ khách hàng tốt hơn.
        """)

        # Divider
        st.divider()

        # Objectives Section
        st.subheader("3. Mục tiêu 🎯")
        st.markdown("""
        - ✅ **Cá nhân hóa trải nghiệm mua sắm**, tăng sự hài lòng và tương tác.  
        - 📈 **Tăng tỷ lệ chuyển đổi** và doanh thu.  
        - 🏆 Khẳng định Hasaki là **nền tảng mỹ phẩm trực tuyến hàng đầu**.
        """)

        # Divider
        st.divider()

        # Solutions Section
        st.subheader("4. Giải pháp 💡")
        st.markdown("""
        - 🤖 **Ứng dụng Machine Learning** để phát triển hệ thống gợi ý sản phẩm cá nhân hóa.  
        - 🔗 **Đề xuất sản phẩm** dựa trên:  
            - Lịch sử mua hàng  
            - Xu hướng tìm kiếm  
            - Hành vi tương tự từ khách hàng khác  
        - 📊 **Liên tục tối ưu** dựa trên phản hồi thực tế và dữ liệu thu thập.  
        - 🌐 **Tích hợp hệ thống gợi ý** trên cả website và ứng dụng di động, mang lại trải nghiệm đồng nhất.
        """)

    with tab_containers[1]:  # Assuming the second tab is for the project process
        st.subheader("Quy trình thực hiện 💡")
        # Step 1: Data Collection
        st.subheader("1. Thu thập dữ liệu 📝")
        st.markdown("""
        - **Nguồn dữ liệu:**  
            - Lịch sử mua sắm của khách hàng trên Hasaki.vn.  
            - Lịch sử xem sản phẩm và tương tác với website.  
            - Đánh giá và bình luận về sản phẩm.  
        - **Mục tiêu:**  
            - Tạo tập dữ liệu phong phú và chất lượng cao để đào tạo và đánh giá hệ thống gợi ý.
            - Xác định hành vi, sở thích và nhu cầu mua sắm của từng khách hàng.
        """)

        # Step 2: Data Preprocessing
        st.subheader("2. Xử lý dữ liệu 🔍")
        st.markdown("""
        ### Quy trình xử lý dữ liệu

        1. **Kết hợp Tiêu đề và Mô tả Sản phẩm 📝**
            - **Mục tiêu**: Tạo cột văn bản duy nhất.
            - **Thao tác**: Ghép cột `ten_san_pham` và `mo_ta` thành `content`.
            - **Kết quả**: Văn bản tổng hợp chứa thông tin sản phẩm.

        2. **Loại bỏ Văn Bản Không Cần Thiết ✂️**
            - **Mục tiêu**: Xóa các đoạn không liên quan.
            - **Thao tác**: Cắt bỏ nội dung sau các cụm từ xác định như `"Lưu ý"`.
            - **Kết quả**: Nội dung sạch hơn, tập trung vào sản phẩm.

        3. **Lọc và Làm Sạch Từ Không Hợp Lệ 🗒️**
            - **Mục tiêu**: Xóa từ thừa và không phù hợp.
            - **Thao tác**:
                - Loại bỏ **stopwords** tiếng Việt và chuyên biệt.
                - Xóa hoặc thay thế các từ sai chính tả.
            - **Kết quả**: Văn bản chỉ giữ lại các từ quan trọng.

        4. **Chuẩn Hóa Văn Bản ✏️**
            - **Mục tiêu**: Đưa văn bản về định dạng chuẩn.
            - **Thao tác**:
                - Chuyển chữ thường, chuẩn hóa ký tự lặp, xóa dấu câu và số.
            - **Kết quả**: Văn bản nhất quán, dễ xử lý.

        5. **Gắn Thẻ Từ Loại (POS Tagging) 🧠**
            - **Mục tiêu**: Xác định từ loại quan trọng.
            - **Thao tác**: Gắn nhãn danh từ, động từ, tính từ; loại bỏ từ không cần thiết.
            - **Kết quả**: Văn bản tinh gọn với các từ ý nghĩa.

        6. **Kết Quả Cuối Cùng ✅**
            - **Mục tiêu**: Lưu trữ văn bản sạch, chuẩn hóa.
            - **Thao tác**: Lưu vào cột `cleaned_content`.
            - **Kết quả**: Văn bản sẵn sàng cho các bước phân tích và mô hình.
        """)

        # Step 3: Model Selection
        st.subheader("3. Chọn mô hình 📊")
        st.markdown("""
        ### Phương pháp xây dựng Recommender System:

        1. **Collaborative Filtering (Lọc cộng tác)**  
        - **Cách hoạt động**:
            - Dựa trên hành vi mua sắm của những khách hàng tương tự để đưa ra gợi ý.
            - Phân tích dữ liệu về sản phẩm mà khách hàng đã mua, đánh giá hoặc tương tác để xác định sự tương đồng giữa các khách hàng.
        - **Thuật toán sử dụng**:
            1. **Surprise Library**:
                - Cung cấp các thuật toán phổ biến như SVD, KNN, và các công cụ đánh giá mô hình Collaborative Filtering.
        - **Ưu điểm**:
            - Hiệu quả với dữ liệu hành vi phong phú.
            - Khai thác sở thích tiềm ẩn dựa trên tương tác giữa người dùng và sản phẩm.

        2. **Content-Based Filtering (Lọc theo nội dung)**  
        - **Cách hoạt động**:
            - Phân tích đặc điểm sản phẩm (ví dụ: mô tả, tiêu đề) để tìm sản phẩm tương tự.
            - Gợi ý dựa trên mức độ tương đồng giữa sản phẩm khách hàng đã xem/mua và các sản phẩm khác.
        - **Thuật toán sử dụng**:
            1. **Gensim**:
                - Tạo từ điển và chuyển đổi dữ liệu văn bản sản phẩm thành ma trận TF-IDF.
            2. **Cosine Similarity**:
                - Tính toán mức độ tương tự giữa các sản phẩm dựa trên ma trận TF-IDF.
            3. **Kết hợp Gensim và Cosine Similarity**:
                - Sử dụng Gensim để xây dựng ma trận TF-IDF và áp dụng Cosine Similarity để tìm sản phẩm có nội dung tương đồng nhất.
        - **Ưu điểm**:
            - Phù hợp khi không có đủ dữ liệu hành vi của khách hàng.
            - Gợi ý dựa trên thông tin cụ thể của sản phẩm, dễ giải thích và mở rộng.

        ### Mục tiêu:
        - **Tối ưu hóa độ chính xác**:
            - Tìm ra phương pháp gợi ý phù hợp nhất với tập dữ liệu hiện tại.
        - **Khả năng mở rộng**:
            - Đảm bảo hệ thống hoạt động hiệu quả ngay cả khi có thêm khách hàng hoặc sản phẩm mới.
        """)

        # Step 4: Model Training and Evaluation
        st.subheader("4. Huấn luyện và đánh giá mô hình 🛠️")
        st.markdown("""
        ### Quy trình huấn luyện và đánh giá:

        1. **Huấn luyện mô hình:**
        - Sử dụng dữ liệu lịch sử mua sắm và tương tác của khách hàng để xây dựng hệ thống gợi ý.
        - Áp dụng các thuật toán:
            - **Surprise (Collaborative Filtering)**:
                - Dựa trên hành vi mua sắm của khách hàng để gợi ý sản phẩm.
                - Phù hợp với dữ liệu nhỏ hoặc vừa, thích hợp cho GUI với Streamlit.
            - **Gensim với Cosine Similarity (Content-Based Filtering)**:
                - Phân tích nội dung sản phẩm (mô tả, tiêu đề) để tìm sản phẩm tương tự.

        2. **Đánh giá mô hình:**
        - **Chỉ số đánh giá hiệu suất:**
            - **Precision**: Tỷ lệ gợi ý chính xác trong tổng số gợi ý.
            - **Recall**: Tỷ lệ gợi ý chính xác so với tổng số sản phẩm khách hàng đã tương tác.
            - **F1-Score**: Trung bình điều hòa giữa Precision và Recall.
        - **Phương pháp đánh giá:**
            - Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra.
            - Kiểm tra hiệu suất trên tập kiểm tra để đảm bảo độ chính xác và khả năng khái quát hóa.

        ### Kết quả kỳ vọng:
        - Mô hình gợi ý sản phẩm chính xác dựa trên cả hành vi mua sắm và đặc điểm sản phẩm.
        - Hiệu suất ổn định khi triển khai với dữ liệu thực tế và dễ dàng cập nhật khi có dữ liệu mới.
        """)

        # Step 5: Deployment
        st.subheader("5. Triển khai hệ thống 🌐")
        st.markdown("""
        ### Quy trình triển khai:

        1. **Tích hợp hệ thống gợi ý vào nền tảng Hasaki.vn:**
        - Hiển thị gợi ý sản phẩm tại các vị trí quan trọng:
            - **Trang chủ**: Gợi ý sản phẩm phổ biến hoặc cá nhân hóa theo khách hàng.
            - **Trang sản phẩm**: Đề xuất sản phẩm tương tự dựa trên nội dung sản phẩm.
            - **Giỏ hàng**: Gợi ý sản phẩm bổ sung phù hợp.
        - Sử dụng thuật toán:
            - **Collaborative Filtering với Surprise**.
            - **Content-Based Filtering với Gensim**.

        2. **Theo dõi và cải tiến:**
        - Thu thập phản hồi từ khách hàng để đánh giá hiệu quả.
        - Cải thiện giao diện hiển thị gợi ý để tăng trải nghiệm người dùng.
        """)

        # Step 6: Monitoring and Optimization
        st.subheader("6. Giám sát và tối ưu hóa 📈")
        st.markdown("""
        ### Quy trình giám sát và tối ưu hóa:

        1. **Theo dõi hiệu suất:**
        - Đánh giá sự thay đổi của:
            - **Tỷ lệ chuyển đổi (Conversion Rate)**.
            - **Doanh thu trung bình (Average Order Value)**.
        - Ghi nhận dữ liệu tương tác mới để đánh giá độ chính xác của hệ thống.

        2. **Tối ưu hóa mô hình:**
        - **Cập nhật mô hình thường xuyên**:
            - Sử dụng dữ liệu mới để cải thiện độ chính xác.
        - **Thử nghiệm thuật toán tiên tiến**:
            - Khám phá thêm các mô hình **Deep Learning** hoặc Hybrid.
        """)
      
#####################################

elif page == "Recommender System_OLD":
    image = Image.open("src/images/recommend.jpg")
    st.image(image, caption="Hasaki.VN - Quality & Trust", use_container_width=True)

    # Customer Recommendations
    def recommend_products_with_names(algorithm, ma_khach_hang, df, df_sp, n_recommendations=5):
        all_products = df['ma_san_pham'].unique()
        rated_products = df[df['ma_khach_hang'] == ma_khach_hang]['ma_san_pham'].unique()
        unrated_products = [product for product in all_products if product not in rated_products]

        predictions = []
        for ma_san_pham in unrated_products:
            prediction = algorithm.predict(ma_khach_hang, ma_san_pham)
            predictions.append((ma_san_pham, prediction.est))

        predictions.sort(key=lambda x: x[1], reverse=True)
        top_recommendations = predictions[:n_recommendations]

        recommended_df = pd.DataFrame(top_recommendations, columns=['ma_san_pham', 'predicted_rating'])
        recommended_with_names = recommended_df.merge(df_sp, on='ma_san_pham', how='left')
        return recommended_with_names[['ma_san_pham', 'ten_san_pham', 'predicted_rating']]

    # Product-Based Recommendations
    def get_product_recommendations(df_sp, ma_san_pham, gensim_model, n_recommendations=5):
        idx = df_sp.index[df_sp['ma_san_pham'] == ma_san_pham].tolist()
        if not idx:
            # st.error("Product not found!")
            return pd.DataFrame()
        idx = idx[0]
        sim_scores = list(enumerate(gensim_model[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n_recommendations+1]
        product_indices = [i[0] for i in sim_scores]
        return df_sp.iloc[product_indices]

    # Customer History
    def get_customer_history(ma_khach_hang, df, df_sp):
        customer_history = df[df['ma_khach_hang'] == ma_khach_hang]
        customer_history_with_names = customer_history.merge(df_sp, on='ma_san_pham', how='left')
        return customer_history_with_names[['ma_san_pham', 'ten_san_pham', 'so_sao']]

    # Main App
    st.title("🔍 Recommendation System")
    st.markdown("Choose an option below to get recommendations!")
    
    # Load data and models
    df, df_sp = load_data()
    algorithm = load_model(df)

    # Load product similarity matrix
    with open('src/models/gensim_model.pkl', 'rb') as f:
        loaded_gensim = pickle.load(f)

    # Choose an input method
    option = st.radio(
        "Choose an input method:",
        options=["Type Customer ID", "Type Product ID", "Search by Product Name", "Select from Product List"]
    )

    if option == "Type Customer ID":
        # Customer ID Input
        ma_khach_hang = st.text_input("Enter Customer ID:")
        if st.button("Get Recommendations"):
            if ma_khach_hang:
                try:
                    ma_khach_hang = int(ma_khach_hang)
                    if ma_khach_hang not in df['ma_khach_hang'].unique():
                        st.error("Customer ID not found!")
                    else:
                        # Display customer history
                        st.subheader("📜 Customer Shopping History")
                        history = get_customer_history(ma_khach_hang, df, df_sp)
                        st.dataframe(history if not history.empty else "No history available.")
                        
                        # Recommendations
                        st.subheader("✨ Recommendations")
                        recommendations = recommend_products_with_names(algorithm, ma_khach_hang, df, df_sp)
                        st.dataframe(recommendations if not recommendations.empty else "No recommendations available.")
                except ValueError:
                    st.error("Invalid Customer ID!")

    elif option == "Type Product ID":
        # Product ID Input
        product_id = st.text_input("Enter Product ID:")
        if st.button("Get Recommendations for Product"):
            if product_id:
                try:
                    product_id = int(product_id)
                    recommendations = get_product_recommendations(df_sp, product_id, loaded_gensim)
                    if recommendations.empty:
                        st.error("No similar products found!")
                    else:
                        st.dataframe(recommendations if not recommendations.empty else "No similar products found.")
                except ValueError:
                    st.error("Invalid Product ID!")

    elif option == "Search by Product Name":
        # Product Name Search
        product_name = st.text_input("Enter Product Name:")
        if st.button("Search Product"):
            if product_name:
                search_results = df_sp[df_sp['ten_san_pham'].str.contains(product_name, case=False)]
                if search_results.empty:
                    st.error("No matching products found!")
                else:
                    st.dataframe(search_results)

    elif option == "Select from Product List":
        # Select Product from List
        if "product_list" not in st.session_state:
            st.session_state.product_list = df_sp.sample(10)

        product_selection = st.selectbox(
            "Select a product:",
            options=st.session_state.product_list['ten_san_pham']
        )
        selected_product = st.session_state.product_list[st.session_state.product_list['ten_san_pham'] == product_selection]
        st.write("### Selected Product:")
        st.write(selected_product)

        if st.button("Get Recommendations for Selected Product"):
            product_id = selected_product['ma_san_pham'].values[0]
            recommendations = get_product_recommendations(df_sp, product_id, loaded_gensim)
            
            if recommendations.empty:
                st.error("No similar products found!")
            else:
                st.dataframe(recommendations if not recommendations.empty else "No similar products found.")
            
#############################
elif page == "Recommender System":
    st.header("Recommendation Functions")
    # Load product similarity matrix
    with open('src/models/gensim_model.pkl', 'rb') as f:
        loaded_gensim = pickle.load(f)

    # Add 2 tabs for Content-Base Filtering and Collaborative Filtering
    tab_containers = st.tabs(["Content-Based Filtering", "Collaborative Filtering"])
    with tab_containers[0]:
        st.markdown("""
            #### Mô tả:
            Hệ thống sẽ gợi ý những sản phẩm tương tự với những món bạn đã mua trước đây.
        """) 
        st.markdown("""
            **Tại sao nên sử dụng Content-Based Filtering?**
            - Dựa trên sở thích cá nhân để tạo ra các gợi ý phù hợp với bạn.
            - Gợi ý sản phẩm tương tự với những gì bạn đã yêu thích trước đó.
        """)
        st.markdown("""
            #### Tìm sản phẩm đề xuất
        """)

        option = st.radio("Chọn phương thức:", 
                        options=["Chọn tên sản phẩm", "Nhập mã/ tên sản phẩm"])

        product_codes = []
        if option == "Chọn tên sản phẩm":    
            selected_item = st.selectbox("Chọn tên sản phẩm:", df_products['ten_san_pham'].unique())
            product_codes = df_products[df_products["ten_san_pham"] == selected_item]["ma_san_pham"]

        elif option == "Nhập mã/ tên sản phẩm":
            search_criteria = st.text_input("Nhập mã hoặc tên sản phẩm:")

            example_product = df_products.sample(1)['ten_san_pham'].values[0]
            example_code = df_products.sample(1)['ma_san_pham'].values[0]
            st.markdown(f"📝 **Ví dụ tên sản phẩm:** {example_product}  \n📝 **Ví dụ mã sản phẩm:** {example_code}")

            if (search_criteria != "" and search_criteria.isdigit()):
                result = df_products[df_products["ma_san_pham"] == eval(search_criteria)]
                if not result.empty:
                    product_codes = result["ma_san_pham"]
                else:
                    st.write("Không tìm thấy sản phẩm! Xly thêm tìm kiếm sản phẩm gì khác????")

            elif (search_criteria != ""):
                result = df_products[(df_products["ten_san_pham"].str.contains(search_criteria, case=False))]
                result = result.sort_values(by='diem_trung_binh', ascending=False).head(5)
                if not result.empty:
                    product_codes = result["ma_san_pham"]
                else:
                    st.write("Không tìm thấy sản phẩm!")

        st.markdown(button_style, unsafe_allow_html=True)
        if st.button("Hiển thị sản phẩm"):

            if len(product_codes) == 0:
                st.warning("Vui lòng chọn hoặc nhập mã sản phẩm trước khi nhấn 'Hiển thị sản phẩm'.")
            else:
                with st.spinner("Đang tải gợi ý sản phẩm..."):
                    recommendations = filters.get_product_recommendations(df_products, product_codes, loaded_gensim)

                if recommendations.empty:
                    st.error("Không tìm thấy sản phẩm tương tự!")
                else:
                    st.subheader("✨ Gợi ý sản phẩm tương tự")
                    cols = st.columns(2)  # Display 2 items per row
                    for i, (_, row) in enumerate(recommendations.iterrows()):
                        with cols[i % 2]:
                            st.markdown(f"**Tên sản phẩm:** {row['ten_san_pham']}")
                            st.write(f"💵 Giá: {row['gia_ban']} VNĐ")
                            st.write(f"⭐ Đánh giá: {row['diem_trung_binh']} / 5")
                            st.markdown("---")

    with tab_containers[1]:
        st.markdown("""
            #### Mô tả: 
            Hệ thống sẽ gợi ý sản phẩm dựa trên sở thích và thói quen của bạn và những người dùng khác có cùng sở thích.
        """)

        st.markdown("""
            **Tại sao nên sử dụng Collaborative Filtering?**
            - Tận dụng hoạt động của bạn để tạo ra các gợi ý sản phẩm phù hợp.
            - Khám phá các kết nối tiềm ẩn giữa người dùng và sản phẩm để mang đến những trải nghiệm tốt nhất cho bạn.
        """)

        st.markdown("""
            #### Tìm sản phẩm đề xuất:
            Hãy chọn một tùy chọn bên dưới để bắt đầu:
        """)

        option_cf = st.radio("Choose an option:",
            options=["Tìm gợi ý sản phẩm cho bạn", "Xem các sản phẩm đang được quan tâm"])

        # Prepare data
        customer_ids = df['ma_khach_hang'].unique()
        suprise_svdpp_algorithm = filters.load_model(df)

        if option_cf == "Tìm gợi ý sản phẩm cho bạn":
            if "customer_id" not in st.session_state:
                st.session_state.customer_id = ""

            # Login Form
            if st.session_state.customer_id == "" or st.session_state.customer_id not in customer_ids:
                with st.form("login_form"):
                    customer_id = st.text_input(
                    "Nhập Mã Khách Hàng:",
                    value=st.session_state.customer_id).strip()
                    st.markdown("""
                        🗒 **Example User ID:** 2888
                    """)
                    submitted = st.form_submit_button("Login")
                    if submitted:
                        try:
                            customer_id = int(customer_id)
                            if customer_id not in customer_ids:
                                st.error("Customer ID not found!")
                            else:
                                st.session_state.customer_id = customer_id
                        except ValueError:
                            st.error("Invalid Customer ID!")
            else:
                if st.button("Logout"):
                    st.session_state.customer_id = ""

            ##########################################

            if st.session_state.customer_id != "":
                try:
                    st.write(f"Welcome back! Customer code {st.session_state.customer_id}")
                    # Display customer history
                    st.subheader("📜 Customer Shopping History")
                    history = filters.get_customer_history(st.session_state.customer_id, df, df_products)
                    if not history.empty:
                        st.dataframe(history)
                    else:
                        st.info("No history available.")

                    # Recommendations
                    recommendations = filters.recommend_products_with_names(suprise_svdpp_algorithm, st.session_state.customer_id, df, df_products)

                    if recommendations.empty:
                        st.error("No similar products found!")
                    else:
                        st.subheader("✨ Recommendation(s)")
                        cols = st.columns(2)  # Display 2 items per row
                        for i, (_, row) in enumerate(recommendations.iterrows()):
                            with cols[i % 2]:
                                st.markdown(f"**Tên sản phẩm:** {row['ten_san_pham']}")
                                st.write(f"💵 Giá: {row['gia_ban']} VNĐ")
                                st.write(f"⭐ Đánh giá: {row['diem_trung_binh']} / 5")
                                st.markdown("---")
                except ValueError:
                    st.error("Invalid Customer ID!")

        elif option_cf == "Xem các sản phẩm đang được quan tâm":
            trending_products = df_products[df_products['diem_trung_binh'] > 4.5].sort_values(by='diem_trung_binh', ascending=False).head(10)
            trending_product_ids = trending_products['ma_san_pham'].unique()

            with st.spinner("Đang tải gợi ý sản phẩm..."):
                recommendations = filters.get_product_recommendations(df_products, trending_product_ids.tolist(), loaded_gensim)

            if recommendations.empty:
                st.error("Không tìm thấy sản phẩm tương tự!")
            else:
                st.subheader("✨ Gợi ý sản phẩm đang được quan tâm")
                cols = st.columns(2)  # Display 2 items per row
                for i, (_, row) in enumerate(recommendations.iterrows()):
                    with cols[i % 2]:
                        st.markdown(f"**Tên sản phẩm:** {row['ten_san_pham']}")
                        st.write(f"💵 Giá: {row['gia_ban']} VNĐ")
                        st.write(f"⭐ Đánh giá: {row['diem_trung_binh']} / 5")
                        st.markdown("---")

    # with tab_containers[0]:
    #     st.markdown("""
    #         #### Mô tả:
    #         Hệ thống sẽ gợi ý những sản phẩm tương tự với những món bạn đã mua trước đây.
    #     """) 
    #     st.markdown("""
    #         **Tại sao nên sử dụng Content-Based Filtering?**
    #         - Dựa trên sở thích cá nhân để tạo ra các gợi ý phù hợp với bạn.
    #         - Gợi ý sản phẩm tương tự với những gì bạn đã yêu thích trước đó.
    #     """)
    #     st.markdown("""
    #         #### Tìm sản phẩm đề xuất
    #     """)
        
    #     option = st.radio("Chọn phương thức:", 
    #                       options=["Chọn tên sản phẩm", "Nhập mã/ tên sản phẩm"])
        
    #     product_codes = []
    #     if option == "Chọn tên sản phẩm":    
    #         selected_item = st.selectbox("Chọn tên sản phẩm:", df_products['ten_san_pham'].unique())
    #         product_codes = df_products[df_products["ten_san_pham"] == selected_item]["ma_san_pham"]

    #     elif option == "Nhập mã/ tên sản phẩm":
    #         search_criteria = st.text_input("Nhập mã hoặc tên sản phẩm:")

    #         # Add sample text for product name and product code
    #         # st.markdown("📝 **Ví dụ tên sản phẩm:** Nước Hoa Hồng Klairs  \n📝 **Ví dụ mã sản phẩm:** 318900012")
        
    #         example_product = df_products.sample(1)['ten_san_pham'].values[0]
    #         example_code = df_products.sample(1)['ma_san_pham'].values[0]
    #         st.markdown(f"📝 **Ví dụ tên sản phẩm:** {example_product}  \n📝 **Ví dụ mã sản phẩm:** {example_code}")

    #         if (search_criteria != "" and search_criteria.isdigit()):
    #             result = df_products[df_products["ma_san_pham"] == eval(search_criteria)]
    #             if not result.empty:
    #                 product_codes = result["ma_san_pham"]
    #             else:
    #                 st.write("Không tìm thấy sản phẩm! Xly thêm tìm kiếm sản phẩm gì khác????")
                
    #         elif (search_criteria != ""):
    #             result = df_products[(df_products["ten_san_pham"].str.contains(search_criteria, case=False))]
    #             result = result.sort_values(by='diem_trung_binh', ascending=False).head(5)
    #             if not result.empty:
    #                 product_codes = result["ma_san_pham"]
    #             else:
    #                 st.write("Không tìm thấy sản phẩm!")

    #     st.markdown(button_style, unsafe_allow_html=True)
    #     if st.button("Hiển thị sản phẩm"):

    #         if len(product_codes) == 0:
    #             st.warning("Vui lòng chọn hoặc nhập mã sản phẩm trước khi nhấn 'Hiển thị sản phẩm'.")
    #         else:
    #             with st.spinner("Đang tải gợi ý sản phẩm..."):
    #                 recommendations = filters.get_product_recommendations(df_products, product_codes, loaded_gensim)

    #             if recommendations.empty:
    #                 st.error("Không tìm thấy sản phẩm tương tự!")
    #             else:
    #                 st.subheader("✨ Gợi ý sản phẩm tương tự")
    #                 for _, row in recommendations.iterrows():
    #                     helpers.load_item_template(row)
    

    # with tab_containers[1]:
    #     st.markdown("""
    #         #### Mô tả: 
    #         Hệ thống sẽ gợi ý sản phẩm dựa trên sở thích và thói quen của bạn và những người dùng khác có cùng sở thích.
    #     """)

    #     st.markdown("""
    #         **Tại sao nên sử dụng Collaborative Filtering?**
    #         - Tận dụng hoạt động của bạn để tạo ra các gợi ý sản phẩm phù hợp.
    #         - Khám phá các kết nối tiềm ẩn giữa người dùng và sản phẩm để mang đến những trải nghiệm tốt nhất cho bạn.
    #     """)

    #     st.markdown("""
    #         #### Tìm sản phẩm đề xuất:
    #         Hãy chọn một tùy chọn bên dưới để bắt đầu:
    #     """)

    #     option_cf = st.radio("Choose an option:",
    #         options=["Tìm gợi ý sản phẩm cho bạn", "Xem các sản phẩm đang được quan tâm"])

    #     # Prepare data
    #     customer_ids = df['ma_khach_hang'].unique()
    #     suprise_svdpp_algorithm = filters.load_model(df)

    #     if option_cf == "Tìm gợi ý sản phẩm cho bạn":
    #         if "customer_id" not in st.session_state:
    #             st.session_state.customer_id = ""

    #         # Login Form
    #         if st.session_state.customer_id == "" or st.session_state.customer_id not in customer_ids:
    #             with st.form("login_form"):
    #                 customer_id = st.text_input(
    #                 "Nhập Mã Khách Hàng:",
    #                 value=st.session_state.customer_id).strip()
    #                 st.markdown("""
    #                     🗒 **Example User ID:** 2888
    #                 """)
    #                 submitted = st.form_submit_button("Login")
    #                 if submitted:
    #                     try:
    #                         customer_id = int(customer_id)
    #                         if customer_id not in customer_ids:
    #                             st.error("Customer ID not found!")
    #                         else:
    #                             st.session_state.customer_id = customer_id
    #                     except ValueError:
    #                         st.error("Invalid Customer ID!")
    #         else:
    #             if st.button("Logout"):
    #                 st.session_state.customer_id = ""

    #         ##########################################

    #         if st.session_state.customer_id != "":
    #             try:
    #                 st.write(f"Welcome back! Customer code {st.session_state.customer_id}")
    #                 # Display customer history
    #                 st.subheader("📜 Customer Shopping History")
    #                 history = filters.get_customer_history(st.session_state.customer_id, df, df_products)
    #                 if not history.empty:
    #                     st.dataframe(history)
    #                 else:
    #                     st.info("No history available.")

    #                 # Recommendations
    #                 recommendations = filters.recommend_products_with_names(suprise_svdpp_algorithm, st.session_state.customer_id, df, df_products)

    #                 if recommendations.empty:
    #                     st.error("No similar products found!")
    #                 else:
    #                     st.subheader("✨ Recommendation(s)")
    #                     cols = st.columns(2)  # Display 2 items per row
    #                     for i, (_, row) in enumerate(recommendations.iterrows()):
    #                         with cols[i % 2]:
    #                             st.markdown(f"**Tên sản phẩm:** {row['ten_san_pham']}")
    #                             st.write(f"💵 Giá: {row['gia_ban']} VNĐ")
    #                             st.write(f"⭐ Đánh giá: {row['diem_trung_binh']} / 5")
    #                             st.markdown("---")
    #             except ValueError:
    #                 st.error("Invalid Customer ID!")

    #     elif option_cf == "Xem các sản phẩm đang được quan tâm":
    #         trending_products = df_products[df_products['diem_trung_binh'] > 4.5].sort_values(by='diem_trung_binh', ascending=False).head(10)
    #         trending_product_ids = trending_products['ma_san_pham'].unique()

    #         with st.spinner("Đang tải gợi ý sản phẩm..."):
    #             recommendations = filters.get_product_recommendations(df_products, trending_product_ids.tolist(), loaded_gensim)

    #         if recommendations.empty:
    #             st.error("Không tìm thấy sản phẩm tương tự!")
    #         else:
    #             st.subheader("✨ Gợi ý sản phẩm đang được quan tâm")
    #             cols = st.columns(2)  # Display 2 items per row
    #             for i, (_, row) in enumerate(recommendations.iterrows()):
    #                 with cols[i % 2]:
    #                     st.markdown(f"**Tên sản phẩm:** {row['ten_san_pham']}")
    #                     st.write(f"💵 Giá: {row['gia_ban']} VNĐ")
    #                     st.write(f"⭐ Đánh giá: {row['diem_trung_binh']} / 5")
    #                     st.markdown("---")


            # recommendations = filters.get_product_recommendations(df_products, product_codes, loaded_gensim)
            
            # if recommendations.empty:
            #     st.error("No similar products found!")
            # else:
            #     st.subheader("✨ Recommendation(s)")
            #     for _, row in recommendations.iterrows():
            #         helpers.load_item_template(row)
    # with tab_containers[1]:
    #     st.markdown("""
    #         #### Mô tả: 
    #         Hệ thống sẽ gợi ý sản phẩm dựa trên sở thích và thói quen của bạn và những người dùng khác có cùng sở thích.
    #     """)

    #     st.markdown("""
    #         **Tại sao nên sử dụng Collaborative Filtering?**
    #         - Tận dụng hoạt động của bạn để tạo ra các gợi ý sản phẩm phù hợp.
    #         - Khám phá các kết nối tiềm ẩn giữa người dùng và sản phẩm để mang đến những trải nghiệm tốt nhất cho bạn.
    #     """)

    #     st.markdown("""
    #         #### Tìm sản phẩm đề xuất:
    #         Hãy chọn một tùy chọn bên dưới để bắt đầu:
    #     """)

    #     option_cf = st.radio("Choose an option:",
    #         options=["Tìm gợi ý sản phẩm cho bạn", "Xem các sản phẩm đang được quan tâm"])

    #     # Prepare data
    #     customer_ids = df['ma_khach_hang'].unique()
    #     suprise_svdpp_algorithm = filters.load_model(df)

    #     if option_cf == "Tìm gợi ý sản phẩm cho bạn":
    #         if "customer_id" not in st.session_state:
    #             st.session_state.customer_id = ""

    #         # Login Form
    #         if st.session_state.customer_id == "" or st.session_state.customer_id not in customer_ids:
    #             with st.form("login_form"):
    #                 customer_id = st.text_input(
    #                 "Nhập Mã Khách Hàng:",
    #                 value=st.session_state.customer_id).strip()
    #                 st.markdown("""
    #                     🗒 **Example User ID:** 2888
    #                 """)
    #                 submitted = st.form_submit_button("Login")
    #                 if submitted:
    #                     try:
    #                         customer_id = int(customer_id)
    #                         if customer_id not in customer_ids:
    #                             st.error("Customer ID not found!")
    #                         else:
    #                             st.session_state.customer_id = customer_id
    #                     except ValueError:
    #                         st.error("Invalid Customer ID!")
    #         else:
    #             if st.button("Logout"):
    #                 st.session_state.customer_id = ""

    #         ##########################################

    #         if st.session_state.customer_id != "":
    #             try:
    #                 st.write(f"Welcome back! Customer code {st.session_state.customer_id}")
    #                 # Display customer history
    #                 st.subheader("📜 Customer Shopping History")
    #                 history = filters.get_customer_history(st.session_state.customer_id, df, df_products)
    #                 st.dataframe(history if not history.empty else "No history available.")

    #                 # Recommendations
    #                 recommendations = filters.recommend_products_with_names(suprise_svdpp_algorithm, st.session_state.customer_id, df, df_products)

    #                 if recommendations.empty:
    #                     st.error("No similar products found!")
    #                 else:
    #                     st.subheader("✨ Recommendation(s)")
    #                     for _, row in recommendations.iterrows():
    #                         helpers.load_item_template(row)
    #             except ValueError:
    #                 st.error("Invalid Customer ID!")

    #     elif option_cf == "Xem các sản phẩm đang được quan tâm":
    #         trending_products = df_products[df_products['diem_trung_binh'] > 4.5].sort_values(by='diem_trung_binh', ascending=False).head(10)
    #         trending_product_ids = trending_products['ma_san_pham'].unique()

    #         with st.spinner("Đang tải gợi ý sản phẩm..."):
    #             recommendations = filters.get_product_recommendations(df_products, trending_product_ids.tolist(), loaded_gensim)

    #         if recommendations.empty:
    #             st.error("Không tìm thấy sản phẩm tương tự!")
    #         else:
    #             st.subheader("✨ Gợi ý sản phẩm đang được quan tâm")
    #             for _, row in recommendations.iterrows():
    #                 helpers.load_item_template(row)

    # # with tab_containers[1]:
    #         # --- Collaborative Filtering ---
    # with tab_containers[1]:
    #     st.markdown("""
    #         #### Mô tả: 
    #         Hệ thống sẽ gợi ý sản phẩm dựa trên sở thích và thói quen của bạn và những người dùng khác có cùng sở thích.
    #     """)

    #     st.markdown("""
    #         **Tại sao nên sử dụng Collaborative Filtering?**
    #         - Tận dụng hoạt động của bạn để tạo ra các gợi ý sản phẩm phù hợp.
    #         - Khám phá các kết nối tiềm ẩn giữa người dùng và sản phẩm để mang đến những trải nghiệm tốt nhất cho bạn.
    #     """)

    #     st.markdown("""
    #         #### Tìm sản phẩm đề xuất:
    #         Hãy chọn một tùy chọn bên dưới để bắt đầu:
    #     """)

    #     option_cf = st.radio("Choose an option:",
    #         options=["Tìm gợi ý sản phẩm cho bạn", "Xem các sản phẩm đang được quan tâm"])

    #     # Prepare data
    #     customer_ids = df['ma_khach_hang'].unique()
    #     suprise_svdpp_algorithm = filters.load_model(df)

    #     if option_cf == "Find recommendations for a user":
    #         if "customer_id" not in st.session_state:
    #             st.session_state.customer_id = ""

    #         # Login Form
    #         if st.session_state.customer_id == "" or st.session_state.customer_id not in customer_ids:
    #             with st.form("login_form"):
    #                 customer_id = st.text_input(
    #                 "Nhập Mã Khách Hàng:",
    #                 value=st.session_state.customer_id).strip()
    #                 st.markdown("""
    #                     📝 **Example User ID:** 2888
    #                 """)
    #                 submitted = st.form_submit_button("Login")
    #                 if submitted:
    #                     try:
    #                         customer_id = int(customer_id)
    #                         if customer_id not in customer_ids:
    #                             st.error("Customer ID not found!")
    #                         else:
    #                             st.session_state.customer_id = customer_id
    #                     except ValueError:
    #                         st.error("Invalid Customer ID!")
    #         else:
    #             if st.button("Logout"):
    #                 st.session_state.customer_id = ""

    #         ##########################################
            
    #         if st.session_state.customer_id != "":
    #             try:
    #                 st.write(f"Welcome back! Customer code {st.session_state.customer_id}")
    #                 # Display customer history
    #                 st.subheader("📜 Customer Shopping History")
    #                 history = filters.get_customer_history(st.session_state.customer_id, df, df_products)
    #                 st.dataframe(history if not history.empty else "No history available.")
                    
    #                 # Recommendations
    #                 recommendations = filters.recommend_products_with_names(suprise_svdpp_algorithm, st.session_state.customer_id, df, df_products)
    #                 # st.dataframe(recommendations if not recommendations.empty else "No recommendations available.")

    #                 if recommendations.empty:
    #                     st.error("No similar products found!")
    #                 else:
    #                     st.subheader("✨ Recommendation(s)")
    #                     for _, row in recommendations.iterrows():
    #                         helpers.load_item_template(row)
    #             except ValueError:
    #                 st.error("Invalid Customer ID!")

    #     elif option_cf == "View trending products":
    #         trending_products = df_products[df_products['diem_trung_binh'] > 4.5].sort_values(by='diem_trung_binh', ascending=False).head(10)
    #         trending_product_ids = trending_products['ma_san_pham'].unique()

    #         # Get trending top 5 products with diem_trung_binh > 4.5
    #         list_product_ids = df_products[df_products['diem_trung_binh'] > 4.5]['ma_san_pham'].unique()
    #         trending_product_ids = pd.Series(list_product_ids).sample(n=10)
    #         recommendations = filters.get_product_recommendations(df_products, product_codes, loaded_gensim)
    #         with st.spinner("Đang tải gợi ý sản phẩm..."):
    #              recommendations = filters.get_product_recommendations(df_products, product_codes, loaded_gensim)

    #         # recommendations = filters.get_product_recommendations(df_products, trending_product_ids.tolist(), loaded_gensim)
            
    #         if recommendations.empty:
    #             st.error("No similar products found!")
    #         else:
    #             st.subheader("✨ Trending Recommendation(s)")
    #             for _, row in recommendations.iterrows():
    #                 helpers.load_item_template(row)