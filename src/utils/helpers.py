import pandas as pd
from PIL import Image
import streamlit as st

def read_csv(file_path):
    return pd.read_csv(file_path, encoding='utf-8')

def load_item_template(data):
    with st.container():
        cols = st.columns([1, 2, 1])
        with cols[0]:
            cols_0 = st.columns([1,3])
            with cols_0[1]:
                try:
                    image_path = "src/images/default_sample.png"
                    image = Image.open(image_path)
                    st.image(image, use_container_width=True)
                except FileNotFoundError:
                    st.error(f"Image not found at {image_path}")
        with cols[1]:
            st.markdown(f"##### {data['ten_san_pham']}")
            st.markdown(f"**Mã sản phẩm**: {data['ma_san_pham']}")
            st.markdown(f"**Điểm trung bình**: {data['diem_trung_binh']:.2f}")
            st.markdown(f"**Giá bán**: {data['gia_ban']:,} VND")
            expander = st.expander("Chi tiết sản phẩm")
            expander.write(data.get('mo_ta', "Không có mô tả."))

def load_item_template_block(data):
    with st.container():
        cols = st.columns([1, 2, 1])
        with cols[0]:
            cols_0 = st.columns([1,3])
            with cols_0[1]:
                try:
                    image_path = "src/images/default_sample.png"
                    image = Image.open(image_path)
                    st.image(image, use_container_width=True)
                except FileNotFoundError:
                    st.error(f"Image not found at {image_path}")
        with cols[1]:
            st.markdown(f"##### {data['ten_san_pham']}")
            st.markdown(f"**Mã sản phẩm**: {data['ma_san_pham']}")
            st.markdown(f"**Điểm trung bình**: {data['diem_trung_binh']:.2f}")
            st.markdown(f"**Giá bán**: {data['gia_ban']:,} VND")
            expander = st.expander("Chi tiết sản phẩm")
            expander.write(data.get('mo_ta', "Không có mô tả."))