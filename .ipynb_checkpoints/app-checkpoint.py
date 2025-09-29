import streamlit as st
import pandas as pd
from src.recommend import recommend_for_user

st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")

st.title("🎬 Movie Recommender System")
st.markdown("Compare **RBM vs Autoencoder** trained on MovieLens 1M dataset.")

user_id = st.number_input("Enter User ID:", min_value=0, max_value=6040, value=1)
model_type = st.radio("Choose Model:", ["rbm", "ae"])
top_k = st.slider("Top K Recommendations", min_value=5, max_value=20, value=10)

if st.button("Recommend"):
    recs = recommend_for_user(user_id, model_type=model_type, top_k=top_k)
    st.subheader(f"Top {top_k} Recommendations for User {user_id} ({model_type.upper()})")
    st.table(recs)
