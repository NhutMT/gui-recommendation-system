import pandas as pd
# Customer Recommendations
def recommend_products_with_names(algorithm, cust_code, df, df_products, n_recommendations=5):
    all_products = df['ma_san_pham'].unique()
    rated_products = df[df['ma_khach_hang'] == cust_code]['ma_san_pham'].unique()
    unrated_products = [product for product in all_products if product not in rated_products]

    predictions = []
    for ma_san_pham in unrated_products:
        prediction = algorithm.predict(cust_code, ma_san_pham)
        predictions.append((ma_san_pham, prediction.est))

    predictions.sort(key=lambda x: x[1], reverse=True)
    top_recommendations = predictions[:n_recommendations]

    recommended_df = pd.DataFrame(top_recommendations, columns=['ma_san_pham', 'predicted_rating'])
    recommended_with_names = recommended_df.merge(df_products, on='ma_san_pham', how='left')
    return recommended_with_names[['ma_san_pham', 'ten_san_pham', 'predicted_rating']]

# Product-Based Recommendations
def get_product_recommendations(df_products, product_codes, gensim_model, n_recommendations=5):
    idxes = df_products.index[df_products['ma_san_pham'].isin(list(product_codes))].tolist()
    product_indices = []
    for idx in idxes:
        sim_scores = list(enumerate(gensim_model[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[:10]
        for i in sim_scores:
            # print("wat is : i", i)
            product_indices.append(i[0])

    recommended_products = df_products.iloc[product_indices]
    recommended_products = recommended_products.sort_values(by='diem_trung_binh', ascending=False).head(n_recommendations)
    return recommended_products

# Customer History
def get_customer_history(cust_code, df, df_products):
    customer_history = df[df['ma_khach_hang'] == cust_code]
    customer_history_with_names = customer_history.merge(df_products, on='ma_san_pham', how='left')
    return customer_history_with_names[['ma_san_pham', 'ten_san_pham', 'so_sao']]
