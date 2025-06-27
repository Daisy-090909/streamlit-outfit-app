import streamlit as st
from outfit_recommender import OutfitRecommender
from PIL import Image
import os

# Initialize recommender (cached for performance)
@st.cache_resource
def get_recommender():
    return OutfitRecommender()

recommender = get_recommender()

# Streamlit UI
st.title("👗 AI Personal Stylist")
query = st.text_input("Describe your style and occasion:")

if st.button("Get Recommendation"):
    if not query.strip():
        st.warning("Please describe your style and occasion")
    else:
        with st.spinner("Finding the perfect outfit..."):
            result = recommender.recommend(query)
        
        if not result["outfit_details"]:
            st.error("No matching outfits found")
        else:
            # Display recommendation
            st.subheader("💡 AI Recommendation")
            st.write(result["outfit_recommendation"])
            # Display outfit items
            st.subheader("🛍️ Recommended Outfit")
            details = result["outfit_details"]
            
            cols = st.columns(3)
            for item_type in ["top", "bottom", "shoes"]:
                description_key = f"{item_type}_description"
                image_key = f"{item_type}_image"

                # Fallback-safe access
                description = result["outfit_details"]['metadata'].get(description_key, f"No description for {item_type}.")
                image_url = result["outfit_details"]['metadata'].get(image_key)

                st.subheader(f"{item_type.title()}")
                st.markdown(description)

                if image_url:
                    st.image(image_url, width=300)

