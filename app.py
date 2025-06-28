# Import necessary libraries
import streamlit as st                    # For creating the web app UI
from outfit_recommender import OutfitRecommender  # Your custom recommendation class
from PIL import Image                    # For image processing (optional use here)
import os                                # For environment variables or file paths

# Cache the model instance to avoid reloading on every interaction
@st.cache_resource
def get_recommender():
    return OutfitRecommender()

# Instantiate the recommender
recommender = get_recommender()

# ---------- Streamlit User Interface (UI) ---------- #

# Set the app title
st.title("👗 AI Personal Stylist")

# User input for style and occasion
query = st.text_input("Describe your style and occasion:")

# When "Get Recommendation" button is clicked
if st.button("Get Recommendation"):

    # Warn the user if input is empty
    if not query.strip():
        st.warning("Please describe your style and occasion")
    else:
        # Display a spinner while processing
        with st.spinner("Finding the perfect outfit..."):
            result = recommender.recommend(query)  # Call the model
        
        # Show error if no items were found
        if not result["outfit_details"]:
            st.error("No matching outfits found")
        else:
            # Display generated outfit recommendation text
            st.subheader("💡 AI Recommendation")
            st.write(result["outfit_recommendation"])

            # Display associated outfit item metadata (text + images)
            st.subheader("🛍️ Recommended Outfit")
            details = result["outfit_details"]
            
            # Create three columns for top, bottom, and shoes
            cols = st.columns(3)

            # Loop through each outfit component
            for item_type in ["top", "bottom", "shoes"]:
                description_key = f"{item_type}_description"
                image_key = f"{item_type}_image"

                # Retrieve text description and image URL (with fallback)
                description = result["outfit_details"]['metadata'].get(
                    description_key, f"No description for {item_type}."
                )
                image_url = result["outfit_details"]['metadata'].get(image_key)

                # Display the section title and description
                st.subheader(f"{item_type.title()}")
                st.markdown(description)

                # Display image if available
                if image_url:
                    st.image(image_url, width=300)
