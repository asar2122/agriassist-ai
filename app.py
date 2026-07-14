import streamlit as st
import pandas as pd

from PIL import Image


from chatbot.chatbot import AgriChatbot
from crop_recommendation.predict import CropPredictor
from fertilizer_recommendation.predict import FertilizerPredictor
from disease_detection.predict import DiseasePredictor
from market_prediction.predict import MarketPricePredictor

from database.database import (
    initialize_database,
    save_message,
    get_chat_history,
)


# --------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AgriAssist AI",
    page_icon="🌱",
    layout="wide",
)


# --------------------------------------------------
# DATABASE INITIALIZATION
# --------------------------------------------------

initialize_database()


# --------------------------------------------------
# LOAD CHATBOT
# --------------------------------------------------

@st.cache_resource
def load_chatbot():
    return AgriChatbot()


chatbot = load_chatbot()


# --------------------------------------------------
# LOAD CROP RECOMMENDATION MODEL
# --------------------------------------------------

@st.cache_resource
def load_crop_predictor():
    return CropPredictor()


crop_predictor = load_crop_predictor()


# --------------------------------------------------
# LOAD FERTILIZER RECOMMENDATION MODEL
# --------------------------------------------------

@st.cache_resource
def load_fertilizer_predictor():
    return FertilizerPredictor()


fertilizer_predictor = load_fertilizer_predictor()


@st.cache_resource
def load_disease_predictor():
    return DiseasePredictor()


disease_predictor = load_disease_predictor()


# --------------------------------------------------
# LOAD MARKET PRICE MODEL
# --------------------------------------------------

@st.cache_resource
def load_market_predictor():
    return MarketPricePredictor()


market_predictor = load_market_predictor()


# --------------------------------------------------
# LOAD MARKET DATA
# --------------------------------------------------

@st.cache_data
def load_market_data():
    return pd.read_csv(
        "data/market_data.csv"
    )


market_data = load_market_data()


# --------------------------------------------------
# MAIN APPLICATION
# --------------------------------------------------

def main():

    st.title("🌱 AgriAssist AI")

    st.write(
        "AI-Powered Smart Agriculture Assistant"
    )

    menu = st.sidebar.selectbox(
        "Select Feature",
        [
            "Agriculture Chatbot",
            "Crop Recommendation",
            "Disease Detection",
            "Fertilizer Recommendation",
            "Market Prediction",
            "Chat History",
        ],
    )

    if menu == "Agriculture Chatbot":

        chatbot_page()

    elif menu == "Crop Recommendation":

        crop_recommendation_page()

    elif menu == "Disease Detection":

        disease_detection_page()

    elif menu == "Fertilizer Recommendation":

        fertilizer_recommendation_page()

    elif menu == "Market Prediction":

        market_prediction_page()

    elif menu == "Chat History":

        chat_history_page()


# --------------------------------------------------
# AGRICULTURE CHATBOT PAGE
# --------------------------------------------------

def chatbot_page():

    st.header(
        "🤖 Agriculture Chatbot"
    )

    st.write(
    "Answers are generated using agriculture "
    "knowledge retrieval and cloud AI."
)

    user_question = st.chat_input(
        "Ask an agriculture question..."
    )

    if user_question:

        # Display user message

        with st.chat_message("user"):

            st.write(
                user_question
            )


        # Generate and display chatbot response

        with st.chat_message("assistant"):

            with st.spinner(
                "Searching agriculture knowledge..."
            ):

                response = chatbot.get_response(
                    user_question
                )


            # Display chatbot answer

            st.write(
                response["answer"]
            )


            # Display simple knowledge-base information

            st.caption(
                "Answer generated from the local "
                "agriculture knowledge base."
            )


        # Save conversation to SQLite

        save_message(
            user_message=user_question,
            bot_message=response["answer"],
        )

# --------------------------------------------------
# CROP RECOMMENDATION PAGE
# --------------------------------------------------

def crop_recommendation_page():

    st.header(
        "🌾 Crop Recommendation"
    )

    st.write(
        "Enter soil nutrient values and environmental "
        "conditions to receive a crop recommendation."
    )

    with st.form(
        "crop_recommendation_form"
    ):

        st.subheader(
            "🌱 Soil Nutrient Information"
        )

        nitrogen = st.number_input(
            "Nitrogen (N)",
            min_value=0.0,
            max_value=200.0,
            value=90.0,
        )

        phosphorus = st.number_input(
            "Phosphorus (P)",
            min_value=0.0,
            max_value=200.0,
            value=42.0,
        )

        potassium = st.number_input(
            "Potassium (K)",
            min_value=0.0,
            max_value=300.0,
            value=43.0,
        )

        st.subheader(
            "🌦 Environmental Information"
        )

        temperature = st.number_input(
            "Temperature (°C)",
            min_value=-10.0,
            max_value=60.0,
            value=20.8,
        )

        humidity = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=82.0,
        )

        ph = st.number_input(
            "Soil pH",
            min_value=0.0,
            max_value=14.0,
            value=6.5,
        )

        rainfall = st.number_input(
            "Rainfall (mm)",
            min_value=0.0,
            max_value=1000.0,
            value=202.9,
        )

        submit_button = st.form_submit_button(
            "🌱 Recommend Crop"
        )

    if submit_button:

        with st.spinner(
            "Analyzing soil and environmental conditions..."
        ):

            prediction = crop_predictor.predict(
                nitrogen=nitrogen,
                phosphorus=phosphorus,
                potassium=potassium,
                temperature=temperature,
                humidity=humidity,
                ph=ph,
                rainfall=rainfall,
            )

        st.success(
            f"🌱 Recommended Crop: "
            f"{prediction.title()}"
        )

        st.info(
            "The recommendation was generated by "
            "the locally trained Random Forest model."
        )

        with st.expander(
            "📊 Input Details"
        ):

            st.write(
                f"Nitrogen (N): {nitrogen}"
            )

            st.write(
                f"Phosphorus (P): {phosphorus}"
            )

            st.write(
                f"Potassium (K): {potassium}"
            )

            st.write(
                f"Temperature: {temperature} °C"
            )

            st.write(
                f"Humidity: {humidity} %"
            )

            st.write(
                f"Soil pH: {ph}"
            )

            st.write(
                f"Rainfall: {rainfall} mm"
            )


# --------------------------------------------------
# FERTILIZER RECOMMENDATION PAGE
# --------------------------------------------------

def fertilizer_recommendation_page():

    st.header(
        "💊 Fertilizer Recommendation"
    )

    st.write(
        "Enter soil, crop, nutrient, and environmental "
        "information to receive a fertilizer recommendation."
    )

    with st.form(
        "fertilizer_recommendation_form"
    ):

        st.subheader(
            "🌦 Environmental Information"
        )

        temperature = st.number_input(
            "Temperature (°C)",
            min_value=-10.0,
            max_value=60.0,
            value=26.0,
            key="fertilizer_temperature",
        )

        humidity = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=52.0,
            key="fertilizer_humidity",
        )

        moisture = st.number_input(
            "Soil Moisture (%)",
            min_value=0.0,
            max_value=100.0,
            value=38.0,
        )

        st.subheader(
            "🌱 Soil Nutrient Information"
        )

        nitrogen = st.number_input(
            "Nitrogen (N)",
            min_value=0.0,
            max_value=200.0,
            value=37.0,
            key="fertilizer_nitrogen",
        )

        potassium = st.number_input(
            "Potassium (K)",
            min_value=0.0,
            max_value=300.0,
            value=0.0,
            key="fertilizer_potassium",
        )

        phosphorus = st.number_input(
            "Phosphorus (P)",
            min_value=0.0,
            max_value=200.0,
            value=0.0,
            key="fertilizer_phosphorus",
        )

        st.subheader(
            "🌾 Crop and Soil Information"
        )

        soil_type = st.selectbox(
            "Soil Type",
            [
                "Sandy",
                "Loamy",
                "Black",
                "Red",
                "Clayey",
            ],
        )

        crop_type = st.selectbox(
            "Crop Type",
            [
                "Maize",
                "Sugarcane",
                "Cotton",
                "Tobacco",
                "Paddy",
                "Barley",
                "Wheat",
            ],
        )

        submit_button = st.form_submit_button(
            "💊 Recommend Fertilizer"
        )

    if submit_button:

        with st.spinner(
            "Analyzing crop and soil conditions..."
        ):

            prediction = fertilizer_predictor.predict(
                temperature=temperature,
                humidity=humidity,
                moisture=moisture,
                nitrogen=nitrogen,
                potassium=potassium,
                phosphorus=phosphorus,
                soil_type=soil_type,
                crop_type=crop_type,
            )

        st.success(
            f"💊 Recommended Fertilizer: "
            f"{prediction}"
        )

        st.info(
            "The recommendation was generated by "
            "the locally trained Random Forest model."
        )

        with st.expander(
            "📊 Input Details"
        ):

            st.write(
                f"Temperature: {temperature} °C"
            )

            st.write(
                f"Humidity: {humidity} %"
            )

            st.write(
                f"Soil Moisture: {moisture} %"
            )

            st.write(
                f"Nitrogen (N): {nitrogen}"
            )

            st.write(
                f"Potassium (K): {potassium}"
            )

            st.write(
                f"Phosphorus (P): {phosphorus}"
            )

            st.write(
                f"Soil Type: {soil_type}"
            )

            st.write(
                f"Crop Type: {crop_type}"
            )




# --------------------------------------------------
# DISEASE DETECTION PAGE
# --------------------------------------------------

def disease_detection_page():

    st.header(
        "🦠 Rice Disease Detection"
    )

    st.write(
        "Upload a rice leaf image to detect "
        "the possible disease."
    )

    st.caption(
        "The model can classify six categories: "
        "Bacterial Leaf Blight, Brown Spot, "
        "Healthy Rice Leaf, Leaf Blast, "
        "Leaf scald, and Sheath Blight."
    )

    uploaded_file = st.file_uploader(
        "Upload Rice Leaf Image",
        type=[
            "jpg",
            "jpeg",
            "png",
        ],
    )

    if uploaded_file is not None:

        st.image(
            uploaded_file,
            caption="Uploaded Rice Leaf Image",
            width=350,
        )

        detect_button = st.button(
            "🔍 Detect Disease"
        )

        if detect_button:

            with st.spinner(
                "Analyzing rice leaf image..."
            ):

                image = Image.open(
                    uploaded_file
                )

                result = disease_predictor.predict(
                    image
                )

            disease = result["disease"]

            confidence = (
                result["confidence"] * 100
            )

            st.success(
                f"🦠 Predicted Disease: "
                f"{disease}"
            )

            st.metric(
                label="Prediction Confidence",
                value=f"{confidence:.2f}%",
            )

            if disease == "Healthy Rice Leaf":

                st.info(
                    "🌱 The uploaded rice leaf was "
                    "classified as healthy."
                )

            else:

                st.warning(
                    "The uploaded rice leaf was "
                    "classified as a possible "
                    "disease case."
                )

# --------------------------------------------------
# MARKET PRICE PREDICTION PAGE
# --------------------------------------------------

def market_prediction_page():

    st.header(
        "💰 Market Price Estimation"
    )

    st.write(
        "Select market information and enter the "
        "minimum and maximum prices to estimate "
        "the modal market price."
    )

    st.caption(
        "The estimation is generated by a locally "
        "trained Random Forest regression model."
    )


    # --------------------------------------------------
    # PREPARE DROPDOWN OPTIONS
    # --------------------------------------------------

    months = sorted(
        market_data[
            "month"
        ].dropna().unique().tolist()
    )

    commodities = sorted(
        market_data[
            "commodity_name"
        ].dropna().unique().tolist()
    )

    states = sorted(
        market_data[
            "state_name"
        ].dropna().unique().tolist()
    )


    # --------------------------------------------------
    # MARKET FORM
    # --------------------------------------------------

    with st.form(
        "market_prediction_form"
    ):

        st.subheader(
            "🌾 Commodity Information"
        )


        month = st.selectbox(
            "Month",
            months,
        )


        commodity_name = st.selectbox(
            "Commodity",
            commodities,
        )


        state_name = st.selectbox(
            "State",
            states,
        )


        # --------------------------------------------------
        # FILTER DISTRICTS BY SELECTED STATE
        # --------------------------------------------------

        district_data = market_data[
            market_data["state_name"]
            == state_name
        ]


        districts = sorted(
            district_data[
                "district_name"
            ].dropna().unique().tolist()
        )


        district_name = st.selectbox(
            "District",
            districts,
        )


        st.subheader(
            "💵 Market Price Information"
        )


        avg_min_price = st.number_input(
            "Average Minimum Price",
            min_value=0.0,
            value=2000.0,
            step=100.0,
        )


        avg_max_price = st.number_input(
            "Average Maximum Price",
            min_value=0.0,
            value=2500.0,
            step=100.0,
        )


        submit_button = st.form_submit_button(
            "💰 Estimate Market Price"
        )


    # --------------------------------------------------
    # MAKE MARKET PRICE PREDICTION
    # --------------------------------------------------

    if submit_button:

        if avg_max_price < avg_min_price:

            st.error(
                "Average maximum price cannot be "
                "less than average minimum price."
            )

            return


        with st.spinner(
            "Analyzing market information..."
        ):

            prediction = market_predictor.predict(
                month=month,
                commodity_name=commodity_name,
                avg_min_price=avg_min_price,
                avg_max_price=avg_max_price,
                state_name=state_name,
                district_name=district_name,
            )


        st.success(
            f"💰 Estimated Modal Price: "
            f"₹{prediction:,.2f}"
        )


        st.info(
            "The price was estimated using the "
            "locally trained Random Forest "
            "regression model."
        )


        # --------------------------------------------------
        # DISPLAY INPUT INFORMATION
        # --------------------------------------------------

        with st.expander(
            "📊 Prediction Details"
        ):

            st.write(
                f"Month: {month}"
            )

            st.write(
                f"Commodity: {commodity_name}"
            )

            st.write(
                f"State: {state_name}"
            )

            st.write(
                f"District: {district_name}"
            )

            st.write(
                f"Average Minimum Price: "
                f"₹{avg_min_price:,.2f}"
            )

            st.write(
                f"Average Maximum Price: "
                f"₹{avg_max_price:,.2f}"
            )

            st.write(
                f"Estimated Modal Price: "
                f"₹{prediction:,.2f}"
            )

            st.write(
                "Model: Random Forest Regressor"
            )

# --------------------------------------------------
# CHAT HISTORY PAGE
# --------------------------------------------------

def chat_history_page():

    st.header(
        "📚 Chat History"
    )

    history = get_chat_history()

    if not history:

        st.info(
            "No chat history available."
        )

        return

    for row in history:

        st.write(
            f"**User:** {row[1]}"
        )

        st.write(
            f"**AgriAssist:** {row[2]}"
        )

        st.caption(
            f"Date: {row[3]}"
        )

        st.divider()


# --------------------------------------------------
# RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":

    main()