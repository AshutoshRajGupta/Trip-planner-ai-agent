
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
import requests
import datetime

# ----------------- External Tools -----------------
def get_weather(destination):
    """Fetch weather data using Open-Meteo API"""
    try:
        # Geocoding API to get lat/long
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={destination}&count=1"
        geo_res = requests.get(geo_url).json()

        if "results" not in geo_res:
            return f"🌍 Weather info for {destination} not found."

        lat = geo_res["results"][0]["latitude"]
        lon = geo_res["results"][0]["longitude"]

        # Fetch weather forecast (daily temperature)
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max,temperature_2m_min,weathercode&forecast_days=3&timezone=auto"
        weather_res = requests.get(weather_url).json()

        today = datetime.date.today()
        temps = []
        for i in range(len(weather_res["daily"]["time"])):
            date = weather_res["daily"]["time"][i]
            tmax = weather_res["daily"]["temperature_2m_max"][i]
            tmin = weather_res["daily"]["temperature_2m_min"][i]
            temps.append(f"{date}: {tmin}°C - {tmax}°C")

        return " | ".join(temps)

    except Exception as e:
        return f"❌ Weather API error: {str(e)}"

def get_flight_options(destination, amadeus_api_key=None):
    """Fetch mock flight options using Amadeus test API (requires free account).
       If no API key is provided, returns mock flight data instead."""
    try:
        if not amadeus_api_key:
            # Fallback mock flights if user doesn’t provide API key
            return [
                f"✈️ DEL → {destination.upper()} : ₹4,500 (Mock)",
                f"✈️ BOM → {destination.upper()} : ₹5,200 (Mock)"
            ]

        # Real API call if API key exists
        origin = "DEL"
        date = (datetime.date.today() + datetime.timedelta(days=7)).isoformat()

        flight_url = f"https://test.api.amadeus.com/v2/shopping/flight-offers"
        headers = {"Authorization": f"Bearer {amadeus_api_key}"}
        params = {
            "originLocationCode": origin,
            "destinationLocationCode": "GOI" if destination.lower() == "goa" else "KUU",
            "departureDate": date,
            "adults": 1,
            "max": 2
        }
        res = requests.get(flight_url, headers=headers, params=params).json()

        if "data" not in res:
            return ["No flights found."]

        flights = []
        for f in res["data"]:
            price = f["price"]["total"]
            itinerary = f["itineraries"][0]
            dep = itinerary["segments"][0]["departure"]["iataCode"]
            arr = itinerary["segments"][0]["arrival"]["iataCode"]
            flights.append(f"✈️ {dep} → {arr} : ₹{price}")

        return flights
    except Exception as e:
        return [f"❌ Flight API error: {str(e)}"]

# ----------------- Streamlit App -----------------
st.set_page_config(page_title="AI Trip Planner", page_icon="🌍")

st.title("🌍 AI Trip Planner (Groq + Tools Powered)")
st.write("Plan your trips in seconds using Groq LLM 🚀")

# Sidebar for API keys
st.sidebar.header("🔑 API Key Settings")
groq_api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")
amadeus_api_key = st.sidebar.text_input("Enter your Amadeus API Key (optional):", type="password")

# User inputs
destination = st.text_input("Enter Destination:", "Manali")
days = st.number_input("Number of days:", min_value=1, max_value=30, value=2)

budget = st.selectbox(
    "Select your budget:",
    ["Low", "Medium", "High"],
    index=1
)

interests = st.multiselect(
    "Select your interests:",
    ["Adventure", "Food", "History", "Shopping", "Nature", "Relaxation", "Culture", "Nightlife"],
    default=["Nature", "Adventure"]
)

itinerary_text = ""  # Global store

if st.button("Plan My Trip"):
    if not groq_api_key:
        st.error("❌ Please enter your Groq API key in the sidebar.")
    else:
        with st.spinner("✈️ Planning your personalized trip..."):
            try:
                # Fetch external tool data
                weather_info = get_weather(destination)
                flight_info = get_flight_options(destination, amadeus_api_key)

                # Convert interests to text
                interests_text = ", ".join(interests) if interests else "general sightseeing"

                # Prompt that forces LLM to DISPLAY tool data
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a helpful AI travel planner."),
                    ("human", f"""
Plan a {days}-day trip to {destination}.
Budget: {budget}.
Traveler interests: {interests_text}.

IMPORTANT:
1. First, clearly display the WEATHER DATA provided: {weather_info}.
2. Then, clearly display the FLIGHT OPTIONS provided: {', '.join(flight_info)}.
3. After that, suggest a day-by-day itinerary (activities, attractions, food, culture).
Make sure weather and flight info are visible in the final answer.
""")
                ])

                # Initialize Groq LLM
                llm = ChatGroq(
                    model="llama-3.1-8b-instant",
                    api_key=groq_api_key
                )

                # Call model
                response = llm.invoke(prompt.format_messages())
                itinerary_text = response.content

                # Show itinerary
                st.subheader("🗺️ Suggested Itinerary")
                st.write(itinerary_text)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# ----------------- PDF Export -----------------
if itinerary_text:
    def create_pdf(text):
        buffer = io.BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter

        pdf.setFont("Helvetica", 12)
        y = height - 50
        for line in text.split("\n"):
            pdf.drawString(50, y, line)
            y -= 20
            if y < 50:  # new page if content goes beyond
                pdf.showPage()
                pdf.setFont("Helvetica", 12)
                y = height - 50

        pdf.save()
        buffer.seek(0)
        return buffer

    pdf_buffer = create_pdf(itinerary_text)

    st.download_button(
        label="📥 Download Itinerary as PDF",
        data=pdf_buffer,
        file_name=f"{destination}_itinerary.pdf",
        mime="application/pdf"
    )
