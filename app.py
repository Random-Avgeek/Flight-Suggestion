import pandas as pd
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import json
import os
import requests
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


load_dotenv()

app = Flask(__name__)
app_id = os.environ.get('__app_id', 'default-app-id')
firebase_config = json.loads(os.environ.get('__firebase_config', '{}'))
initial_auth_token = os.environ.get('__initial_auth_token', '')


LOCAL_DATA_FILE = 'fdata_with_embeddings.csv'
flight_data_df = pd.DataFrame()
try:
    flight_data_df = pd.read_csv(LOCAL_DATA_FILE)
    print(f"Local flight data loaded successfully from {LOCAL_DATA_FILE}.")

    flight_data_df['embedding'] = flight_data_df['embedding'].apply(
        lambda x: np.array(json.loads(x))
    )
    print("Embeddings parsed successfully.")

    flight_data_df['origin'] = flight_data_df['origin'].astype(str).fillna('')
    flight_data_df['destination'] = flight_data_df['destination'].astype(str).fillna('')
    flight_data_df['scheduledDepartureTime'] = flight_data_df['scheduledDepartureTime'].astype(str).fillna('')
    flight_data_df['scheduledArrivalTime'] = flight_data_df['scheduledArrivalTime'].astype(str).fillna('')
    flight_data_df['airline'] = flight_data_df['airline'].astype(str).fillna('')
    flight_data_df['flightNumber'] = flight_data_df['flightNumber'].astype(str).fillna('')
    flight_data_df['dayOfWeek'] = flight_data_df['dayOfWeek'].astype(str).fillna('')


except FileNotFoundError:
    print(f"Error: {LOCAL_DATA_FILE} not found. Please run 'prepare_local_data.py' first.")
except Exception as e:
    print(f"An error occurred while loading local flight data: {e}")
    flight_data_df = pd.DataFrame() 


TIME_SLOTS = {
    "early morning": {"start": 5, "end": 8},
    "morning": {"start": 8, "end": 12},
    "noon": {"start": 12, "end": 14},
    "afternoon": {"start": 14, "end": 18},
    "evening": {"start": 18, "end": 21},
    "night": {"start": 21, "end": 24},
    "midnight": {"start": 0, "end": 5}
}

AIRLINE_INFO = {
    "Air India": "Air India is India's flag carrier, known for its extensive network and full-service experience.",
    "TestIndiGo": "IndiGo is a leading low-cost carrier in India, popular for its punctuality and wide domestic network.",
    "SpiceJet": "SpiceJet is a budget airline known for its affordable fares and regional connectivity.",
    "Vistara": "Vistara is a full-service airline offering premium services and a comfortable flying experience.",
    "AirAsia India": "AirAsia India is a low-cost airline, part of the AirAsia Group, offering competitive fares.",
    "Go First": "Go First (formerly GoAir) is a low-cost airline focusing on value for money.",
    "Akasa Air": "Akasa Air is a relatively new Indian low-cost carrier, focusing on efficiency and customer service.",
    "Alliance Air": "Alliance Air is a regional airline, a subsidiary of Air India, connecting smaller cities."
    
}

def get_hour_from_time_string(time_str):
    try:
        return int(time_str.split(':')[0])
    except (ValueError, AttributeError):
        return -1
try:
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it.")
    query_embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini_api_key)
    print("Query embedding model initialized.")
except Exception as e:
    print(f"Error initializing query embedding model: {e}")
    query_embeddings_model = None

@app.route('/')
def index():
    """Serves the HTML frontend."""
    return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Flight Finder AI</title>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <style>
        body {
            font-family: 'Inter', sans-serif;
            background-color: #f0f4f8;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .container {
            background-color: #ffffff;
            border-radius: 15px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
            padding: 30px;
            width: 100%;
            max-width: 800px;
        }
        select, input[type="text"] {
            border-radius: 8px;
            border: 1px solid #cbd5e0;
            padding: 10px;
            font-size: 1rem;
            width: 100%;
            transition: border-color 0.2s;
        }
        select:focus, input[type="text"]:focus {
            outline: none;
            border-color: #6366f1;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
        }
        button {
            border-radius: 8px;
            padding: 12px 20px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.3s ease, transform 0.1s ease;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        button:hover {
            transform: translateY(-2px);
        }
        .gradient-button {
            background-image: linear-gradient(to right, #6366f1, #8b5cf6);
            color: white;
            border: none;
        }
        .gradient-button:hover {
            background-image: linear-gradient(to right, #4f46e5, #7c3aed);
        }
        .result-box {
            background-color: #f9fafb;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 20px;
            min-height: 150px;
            white-space: pre-wrap; /* Preserves whitespace and line breaks */
            word-wrap: break-word; /* Breaks long words */
        }
        .loading-spinner {
            border: 4px solid rgba(0, 0, 0, 0.1);
            border-left-color: #6366f1;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            display: none; /* Hidden by default */
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body class="flex items-center justify-center min-h-screen bg-gray-100">
    <div class="container p-8 space-y-6">
        <h1 class="text-3xl font-bold text-center text-gray-800 mb-6">Flight Suggester</h1>

        <form id="flightForm" class="space-y-5">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-5">
                <div>
                    <label for="origin" class="block text-sm font-medium text-gray-700 mb-1">Origin City:</label>
                    <select id="origin" name="origin" class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"></select>
                </div>
                <div>
                    <label for="destination" class="block text-sm font-medium text-gray-700 mb-1">Destination City:</label>
                    <select id="destination" name="destination" class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"></select>
                </div>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-2 gap-5">
                <div>
                    <label for="departureTime" class="block text-sm font-medium text-gray-700 mb-1">Preferred Departure Time:</label>
                    <select id="departureTime" name="departureTime" class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"></select>
                </div>
                <div>
                    <label for="arrivalTime" class="block text-sm font-medium text-gray-700 mb-1">Preferred Arrival Time:</label>
                    <select id="arrivalTime" name="arrivalTime" class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"></select>
                </div>
            </div>

            <button type="submit" class="gradient-button w-full flex items-center justify-center">
                Find Flights
                <div id="loadingSpinner" class="loading-spinner ml-3"></div>
            </button>
        </form>

        <div id="results" class="result-box mt-8 text-gray-800">
            <p class="text-center text-gray-500">Enter your flight preferences and click "Find Flights" to get AI-powered suggestions.</p>
        </div>
    </div>

    <script>
        const cities = [
            "Delhi", "Lucknow", "Kochi", "Ahmedabad", "Jaipur", "Bengaluru", "Guwahati", "Goa", "Kolkata",
            "Hyderabad", "Nagpur", "Bagdogra", "Mumbai", "Leh", "Patna", "Ranchi", "Pune", "Jammu",
            "Srinagar", "Chennai", "Bhubaneswar", "Port Blair", "Chandigarh", "Visakhapatnam",
            "Vijayawada", "Tirupati", "Varanasi", "Aurangabad", "Rajkot", "Amritsar", "Imphal",
            "Jodhpur", "Indore", "Vadodara", "Raipur", "Udaipur", "Surat", "Bhopal", "Gaya",
            "Khajuraho", "Thiruvananthapuram", "Calicut", "Hubli", "Coimbatore", "Mangalore",
            "Aizwal", "Dibrugarh", "Madurai", "Agra", "Dimapur", "Silchar", "Agartala", "Dehradun",
            "Allahabad", "Jorhat", "Tiruchirappalli", "Kolhapur", "Rajahmundry", "Jabalpur",
            "Tuticorin", "Keshod", "Porbandar", "Salem", "Mysore", "Pondicherry", "Kanpur",
            "Kandla", "Lilabari", "Kullu", "Ludhiana", "Shimla", "Gwalior", "Pantnagar",
            "Bhatinda", "Bhavnagar", "Belgaum", "Tezpur", "Shillong", "Tezu", "Kannur",
            "Kadapa", "Darbhanga"
        ];

        const timeSlots = [
            "early morning", "morning", "noon", "afternoon", "evening", "night", "midnight"
        ];

        function populateDropdown(selectElementId, options) {
            const select = document.getElementById(selectElementId);
            options.forEach(option => {
                const opt = document.createElement('option');
                opt.value = option;
                opt.textContent = option.charAt(0).toUpperCase() + option.slice(1); // Capitalize first letter
                select.appendChild(opt);
            });
        }

        document.addEventListener('DOMContentLoaded', () => {
            populateDropdown('origin', cities);
            populateDropdown('destination', cities);
            populateDropdown('departureTime', timeSlots);
            populateDropdown('arrivalTime', timeSlots);
        });

        document.getElementById('flightForm').addEventListener('submit', async (event) => {
            event.preventDefault(); // Prevent default form submission
            const resultsDiv = document.getElementById('results');
            const loadingSpinner = document.getElementById('loadingSpinner');

            resultsDiv.innerHTML = '<p class="text-center text-gray-500">Searching for flights...</p>';
            loadingSpinner.style.display = 'block'; // Show spinner

            const formData = new FormData(event.target);
            const data = Object.fromEntries(formData.entries());

            try {
                const response = await fetch('/find_flights', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(data)
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const result = await response.json();
                if (result.error) {
                    resultsDiv.innerHTML = `<p class="text-red-600">Error: ${result.error}</p>`;
                } else {
                    resultsDiv.innerHTML = `<p>${result.suggestion}</p>`;
                }

            } catch (error) {
                console.error('Error:', error);
                resultsDiv.innerHTML = `<p class="text-red-600">Failed to fetch flight suggestions. Please try again. (${error.message})</p>`;
            } finally {
                loadingSpinner.style.display = 'none'; // Hide spinner
            }
        });
    </script>
</body>
</html>
    """)

@app.route('/find_flights', methods=['POST'])
def find_flights():
    """
    Receives flight preferences, filters data, performs local RAG, and calls Gemini API.
    """
    if flight_data_df.empty:
        return jsonify({"error": f"Flight data not loaded from {LOCAL_DATA_FILE}. Please run 'prepare_local_data.py' first."}), 500
    
    if query_embeddings_model is None:
        return jsonify({"error": "Gemini embedding model for queries not initialized. Check GEMINI_API_KEY."}), 500

    user_preferences = request.get_json()
    origin = user_preferences.get('origin')
    destination = user_preferences.get('destination')
    departure_time_slot = user_preferences.get('departureTime')
    arrival_time_slot = user_preferences.get('arrivalTime')

    if not all([origin, destination, departure_time_slot, arrival_time_slot]):
        return jsonify({"error": "Missing one or more required flight preferences."}), 400

    print(f"Received request: {user_preferences}")

    relevant_flights_context = ""
    
    direct_flights = flight_data_df[
        (flight_data_df['origin'].str.lower() == origin.lower()) &
        (flight_data_df['destination'].str.lower() == destination.lower())
    ].copy()

    direct_flights_filtered_by_time = pd.DataFrame()

    if not direct_flights.empty:
        dep_slot = TIME_SLOTS.get(departure_time_slot.lower())
        arr_slot = TIME_SLOTS.get(arrival_time_slot.lower())

        temp_direct_flights = direct_flights.copy()

        if dep_slot:
            temp_direct_flights['dep_hour'] = temp_direct_flights['scheduledDepartureTime'].apply(get_hour_from_time_string)
            if dep_slot['start'] <= dep_slot['end']:
                temp_direct_flights = temp_direct_flights[
                    (temp_direct_flights['dep_hour'] >= dep_slot['start']) &
                    (temp_direct_flights['dep_hour'] < dep_slot['end'])
                ]
            else:
                temp_direct_flights = temp_direct_flights[
                    (temp_direct_flights['dep_hour'] >= dep_slot['start']) |
                    (temp_direct_flights['dep_hour'] < dep_slot['end'])
                ]
            temp_direct_flights = temp_direct_flights.drop(columns=['dep_hour'])

        if arr_slot:
            temp_direct_flights['arr_hour'] = temp_direct_flights['scheduledArrivalTime'].apply(get_hour_from_time_string)
            if arr_slot['start'] <= arr_slot['end']:
                temp_direct_flights = temp_direct_flights[
                    (temp_direct_flights['arr_hour'] >= arr_slot['start']) &
                    (temp_direct_flights['arr_hour'] < arr_slot['end'])
                ]
            else:
                temp_direct_flights = temp_direct_flights[
                    (temp_direct_flights['arr_hour'] >= arr_slot['start']) |
                    (temp_direct_flights['arr_hour'] < arr_slot['end'])
                ]
            temp_direct_flights = temp_direct_flights.drop(columns=['arr_hour'])
        
        direct_flights_filtered_by_time = temp_direct_flights.copy()

    if not direct_flights_filtered_by_time.empty and not direct_flights_filtered_by_time['embedding'].isnull().all():
        print("Direct flights found matching time criteria. Performing similarity search.")
        user_query_text = (
            f"Find direct flights from {origin} to {destination} "
            f"departing {departure_time_slot} and arriving {arrival_time_slot}."
        )
        try:
            query_embedding = query_embeddings_model.embed_query(user_query_text)
            query_embedding_np = np.array(query_embedding).reshape(1, -1)

            valid_embeddings = [emb for emb in direct_flights_filtered_by_time['embedding'] if isinstance(emb, np.ndarray)]
            if valid_embeddings:
                flight_embeddings_np = np.vstack(valid_embeddings)
                similarities = cosine_similarity(query_embedding_np, flight_embeddings_np)[0]

                top_n = 5
                temp_df = direct_flights_filtered_by_time.copy()
                temp_df['similarity'] = similarities
                most_relevant_direct_flights = temp_df.sort_values(by='similarity', ascending=False).head(top_n).drop(columns=['similarity']).copy()
                
                relevant_flights_context += "--- Direct Flights (Matching Time Criteria) ---\n"
                relevant_flights_context += most_relevant_direct_flights.to_string(index=False, columns=['origin', 'destination', 'scheduledDepartureTime', 'scheduledArrivalTime', 'airline', 'flightNumber', 'dayOfWeek']) + "\n\n"
                print("Direct flights context prepared.")
            else:
                print("No valid embeddings for direct flights after time filtering.")
        except Exception as e:
            print(f"Error during direct flight similarity search: {e}")
            relevant_flights_context += (
                f"--- Direct Flights (Error during detailed search) ---\n"
                f"Some direct flights from {origin} to {destination} were found, but an issue occurred during detailed matching:\n"
                f"{direct_flights_filtered_by_time.to_string(index=False, columns=['origin', 'destination', 'scheduledDepartureTime', 'scheduledArrivalTime', 'airline', 'flightNumber', 'dayOfWeek'])}"
            )
    elif not direct_flights.empty:
        print("Direct flights found, but none matching time criteria. Adding general direct flights to context.")
        relevant_flights_context += "--- Direct Flights (General, no exact time match) ---\n"
        relevant_flights_context += direct_flights.to_string(index=False, columns=['origin', 'destination', 'scheduledDepartureTime', 'scheduledArrivalTime', 'airline', 'flightNumber', 'dayOfWeek']) + "\n\n"
    else:
        print("No direct flights found. Searching for layover options.")
    layover_flights_context = ""
    if direct_flights_filtered_by_time.empty:
        
        possible_first_legs = flight_data_df[
            flight_data_df['origin'].str.lower() == origin.lower()
        ].copy()

        possible_second_legs = flight_data_df[
            flight_data_df['destination'].str.lower() == destination.lower()
        ].copy()

        layover_cities = set(possible_first_legs['destination'].str.lower()).intersection(
                             set(possible_second_legs['origin'].str.lower()))
        
        layover_cities = [city for city in layover_cities if city != origin.lower() and city != destination.lower()]

        found_layover_paths = []

        max_layover_paths = 5

        for layover_city in layover_cities:
            first_leg_options = possible_first_legs[
                possible_first_legs['destination'].str.lower() == layover_city
            ].copy()
            
            second_leg_options = possible_second_legs[
                possible_second_legs['origin'].str.lower() == layover_city
            ].copy()

            if not first_leg_options.empty and not second_leg_options.empty:
                first_leg_sample = first_leg_options.head(1)
                second_leg_sample = second_leg_options.head(1)

                if not first_leg_sample.empty and not second_leg_sample.empty:
                    path_description = (
                        f"Layover via {layover_city.capitalize()}:\n"
                        f"  Leg 1: Flight {first_leg_sample['flightNumber'].iloc[0]} by {first_leg_sample['airline'].iloc[0]} "
                        f"from {first_leg_sample['origin'].iloc[0]} to {first_leg_sample['destination'].iloc[0]} "
                        f"departs {first_leg_sample['scheduledDepartureTime'].iloc[0]} arrives {first_leg_sample['scheduledArrivalTime'].iloc[0]} on {first_leg_sample['dayOfWeek'].iloc[0]}.\n"
                        f"  Leg 2: Flight {second_leg_sample['flightNumber'].iloc[0]} by {second_leg_sample['airline'].iloc[0]} "
                        f"from {second_leg_sample['origin'].iloc[0]} to {second_leg_sample['destination'].iloc[0]} "
                        f"departs {second_leg_sample['scheduledDepartureTime'].iloc[0]} arrives {second_leg_sample['scheduledArrivalTime'].iloc[0]} on {second_leg_sample['dayOfWeek'].iloc[0]}."
                    )
                    found_layover_paths.append(path_description)
                    if len(found_layover_paths) >= max_layover_paths:
                        break

        if found_layover_paths:
            layover_flights_context = "--- Layover Flight Options ---\n" + "\n".join(found_layover_paths) + "\n\n"
            print("Layover flights context prepared.")
        else:
            print("No suitable layover paths found.")

    relevant_flights_context += layover_flights_context

    prompt = f"""
    You are an expert AI Flight Booking Assistant. Your task is to provide the user with the most appropriate flight suggestions based on their preferences and the available flight data.

    User's Flight Request:
    - Origin: {origin}
    - Destination: {destination}
    - Preferred Departure Time Slot: {departure_time_slot}
    - Preferred Arrival Time Slot: {arrival_time_slot}

    Available Flight Data (Retrieved from our knowledge base - prioritize direct, then layover):
    {relevant_flights_context if relevant_flights_context else "No specific flight data found for this route."}

    Airline Information:
    {json.dumps(AIRLINE_INFO, indent=2)}

    Instructions for your response:
    1.  **Prioritize Direct Flights:** First, analyze if any direct flights are available that match the user's origin, destination, and preferred time slots.
    2.  **Suggest Direct Flights:** If direct flights are found, list them clearly. For each direct flight, include:
        -   Flight Number
        -   Airline Name
        -   Origin, Destination
        -   Scheduled Departure Time, Scheduled Arrival Time
        -   Day of Week
        -   A brief, relevant description of the airline from the 'Airline Information' section, mixed with some small facts that the user might find interesting (If you encounter TestIndiGo, you can assume it as Indigo).
    3.  **Suggest Layover Flights (if no direct matches or as alternatives):**
        -   If no direct flights are found that perfectly match the time criteria, OR if direct flights are limited, check the 'Layover Flight Options' in the provided data.
        -   If layover options exist, suggest them. For each layover path, describe both legs of the journey (Flight Number, Airline, Origin-Layover, Layover-Destination, Departure/Arrival Times, Day of Week for each leg).
        -   Include a brief, relevant description of the airlines involved in the layover from the 'Airline Information' section.
    4.  **Handle No Flights Found:** If neither direct nor suitable layover flights are found for the requested origin and destination, clearly state that no flights could be found for the specified criteria and suggest being flexible with dates/times or trying different routes.
    5.  **Polite and Actionable Conclusion:** End your response with a polite and actionable tip for the user (e.g., "Always check the latest flight status," "Consider booking in advance," "Flexibility with travel dates/times can offer more options").
    6.  **Formatting:** Use clear headings, bullet points, and bold text for readability. Present flight details in an easy-to-digest format.

    Based on the above, please provide your flight suggestion:
    
    Side note, please keep in mind that if the user has entered the same origin andd the destination, make a small and non-offensive joke about it, like "In mood for a tour are we, or perhaps a U-Turn flight.... dont say you have something else in your mind" somewhere along these lines.
    """

    api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    headers = {'Content-Type': 'application/json'}
    params = {'key': os.getenv("GEMINI_API_KEY")}

    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": prompt}]}
        ]
    }

    try:
        response = requests.post(api_url, headers=headers, params=params, data=json.dumps(payload))
        response.raise_for_status()
        gemini_result = response.json()

        if gemini_result.get('candidates') and gemini_result['candidates'][0].get('content') and gemini_result['candidates'][0]['content'].get('parts'):
            ai_suggestion = gemini_result['candidates'][0]['content']['parts'][0]['text']
            return jsonify({"suggestion": ai_suggestion})
        else:
            print(f"Unexpected Gemini API response structure: {gemini_result}")
            return jsonify({"error": "AI model did not return a valid suggestion. Please try again."}), 500

    except requests.exceptions.RequestException as e:
        print(f"Gemini API request failed: {e}")
        error_detail = ""
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
            except json.JSONDecodeError:
                error_detail = e.response.text
        print(f"Gemini API error response: {error_detail}")
        return jsonify({"error": f"Failed to connect to AI service: {e}. Detail: {error_detail}"}), 500
    except Exception as e:
        print(f"An unexpected error occurred during AI processing: {e}")
        return jsonify({"error": f"An internal error occurred: {e}"}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080)