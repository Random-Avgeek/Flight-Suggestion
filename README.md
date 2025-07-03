Flight Suggester

This model uses an intelligent AI model (Gemini 1.5-flash) to first figure direct then layover flights from a city to another based on the data.
Further, it also suggests flight based on time and even outside of the preferences that are close enough to recognize
Lastly, it also suggests the most appropriate flight based on the user's preferences.

## Features

* **Dynamic Flight Search:** Find flights between specified origin and destination cities.
* **Time Slot Preferences:** Filter flights based on preferred departure and arrival time slots (e.g., "morning," "evening").
* **AI-Powered Suggestions:** Utilizes Google Gemini AI to provide intelligent flight recommendations, including direct routes and potential layover options.
* **Local RAG:** Integrates a local dataset of flight schedules and their embeddings for quick and relevant information retrieval.
* **Airline Information:** Provides brief descriptions of airlines involved in the suggested flights.
* **Interactive Web Interface:** A user-friendly web interface built with Flask and Tailwind CSS.

Note: The cleaned Data file and the embeddings file was not yet generated for the file.
Note 2: The data from www.kaggle.com/datasets/nikhilkhetan/indian-flight-schedules is of the year 2019 only, so this data is only for a trial purpose and may not have flights up to date.
