import pandas as pd
from dotenv import load_dotenv
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import time
import json 
load_dotenv()

INPUT_CSV_FILE = 'fdata_cleaned.csv'
OUTPUT_CSV_FILE = 'fdata_with_embeddings.csv'

try:
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in a .env file.")

    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini_api_key)
    print(f"Embedding model initialized: {embeddings_model.model}")
except Exception as e:
    print(f"Error initializing GoogleGenerativeAIEmbeddings: {e}")
    print("Please ensure your GEMINI_API_KEY is correctly set in the .env file.")
    embeddings_model = None

def get_embedding_batch(texts):
    """Generates embeddings for a list of texts."""
    if not embeddings_model:
        return [None] * len(texts) 
    try:
        return embeddings_model.embed_documents(texts)
    except Exception as e:
        print(f"Error generating embeddings for batch (first text: '{texts[0][:50]}...'): {e}")
        return [None] * len(texts)
def prepare_data_with_embeddings():
    if embeddings_model is None:
        print("Cannot proceed without a properly initialized embedding model.")
        return

    try:
        df = pd.read_csv(INPUT_CSV_FILE)
        print(f"Loaded {len(df)} rows from {INPUT_CSV_FILE}")
        required_cols = ['flightNumber', 'airline', 'scheduledDepartureTime', 'scheduledArrivalTime', 'origin', 'destination']
        for col in required_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna('')
            else:
                print(f"Warning: Column '{col}' not found in {INPUT_CSV_FILE}. This might affect embedding quality.")
                df[col] = ''
        df['text_content'] = df.apply(
            lambda row: (
                f"Flight number {row['flightNumber']} by {row['airline']} "
                f"from {row['origin']} to {row['destination']} "
                f"departing at {row['scheduledDepartureTime']} and arriving at {row['scheduledArrivalTime']}."
            ),
            axis=1
        )
        batch_size = 50
        all_embeddings = []
        print(f"Generating embeddings for {len(df)} entries (batch size: {batch_size})...")
        
        original_indices = df.index.tolist()
        processed_count = 0
        
        for i in range(0, len(df), batch_size):
            batch_texts = df['text_content'].iloc[i:i+batch_size].tolist()
            batch_embeddings = get_embedding_batch(batch_texts)
            
            all_embeddings.extend(batch_embeddings)
            
            processed_count += len(batch_texts)
            print(f"Processed embeddings for {processed_count}/{len(df)} rows.")
            time.sleep(0.5)

        df['embedding'] = all_embeddings
        df_final = df.dropna(subset=['embedding']).reset_index(drop=True)
        df_final['embedding'] = df_final['embedding'].apply(
            lambda x: json.dumps(x.tolist() if hasattr(x, 'tolist') else x)
        )

        df_final.to_csv(OUTPUT_CSV_FILE, index=False)
        
        print(f"Prepared {len(df_final)} rows with embeddings and saved to '{OUTPUT_CSV_FILE}'.")
        print(f"Dropped {len(df) - len(df_final)} rows due to embedding failure.")

    except FileNotFoundError:
        print(f"Error: The file '{INPUT_CSV_FILE}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    prepare_data_with_embeddings()