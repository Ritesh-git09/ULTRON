

import time

from ollama import Client



client = Client(host="http://localhost:11434")


from config import MODEL




print("Connecting to ULTRON's neural core...")



start = time.perf_counter()

first_token = None

full_response = ""



try:

    stream = client.chat(

        model=MODEL,

        messages=[

            {

                "role": "user",

                "content": "Do you know who Ultron is? Describe your personality in three sentences."

            }

        ],

        stream=True

    )



    print("\nULTRON: ", end="", flush=True)



    for chunk in stream:

        token = chunk.message.content or ""



        if token:

            if first_token is None:

                first_token = time.perf_counter()



            print(token, end="", flush=True)

            full_response += token



    end = time.perf_counter()



    print("\n\n--- PERFORMANCE ---")

    print(f"First token latency: {first_token - start:.2f}s")

    print(f"Total generation time: {end - start:.2f}s")

    print(f"Response length: {len(full_response)} characters")



except Exception as e:

    print(f"\nULTRON CONNECTION ERROR: {e}")

