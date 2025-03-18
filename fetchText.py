from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import main  # Import your second Python file where classify_and_summarize is defined

app = FastAPI()

class TextData(BaseModel):
    text: List[str]

@app.post("/send-text/")
async def receive_text(data: TextData):
    # Extract the text from the request
    text = data.text #list of strings

    # Call the classify_and_summarize function from external_api.py
    summaries = main.classify_and_summarize(text)
    for category, category_summaries in summaries.items():
        print(f"Category: {category}")
        for summary in category_summaries:
            print(f" - {summary}")
        print()  # Add a newline for clarity
    # Return the summarized data
    if summaries:
        return {"summaries": summaries}
    else:
        return {"error": "No summaries found"}
