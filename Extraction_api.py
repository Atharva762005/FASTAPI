from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, List

app = FastAPI()

# Define a model for the request body
class TextQuery(BaseModel):
    text: List[str]  # Expecting a list of strings (each string is a paragraph)

# Classify the text into different categories
def classify_text(List_text: List[str]) -> Dict[str, List[str]]:
    clauses = {
        "Data Related Clauses": [],
        "Intellectual Property Related Clauses": [],
        "Legal Clauses": [],
        "Monetary Clauses": [],
        "After Sale Clauses": [],
        "Customer Service Clauses": []
    }

    # Define patterns or keywords for classification (this can be expanded)
    data_keywords = ['data', 'privacy', 'personal information', 'share', 'sell']
    intellectual_property_keywords = ['intellectual property', 'copyright', 'patent', 'trademark']
    legal_keywords = ['law', 'governance', 'jurisdiction', 'dispute']
    monetary_keywords = ['payment', 'cost', 'fees', 'compensation']
    after_sale_keywords = ['warranty', 'returns', 'refund', 'exchange']
    customer_service_keywords = ['customer service', 'support', 'contact']

    # Iterate over the paragraphs (each paragraph is already a separate string in List_text)
    for paragraph in List_text:
        classified = False
        for category, keywords in {
            "Data Related Clauses": data_keywords,
            "Intellectual Property Related Clauses": intellectual_property_keywords,
            "Legal Clauses": legal_keywords,
            "Monetary Clauses": monetary_keywords,
            "After Sale Clauses": after_sale_keywords,
            "Customer Service Clauses": customer_service_keywords,
        }.items():
            if any(keyword.lower() in paragraph.lower() for keyword in keywords):
                clauses[category].append(paragraph)
                classified = True
                break

    return clauses

# Define an endpoint to receive text and return classified clauses
@app.post("/classify-text/")
async def classify_text_endpoint(query: TextQuery):
    # Call the classification function with the list of paragraphs
    classified_clauses = classify_text(query.text)
    return classified_clauses
