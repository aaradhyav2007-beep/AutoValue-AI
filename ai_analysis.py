from google import genai
import os
from dotenv import load_dotenv
load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def analyze_car(
    age,
    mileage,
    original_price,
    predicted_price,
    seller_price,
    depreciation):

    prompt = f"""
    You are an expert used-car valuation assistant.

    Analyze this vehicle.

    Age:
    {age} years

    Mileage:
    {mileage} km

    Original Price:
    ${original_price:,.2f}

    Estimated Market Value:
    ${predicted_price:,.2f}

    Seller Asking Price:
    ${seller_price:,.2f}

    Depreciation:
    {depreciation:.2f}%

    Return your response in this format:

    Recommendation:
    Decision:
    Pros:
    Cons:
    Negotiation Tip:

    Be concise and practical.
    """
    response = client.models.generate_content(
    model="gemini-3.1-flash-lite",
    contents=prompt
    )

    return response.text



