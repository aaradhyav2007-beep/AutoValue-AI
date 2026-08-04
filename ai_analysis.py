from google import genai
import os
from dotenv import load_dotenv
load_dotenv()
print(os.getenv("GEMINI_API_KEY"))
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def analyze_car(
    age,
    model,
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

    Model:
    {model}

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
    model="gemini-2.5-flash",
    contents=prompt
    )

    return response.text



