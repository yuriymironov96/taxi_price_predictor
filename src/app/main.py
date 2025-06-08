from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd

from src.ml.model import get_model

app = FastAPI()
templates = Jinja2Templates(directory="src/app/templates")

# Hardcoded fare components
BASE_FARE = 3.00
PER_KM_RATE = 1.25
PER_MINUTE_RATE = 0.50


@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/", response_class=HTMLResponse)
async def predict_fare(
    request: Request,
    distance: float = Form(...),
    time_of_day: str = Form(...),
    day_of_week: str = Form(...),
    passenger_count: int = Form(...),
    traffic: str = Form(...),
    weather: str = Form(...),
    duration: float = Form(...)
):
    model = get_model()
    # Simple pricing logic
    # price = BASE_FARE + PER_KM_RATE * distance + PER_MINUTE_RATE * duration
    # input_data = [[
    #     distance,
    #     time_of_day,
    #     day_of_week,
    #     passenger_count,
    #     traffic,
    #     weather,
    #     BASE_FARE,
    #     PER_KM_RATE,
    #     PER_MINUTE_RATE,
    #     duration
    # ]]
    # # import pdb; pdb.set_trace()
    # price = model.predict(input_data)[0]
    # return templates.TemplateResponse("index.html", {
    #     "request": request,
    #     "predicted_price": round(price, 2)
    # })

    # Initialize all one-hot flags to False
    input_dict = {
        "Trip_Distance_km": distance,
        "Passenger_Count": passenger_count,
        "Base_Fare": BASE_FARE,
        "Per_Km_Rate": PER_KM_RATE,
        "Per_Minute_Rate": PER_MINUTE_RATE,
        "Trip_Duration_Minutes": duration,
        "Time_of_Day_Evening": False,
        "Time_of_Day_Morning": False,
        "Time_of_Day_Night": False,
        "Day_of_Week_Weekend": False,
        "Traffic_Conditions_Low": False,
        "Traffic_Conditions_Medium": False,
        "Weather_Rain": False,
        "Weather_Snow": False,
    }

    # Set one-hot encoded features based on input
    if time_of_day == "Evening":
        input_dict["Time_of_Day_Evening"] = True
    elif time_of_day == "Morning":
        input_dict["Time_of_Day_Morning"] = True
    elif time_of_day == "Night":
        input_dict["Time_of_Day_Night"] = True
    # Afternoon is implicit — all flags False

    if day_of_week == "Weekend":
        input_dict["Day_of_Week_Weekend"] = True

    if traffic == "Low":
        input_dict["Traffic_Conditions_Low"] = True
    elif traffic == "Medium":
        input_dict["Traffic_Conditions_Medium"] = True
    # High is implicit

    if weather == "Rain":
        input_dict["Weather_Rain"] = True
    elif weather == "Snow":
        input_dict["Weather_Snow"] = True
    # Clear or Fog is implicit

    # Create DataFrame and predict
    input_df = pd.DataFrame([input_dict])
    predicted_price = model.predict(input_df)[0]

    return templates.TemplateResponse("index.html", {
        "request": request,
        "predicted_price": round(predicted_price, 2)
    })
