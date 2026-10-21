import json
import uuid
import os

import boto3
import numpy as np
import tensorflow as tf

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI()


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# UPLOAD DIRECTORY
# ============================================================

UPLOAD_DIR = "uploadimages"

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

app.mount(
    "/uploadimages",
    StaticFiles(directory=UPLOAD_DIR),
    name="uploadimages"
)


# ============================================================
# GROQ CLIENT
# ============================================================

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


# ============================================================
# CLOUDflare R2 MODEL CONFIGURATION
# ============================================================

MODEL_DIR = "models"

MODEL_FILENAME = "plant_disease_recog_model_pwp.keras"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    MODEL_FILENAME
)

# Make sure the models directory exist
os.makedirs(MODEL_DIR, exist_ok=True)


def download_model_from_r2():
    """
    Download the plant disease recognition model
    from Cloudflare R2 if it does not already exist
    locally.
    """

    # --------------------------------------------------------
    # If the model already exists locally, don't download it
    # --------------------------------------------------------

    if os.path.exists(MODEL_PATH):
        print("==========================================")
        print("Model already exists locally.")
        print("==========================================")

        return


    print("==========================================")
    print("Downloading plant disease model from R2...")
    print("==========================================")


    # --------------------------------------------------------
    # Get R2 environment variables
    # --------------------------------------------------------

    r2_endpoint = os.getenv("R2_ENDPOINT")
    r2_access_key = os.getenv("R2_ACCESS_KEY_ID")
    r2_secret_key = os.getenv("R2_SECRET_ACCESS_KEY")
    r2_bucket = os.getenv("R2_BUCKET_NAME")
    r2_model_key = os.getenv("R2_MODEL_KEY")


    # --------------------------------------------------------
    # Validate R2 configuration
    # --------------------------------------------------------

    missing_variables = []

    if not r2_endpoint:
        missing_variables.append("R2_ENDPOINT")

    if not r2_access_key:
        missing_variables.append("R2_ACCESS_KEY_ID")

    if not r2_secret_key:
        missing_variables.append("R2_SECRET_ACCESS_KEY")

    if not r2_bucket:
        missing_variables.append("R2_BUCKET_NAME")

    if not r2_model_key:
        missing_variables.append("R2_MODEL_KEY")


    if missing_variables:

        raise RuntimeError(
            "Missing R2 environment variables: "
            + ", ".join(missing_variables)
        )


    # --------------------------------------------------------
    # Create Cloudflare R2 client
    #
    # R2 uses an S3-compatible API.
    # --------------------------------------------------------

    r2 = boto3.client(
        "s3",

        endpoint_url=r2_endpoint,

        aws_access_key_id=r2_access_key,

        aws_secret_access_key=r2_secret_key,

        region_name="auto"
    )


    # --------------------------------------------------------
    # Download the model
    # --------------------------------------------------------

    try:

        r2.download_file(
            r2_bucket,
            r2_model_key,
            MODEL_PATH
        )

        print("==========================================")
        print("Model downloaded successfully!")
        print(f"Model saved to: {MODEL_PATH}")
        print("==========================================")

    except Exception as e:

        print("==========================================")
        print("ERROR DOWNLOADING MODEL FROM R2")
        print(e)
        print("==========================================")

        raise


# ============================================================
# LOAD MODEL FROM R2
# ============================================================

try:

    download_model_from_r2()

    print("==========================================")
    print("Loading TensorFlow model...")
    print("==========================================")


    model = tf.keras.models.load_model(
        MODEL_PATH
    )


    print("==========================================")
    print("Plant disease model loaded successfully!")
    print("==========================================")


except Exception as e:

    print("==========================================")
    print("ERROR LOADING PLANT DISEASE MODEL")
    print(e)
    print("==========================================")

    model = None


# ============================================================
# LOAD PLANT DISEASE LABELS
# ============================================================

try:

    with open(
        "plant_disease.json",
        "r"
    ) as file:

        plant_disease = json.load(file)


except Exception as e:

    print(
        f"Error loading plant_disease.json: {e}"
    )

    plant_disease = []


NUM_CLASSES = len(plant_disease)


# ============================================================
# REQUEST MODELS
# ============================================================

class AnalyzeRequest(BaseModel):

    prediction: str


class ImageUploadRequest(BaseModel):

    img: str


# ============================================================
# IMAGE FEATURE EXTRACTION
# ============================================================

def extract_features(image_path):

    image = tf.keras.utils.load_img(
        image_path,
        target_size=(160, 160)
    )

    feature = tf.keras.utils.img_to_array(
        image
    )

    feature = np.array(
        [feature]
    )

    return feature


# ============================================================
# MODEL PREDICTION
# ============================================================

def model_predict(image_path):

    if model is None:

        return None, "Model not loaded"


    img = extract_features(
        image_path
    )


    prediction = model.predict(
        img,
        verbose=0
    )


    idx = int(
        prediction.argmax()
    )


    if idx >= NUM_CLASSES or NUM_CLASSES == 0:

        return (
            None,
            f"Prediction index {idx} out of bounds "
            f"(labels loaded: {NUM_CLASSES})"
        )


    disease_info = plant_disease[idx]

    confidence = float(
        np.max(prediction)
    )


    return disease_info, confidence


# ============================================================
# AI SYSTEM PROMPT
# ============================================================

def generate_system_prompt(
    disease_name: str
) -> str:

    return f"""
You are the official Farmee AI Plant Expert assistant.
You provide detailed agricultural analysis for plant diseases.

Disease Detected: {disease_name}

YOUR NUMBER ONE RULE:
- ALWAYS PROVIDE PRACTICAL, ACTIONABLE ADVICE FOR FARMERS.
- RESPOND DIRECTLY TO THE DETECTED DISEASE.

Your Role:
- ONLY respond to plant health and agricultural questions.
- Represent Farmee in a professional, expert, and trustworthy way.
- Help users understand symptoms, severity, and cures.

Response Structure (WhatsApp-friendly, 4-8 lines):
- Analysis: A brief professional explanation of the disease.
- Symptoms: List 2-3 key symptoms associated with {disease_name}.
- Cure: Provide the most effective treatment or organic remedy.
- Prevention: 1 key step to prevent recurrence.
- Expert Note: Mention that a professional agricultural expert can be reached via our portal for a field visit.

Language & Tone:
- Use an expert and knowledgeable tone.
- Use light emojis where appropriate (🌱, 🏥, 🚜).
- Keep it concise but thorough.
"""


# ============================================================
# EXPERT MAPPING
# ============================================================

def get_experts_for_disease(
    disease_name: str
) -> list:

    disease_expert_mapping = {

        "corn___leaf_blight": [

            {
                "id": "exp_001",
                "name": "Dr. Ahmed Hassan",
                "specialization": "Corn & Cereal Diseases",
                "contact": "+234-701-234-5678",
                "email": "ahmed.hassan@farmee.ng",
                "experience_years": 15,
                "response_time": "2-4 hours",
                "availability": "Available",
                "verified": True
            }

        ],


        "tomato___early_blight": [

            {
                "id": "exp_002",
                "name": "Eng. Chioma Okonkwo",
                "specialization": "Vegetable Disease Management",
                "contact": "+234-803-456-7890",
                "email": "chioma.okonkwo@farmee.ng",
                "experience_years": 12,
                "response_time": "1-3 hours",
                "availability": "Available",
                "verified": True
            }

        ],


        "pepper___bacterial_spot": [

            {
                "id": "exp_003",
                "name": "Prof. Ibrahim Musa",
                "specialization": "Bacterial Disease Control",
                "contact": "+234-905-678-9012",
                "email": "ibrahim.musa@farmee.ng",
                "experience_years": 20,
                "response_time": "Same day",
                "availability": "Available",
                "verified": True
            }

        ],


        "default": [

            {
                "id": "exp_001",
                "name": "Dr. Ahmed Hassan",
                "specialization": "General Plant Diseases",
                "contact": "+234-701-234-5678",
                "email": "ahmed.hassan@farmee.ng",
                "experience_years": 15,
                "response_time": "2-4 hours",
                "availability": "Available",
                "verified": True
            },


            {
                "id": "exp_002",
                "name": "Eng. Chioma Okonkwo",
                "specialization": "Crop Disease Management",
                "contact": "+234-803-456-7890",
                "email": "chioma.okonkwo@farmee.ng",
                "experience_years": 12,
                "response_time": "1-3 hours",
                "availability": "Available",
                "verified": True
            },


            {
                "id": "exp_003",
                "name": "Prof. Ibrahim Musa",
                "specialization": "Pest & Disease Control",
                "contact": "+234-905-678-9012",
                "email": "ibrahim.musa@farmee.ng",
                "experience_years": 20,
                "response_time": "Same day",
                "availability": "Available",
                "verified": True
            }

        ]

    }


    disease_key = (
        disease_name
        .lower()
        .replace(" ", "_")
    )


    return disease_expert_mapping.get(
        disease_key,
        disease_expert_mapping["default"]
    )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
async def root():

    return {
        "message": "Farmee API is running"
    }


# ============================================================
# IMAGE UPLOAD + DISEASE PREDICTION
# ============================================================

@app.post("/upload")
async def upload_image(
    file: UploadFile = File(None),
    img: UploadFile = File(None)
):

    file_path = None
    temp_name = None


    try:

        # ----------------------------------------------------
        # Accept either "file" or "img"
        # ----------------------------------------------------

        image_file = file or img


        if (
            not image_file
            or not image_file.filename
        ):

            raise HTTPException(
                status_code=400,
                detail="No image file provided"
            )


        # ----------------------------------------------------
        # Create unique filename
        # ----------------------------------------------------

        file_extension = (
            os.path.splitext(
                image_file.filename
            )[1]
            or ".jpg"
        )


        temp_name = (
            f"temp_"
            f"{uuid.uuid4().hex}"
            f"{file_extension}"
        )


        file_path = os.path.join(
            UPLOAD_DIR,
            temp_name
        )


        # ----------------------------------------------------
        # Save uploaded image
        # ----------------------------------------------------

        with open(
            file_path,
            "wb"
        ) as buffer:

            buffer.write(
                await image_file.read()
            )


        # ----------------------------------------------------
        # Predict disease
        # ----------------------------------------------------

        disease_info, confidence = model_predict(
            file_path
        )


        if disease_info is None:

            raise HTTPException(
                status_code=500,
                detail=str(confidence)
            )


        # ----------------------------------------------------
        # Return result
        # ----------------------------------------------------

        return {

            "result": True,

            "imagepath":
                f"/uploadimages/{temp_name}",

            "prediction":
                disease_info.get(
                    "name",
                    "Unknown"
                ),

            "disease_info":
                disease_info,

            "confidence":
                confidence

        }


    except HTTPException:

        if (
            file_path
            and os.path.exists(file_path)
        ):

            try:

                os.remove(
                    file_path
                )

            except Exception:

                pass


        raise


    except Exception as e:

        print(
            f"Upload Error: {e}"
        )


        if (
            file_path
            and os.path.exists(file_path)
        ):

            try:

                os.remove(
                    file_path
                )

            except Exception:

                pass


        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to process image: "
                f"{str(e)}"
            )
        )


# ============================================================
# DISEASE ANALYSIS
# ============================================================

@app.post("/analyze")
async def analyze_disease(
    req: AnalyzeRequest
):

    try:

        if (
            not req.prediction
            or not req.prediction.strip()
        ):

            raise HTTPException(
                status_code=400,
                detail="Prediction is required"
            )


        # ----------------------------------------------------
        # Groq AI request
        # ----------------------------------------------------

        chat_completion = (
            client.chat.completions.create(

                messages=[

                    {
                        "role": "system",

                        "content":
                            generate_system_prompt(
                                req.prediction
                            )
                    },


                    {
                        "role": "user",

                        "content":
                            (
                                "Please provide a deep "
                                "analysis, symptoms, "
                                f"and cure for "
                                f"{req.prediction}."
                            )
                    }

                ],

                model="openai/gpt-oss-120b"
            )
        )


        ai_response = (
            chat_completion
            .choices[0]
            .message
            .content
        )


        # ----------------------------------------------------
        # Disease details
        # ----------------------------------------------------

        disease_details = next(
            (
                d
                for d in plant_disease
                if d.get("name")
                == req.prediction
            ),
            None
        )


        # ----------------------------------------------------
        # Experts
        # ----------------------------------------------------

        experts = get_experts_for_disease(
            req.prediction
        )


        return {

            "status": "success",

            "prediction":
                req.prediction,

            "deep_analysis":
                ai_response,

            "disease_details":
                disease_details,

            "available_experts":
                experts,

            "expert_contact_note":
                (
                    f"Found {len(experts)} "
                    f"expert(s) who can assist "
                    f"with {req.prediction}. "
                    "Contact them directly or "
                    "request a field visit "
                    "through Farmee."
                )

        }


    except HTTPException:

        raise


    except Exception as e:

        print(
            f"AI Analysis Error: {e}"
        )


        return JSONResponse(

            status_code=500,

            content={

                "status": "error",

                "message":
                    "Could not complete AI "
                    "analysis at this time.",

                "fallback_info": {

                    "prediction":
                        req.prediction
                        if req
                        else "",

                    "note":
                        "Please consult a local "
                        "agricultural officer."

                }

            }

        )


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )