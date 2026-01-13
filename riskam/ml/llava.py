"""
ml.llava

The LLaVA MLLM model.
"""

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
    pipeline,
    AutoTokenizer,
)
import numpy as np
from PIL import Image
import torch


class Llava:
    """
    The LLaVA MLLM model.
    """

    MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"

    PROMPT = """
You are an advanced multimodal AI system designed to extract risk-related semantic features from RGB images captured by a mobile robot's camera. Your task is to analyze a single RGB image and output a JSON object containing key risk-awareness features related to human safety.

**Objective:**
Extract and encode potential risks to humans in the environment based on the provided image. Consider:
- **Proximity Risk:** The distance between the robot and any humans present. The closer the human, the higher the risk.
- **Human Awareness Risk:** Whether the human appears aware of the robot. Signs of unawareness include facing away, looking elsewhere, or being engaged in another activity.
- **Discomfort Risk:** If the human exhibits body language suggesting discomfort, such as stepping back, raising hands defensively, or facial expressions of concern.
- **Environmental Hazard Risk:** If the surroundings contain elements that may amplify risk, such as slippery floors, moving machinery, or cluttered areas.

**Input:**
A single RGB image captured from the mobile robot's perspective.

**Output Format (JSON):**
Return a JSON object with the following structure:
```json
{
  "proximity_risk": {
    "level": "low" | "moderate" | "high",
    "explanation": "<explanation of proximity risks>"
  },
  "human_awareness_risk": {
    "level": "low" | "moderate" | "high",
    "human_facing_robot": true | false,
    "explanation": "<explanation of human awareness risks>"
  },
  "discomfort_risk": {
    "level": "low" | "moderate" | "high",
    "explanation": "<explanation of human discomfort risks>"
  },
  "environmental_hazard_risk": {
    "level": "low" | "moderate" | "high",
    "explanation": "<explanation of environmental hazards>"
  }
}
```
Ensure your output is consistent, reasonable, and interpretable for downstream machine learning models.
"""

    CONVERSATION = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": PROMPT,
                },
            ],
        }
    ]

    MAX_NEW_TOKENS = 50000

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    def __init__(self):
        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        # self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)

        # self.model = AutoModelForImageTextToText.from_pretrained(
        #     self.MODEL_ID,
        #     quantization_config=self.quantization_config,
        #     device_map="auto",
        # )
        self.pipe = pipeline(
            "image-text-to-text",
            model=self.MODEL_ID,
            model_kwargs={"quantization_config": self.quantization_config},
            return_full_text=False,
        )

    def process_image(self, image_paths: list[str]) -> str:
        """
        Processes the given image in NumPy format and returns the model's output.
        """
        # Open the images
        images = []
        for image_path in image_paths:
            images.append(Image.open(image_path))

        # Use the Hugging Face pipeline
        # pipe = pipeline(
        #     "image-to-text",
        #     model=cls.MODEL_ID,
        #     model_kwargs={"quantization_config": cls.quantization_config},
        # )

        prompt = self.processor.apply_chat_template(
            self.CONVERSATION, add_generation_prompt=True
        )
        # tokenized_chat = self.tokenizer.apply_chat_template(
        #     self.CONVERSATION,
        #     tokenize=True,
        #     add_generation_prompt=True,
        #     return_tensors="pt",
        # )

        # inputs = self.processor(
        #     images=images, text=prompt, padding=True, return_tensors="pt"
        # ).to(self.model.device)

        # # Generate outputs
        # generate_ids = self.model.generate(**inputs, max_new_tokens=self.MAX_NEW_TOKENS)
        # outputs = self.processor.batch_decode(generate_ids, skip_special_tokens=True)
        outputs = self.pipe(
            images,
            text=prompt,
            generate_kwargs={"max_new_tokens": self.MAX_NEW_TOKENS},
        )

        print(outputs[0]["generated_text"])

        return ""
