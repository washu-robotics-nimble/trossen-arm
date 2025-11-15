from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from PIL import Image


image = None

processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
pixel_values = processor(images = image, return_tensors="pt").pixel_values


generated_ids = model.generate(pixel_values)


generated_text = processor.batch_decode(generated_ids, skip_special_tokens = True)[0]