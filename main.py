from pathlib import Path
import os

from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import HTMLResponse

from src.model.spacer import Spacer
from src.data.utils import decode_labels, get_unk_tokens


model_path = Path(__file__).parent / 'model' / 'rt-spacer.ckpt'

class RtSpacer:
    def __init__(self):
        self.model = Spacer.load_model_from_experiment(Path(model_path))
    def space(self, text):
        inputs = self.model.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_offsets_mapping=True,
            return_tensors='pt'
        )
        unk_tokens = get_unk_tokens([text], inputs['input_ids'], inputs['offset_mapping'], self.model.tokenizer.unk_token_id)[0]
        src_tokens = self.model.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        del inputs['offset_mapping']
        outputs = self.model(**inputs)
        y_hats = self.apply_threshold_to_outputs(outputs)
        return decode_labels(y_hats[0], src_tokens, unk_tokens, exclude_special_tokens=True)

model = RtSpacer()
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello, world!"}

@app.get("/form")
async def form_post(request: Request):
    return HTMLResponse("""
        <html>
            <head>
                <title>NLP Demo</title>
            </head>
            <body>
                <form method="post">
                    <input type="text" name="text" autofocus />
                    <button type="submit">Process</button>
                </form>
            </body>
        </html>
    """)

@app.post("/form")
async def form_post(request: Request):
    form = await request.form()
    text = form['text']
    output = model.space(text)
    return HTMLResponse(f"""
        <html>
            <head>
                <title>NLP Demo</title>
            </head>
            <body>
                <form method="post">
                    <input type="text" name="text" autofocus value="{text}" />
                    <button type="submit">Process</button>
                </form>
                <br>
                <h2>Output:</h2>
                {output}
            </body>
        </html>
    """)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))