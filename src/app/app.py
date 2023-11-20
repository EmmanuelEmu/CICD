import gradio as gr 
import tensorflow as tf


def predict(x: float) -> float:
    model = tf.keras.model.load_model("../../models/my_model.h5")
    y = model.predict([[x]])
    return y[0][0]


def run():
    app = gr.Interface(
        fn = predict,
        inputs = gr.components.Number(label='X'),
        outputs= gr.components.Number(label='Y'),
        allow_flagging = 'never',
    )
    
    app.launch(share=True)


if __name__ == "__main__":
    run()