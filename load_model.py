import keras
import json
import os

default_metrics = ['accuracy', 'precision', 'recall', keras.metrics.AUC()]

"""
    A wrapper around Keras' `load_model` method. This
    is needed because the Grad-Cam implementation does
    not work with the loaded model. This method clones
    the loaded model and uses the loaded models weights.
"""
def load_model(
    path,
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=default_metrics
):
    original_model = keras.models.load_model(path)
    model = keras.models.clone_model(original_model)
    model.set_weights(original_model.get_weights())
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    return model


"""
    Saves the results of a model evaluation as a JSON file.
    This allows for not needing to rerun the full training
    when running certain blocks in the notebook.
"""
# Title: Reading and Writing JSON to a File in Python
# Author: GeeksForGeeks
# Date: 19/07/2022
# Available: https://www.geeksforgeeks.org/reading-and-writing-json-to-a-file-in-python/
def save_results(
    path,
    histories,
    predictions,
    y_true
):
    os.makedirs(path, exist_ok=True)
    loss = []
    val_loss = []
    list_predictions = []
    list_y_true = []
    for i in range(len(histories)):
        loss.append(histories[i].history["loss"])
        val_loss.append(histories[i].history["val_loss"])
        list_predictions.append(predictions[i].tolist())
        list_y_true.append(y_true[i].tolist())

    to_write = {
        "loss": loss,
        "val_loss": val_loss,
        "predictions": list_predictions,
        "y_true": list_y_true
    }

    json_object = json.dumps(to_write, indent=4)
    with open(os.path.join(path, 'results.json'), "w") as outfile:
        outfile.write(json_object)


"""
    The follow up to the `save_results` method above.
    This reads the saved JSON results.
"""
# Title: Reading and Writing JSON to a File in Python
# Author: GeeksForGeeks
# Date: 19/07/2022
# Available: https://www.geeksforgeeks.org/reading-and-writing-json-to-a-file-in-python/
def load_results(path):
    with open(os.path.join(path, 'results.json'), "r") as openfile:
        return json.load(openfile)