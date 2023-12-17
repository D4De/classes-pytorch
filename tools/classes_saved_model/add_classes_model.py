from tensorflow import keras
import json
import h5py
from src.error_simulator_keras import create_injection_sites_layer_simulator
import os

with open("config.json") as f:
    config = json.load(f)


def gen_injection_sites():
    available_injection_sites, masks = create_injection_sites_layer_simulator(
        num_requested_injection_sites=config["injection_sites"] * 5,
        layer_type=config["layer_type"],
        layer_output_shape_cf=config["layer_output_shape_cf"],
        layer_output_shape_cl=config["layer_output_shape_cl"],
        models_folder=config["models_folder"],
        range_min=config["range_min"],
        range_max=config["range_max"],
    )

    return available_injection_sites, masks


def get_user_choice():
    model = keras.models.load_model(config["model"], compile=False)
    print("The target model is composed by the following layers:")
    for i, layer in enumerate(model.layers):
        print(f"\t{i}: {layer.name}")

    user_choice = int(input("Please select which layer you want to target"))

    tries = 0

    while user_choice < 0 or user_choice > len(model.layers):
        tries += 1
        if tries == 3:
            exit(0)
        print("You didn't provide a valid layer id, please try again")
        user_choice = int(input("Please select which layer you want to target"))

    selected_layer = model.layers[user_choice].name
    return selected_layer


def get_model_description():
    original_model_path = config["model"]
    original_model_file = h5py.File(original_model_path, "r")

    # Get the description of the model
    description = original_model_file.attrs["model_config"]
    return description


def add_classes(model_description, selected_layer):
    with open("classes_description.json", "r") as g:
        classes_description = json.load(g)

    injections, masks = gen_injection_sites()
    classes_description["inbound_nodes"][0][0][0] = selected_layer
    classes_description["config"]["INJ_SITES"] = injections.tolist()
    classes_description["config"]["MASKS"] = masks.tolist()
    classes_description["config"]["NUM_INJECTION_SITES"] = config["injection_sites"]

    model_description["config"]["layers"].append(classes_description)

    for layer in model_description["config"]["layers"]:
        if (
            layer["class_name"] != "ErrorSimulator"
            and layer["inbound_nodes"][0][0][0] == selected_layer
        ):
            layer["inbound_nodes"][0][0][0] = "simulator"

    return model_description


def save_model_with_classes(modified_description):
    # Open the original model file
    original_model_path = config["model"]
    original_model_file = h5py.File(original_model_path, "r")

    # Create a new model file to save the modified model
    modified_model_path = os.path.join(config["new_model"], "new_weights.h5")
    modified_model_file = h5py.File(modified_model_path, "w")

    # Write the modified model configuration to the new model file
    modified_model_file.attrs.create(
        "model_config", modified_description.encode("utf-8")
    )

    # Iterate over the original model file and copy each layer to the new model file
    for layer_name in original_model_file.keys():
        original_model_file.copy(layer_name, modified_model_file)

    # Close both model files
    original_model_file.close()
    modified_model_file.close()


def main():
    selected_layer = get_user_choice()
    model_description = json.loads(get_model_description())
    modified_description = add_classes(model_description, selected_layer)
    save_model_with_classes(modified_description)


if __name__ == "__main__":
    main()
