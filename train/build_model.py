from tensorflow.keras.applications import MobileNetV2 
from tensorflow.python.keras.layers import Dense, Flatten, Dropout #type: ignore
from tensorflow.keras.models import Model 


# Load MobileNetV2 (pretrained)
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(128, 128, 3))

# Freeze pretrained layers (we only train the classifier on top)
base_model.trainable = False

# Add custom classification layers
x = Flatten()(base_model.output)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1, activation="sigmoid")(x)  # Binary output: 0 or 1

# Create the model
model = Model(inputs=base_model.input, outputs=x)

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
