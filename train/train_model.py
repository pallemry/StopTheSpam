from build_model import model
from generate import train_generator, test_generator

history = model.fit(
    train_generator,
    epochs=10,  # Adjust epochs if needed
    validation_data=test_generator
)

# Evaluate on test data
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.4f}")

# Save the model for future use
model.save("image_classifier.h5")
print("Model saved as 'image_classifier.h5'.")