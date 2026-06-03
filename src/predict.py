def predict_new(model, scaler, new_data):
    new_data_scaled = scaler.transform(new_data)
    return model.predict(new_data_scaled)