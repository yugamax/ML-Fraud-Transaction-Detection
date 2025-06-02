@app.post("/predict")
async def predict(data: Transaction_data):
    global df2
    acc_holder = data.acc_holder
    features = data.features

    if len(features) != 18:
        return {"ML error": "Missing features. Expected 18 features."}

    input_data = np.array(features).reshape(1, -1)
    probs = model.predict_proba(input_data)[0]
    prediction = np.argmax(probs)
    confidence = probs[prediction]

    if prediction == 1 and confidence > 0.75:
        label = "Fraud"
        fr_type = "Unsafe Transaction"
        row_exists = ((df.iloc[:, 2:20] == features).all(axis=1)).any()
        if not row_exists:
            changes_in_dataset(label, features)

    elif 0.55 < confidence < 0.75:
        label = "Non - Fraud"
        fr_type = "Mildly Unsafe Transaction"

        if acc_holder in df2["IDs"].values:
            label = "Fraud"
            fr_type = "Unsafe Transaction"
            df2 = df2[df2["IDs"] != acc_holder]
            df2.to_csv("dataset/mildly_unsafe_transactions.csv", index=False)
            changes_in_dataset(label, features)
        else:
            new_row2 = [len(df2)] + [acc_holder] + [datetime.now().strftime("%d-%m-%Y %H:%M:%S")]
            df2.loc[len(df2)] = new_row2
            df2.to_csv("dataset/mildly_unsafe_transactions.csv", index=False)
    else:
        label = "Non - Fraud"
        fr_type = "Safe Transaction"

    print(f"Prediction: {label}, Confidence: {confidence:.2f}, Type: {fr_type}")

    return {"prediction": label, "Type": fr_type}
