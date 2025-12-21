from flask import Flask, render_template, request
import pickle
import numpy as np
import json

app = Flask(__name__)

# 1. Eğitilmiş Modeli ve Scaler'ı Yükle
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# 2. Sütun İsimlerini Yükle (Kontrol amaçlı)
with open('feature_columns.json', 'r') as f:
    features = json.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # --- HTML FORMUNDAN VERİLERİ AL ---

        # Sayısal Değerler
        Year = int(request.form['Year'])
        Present_Price = float(request.form['Present_Price'])
        Owner = int(request.form['Owner'])

        # --- KATEGORİK DÖNÜŞÜMLER (Mantık Kısmı) ---

        # 1. Yakıt Türü: Modelimiz 'Fuel_Type_Diesel' bekliyor.
        # Eğer kullanıcı Diesel seçerse 1, diğerleri (Petrol/CNG) için 0.
        Fuel_Type_Diesel = 1 if request.form['Fuel_Type'] == 'Diesel' else 0

        # 2. Satıcı Türü: Modelimiz 'Seller_Type_Individual' bekliyor.
        Seller_Type_Individual = 1 if request.form['Seller_Type'] == 'Individual' else 0

        # 3. Vites Türü: Modelimiz 'Transmission_Manual' bekliyor.
        Transmission_Manual = 1 if request.form['Transmission'] == 'Manual' else 0

        # --- VERİYİ HAZIRLAMA ---
        # Sıralama JSON dosyanla birebir aynı olmalı:
        # ['Year', 'Present_Price', 'Owner', 'Fuel_Type_Diesel', 'Seller_Type_Individual', 'Transmission_Manual']

        input_data = np.array([[Year, Present_Price, Owner, Fuel_Type_Diesel, Seller_Type_Individual, Transmission_Manual]])

        # --- ÖLÇEKLEME (SCALING) ---
        # Modeli eğitirken scaler kullandığımız için, tahminde de kullanmalıyız.
        input_data_scaled = scaler.transform(input_data)

        # --- TAHMİN ---
        prediction = model.predict(input_data_scaled)

        # Sonucu Yuvarla (2 basamak)
        output = round(prediction[0], 2)

        # Negatif fiyat çıkarsa kullanıcıya "Hesaplanamadı" diyelim
        if output < 0:
            return render_template('index.html', prediction_text="Tahmin: Negatif değer çıktı, lütfen girdileri kontrol edin.")
        else:
            return render_template('index.html', prediction_text=f'Tahmini Satış Fiyatı: {output} Lakhs')

if __name__ == "__main__":
    app.run(debug=True)