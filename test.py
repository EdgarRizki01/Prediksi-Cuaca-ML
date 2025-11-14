import json
from urllib import request
from datetime import datetime, timedelta
import math

def ambil_data(adm4):
    # Ambil data prakiraan cuaca dari BMKG
    url = f"https://api.bmkg.go.id/publik/prakiraan-cuaca?adm4={adm4}"
    resp = request.urlopen(url)
    data = json.loads(resp.read().decode())
    return data

def ambil_data_harian(data):
    cuaca_all = data["data"][0]["cuaca"]
    records = []
    for blok in cuaca_all:
        for rec in blok:
            waktu_str = rec.get("local_datetime") or rec.get("datetime")
            waktu = datetime.fromisoformat(waktu_str.replace("Z", "+00:00"))
            hari = waktu.strftime("%A").lower()

            records.append({
                "hari": hari,
                "t": float(rec.get("t", 0)),
                "hu": float(rec.get("hu", 0)),
                "tcc": float(rec.get("tcc", 0)),
                "weather": rec.get("weather_desc", "Tidak diketahui")
            })
    return records

def ekstrak_fitur_hari(records, hari):
    # Ambil semua data dari hari tertentu
    hari = hari.lower()
    subset = [r for r in records if r["hari"] == hari]
    if not subset:
        return None
    # Rata-rata harian
    t = sum(r["t"] for r in subset) / len(subset)
    hu = sum(r["hu"] for r in subset) / len(subset)
    tcc = sum(r["tcc"] for r in subset) / len(subset)
    return t, hu, tcc, subset[0]["weather"]

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def prediksi_hujan(t, hu, tcc):
    # Contoh model logistic regression sederhana
    w1, w2, w3, b = -0.3, 0.5, 0.8, -0.2
    z = w1*t + w2*hu + w3*tcc + b
    p = sigmoid(z/100)
    return p

def klasifikasi_weather(prob):
    if prob > 0.8:
        return "⛈️ Badai atau Hujan Lebat"
    elif prob > 0.6:
        return "🌧️ Hujan Sedang"
    elif prob > 0.4:
        return "🌦️ Hujan Ringan"
    elif prob > 0.2:
        return "☁️ Berawan"
    else:
        return "☀️ Cerah"

if __name__ == "__main__":
    # adm4 = input("Masukkan kode wilayah adm4 BMKG: ")
    adm4 = "35.19.02.2009"
    data = ambil_data(adm4)
    records = ambil_data_harian(data)

    if not records:
        print("❌ Gagal ambil data dari BMKG.")
        exit()

    print("\n🔮 Prediksi Cuaca 3 Hari ke Depan\n")

    today = datetime.now()
    for i in range(3):
        target_day = (today + timedelta(days=i)).strftime("%A").lower()
        fitur = ekstrak_fitur_hari(records, target_day)

        if fitur:
            t, hu, tcc, kondisi_asli = fitur
            prob = prediksi_hujan(t, hu, tcc)
            label = klasifikasi_weather(prob)
            print(f"{target_day.capitalize()} — Prob. Hujan: {prob:.2f} → {label}")
            print(f"   Data BMKG: {kondisi_asli} (T={t:.1f}°C, HU={hu:.1f}%, TCC={tcc:.1f}%)\n")
        else:
            print(f"{target_day.capitalize()} — Data BMKG belum tersedia, gunakan prediksi estimasi sebelumnya.")
