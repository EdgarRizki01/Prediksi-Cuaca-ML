import json
from urllib import request
from datetime import datetime
import math

def ambil_data(adm4):
    url = f"https://api.bmkg.go.id/publik/prakiraan-cuaca?adm4={adm4}"
    resp = request.urlopen(url)
    data = json.loads(resp.read().decode())
    return data  

def ambil_data_harian(data):
    cuaca_data = data["data"][0]["cuaca"]
    records = []
    for blok in cuaca_data:
        for rec in blok:
            waktu_str = rec.get("local_datetime") or rec.get("datetime")
            waktu = datetime.fromisoformat(waktu_str.replace("Z", "+00:00"))
            hari = waktu.strftime("%A").lower()
            records.append({
                "hari": hari,
                "t": float(rec.get("t", 0)),
                "hu": float(rec.get("hu", 0)),
                "tcc": float(rec.get("tcc", 0)),
                "tp": float(rec.get("tp", 0)),  # curah hujan
                "weather": rec.get("weather_desc", "Tidak diketahui"),
                "time": waktu.strftime("%H:%M")
            })
    return records

def ekstrak_fitur_hari(records, hari):
    hari = hari.lower()
    subset = [r for r in records if r["hari"] == hari]
    if not subset:
        return None
    t = sum(r["t"] for r in subset) / len(subset)
    hu = sum(r["hu"] for r in subset) / len(subset)
    tcc = sum(r["tcc"] for r in subset) / len(subset)
    tp = sum(r["tp"] for r in subset) / len(subset)
    cuaca = subset[len(subset)//2]["weather"]
    return t, hu, tcc, tp, cuaca

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def prediksi_hujan(t, hu, tcc, tp):
    # Model sederhana (dummy weight)
    w1, w2, w3, w4, b = -0.2, 0.5, 0.8, 1.2, -0.3
    z = w1*t + w2*hu + w3*tcc + w4*tp + b
    p = sigmoid(z / 100)
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
    # adm4 = input("Masukkan kode wilayah adm4 BMKG: ").strip()
    adm4 = "35.19.02.2009"
    hari_input = input("Masukkan hari yang ingin diprediksi: ").strip().lower()

    data = ambil_data(adm4)
    records = ambil_data_harian(data)

    fitur = ekstrak_fitur_hari(records, hari_input)

    print(f"\n🔮 Prediksi Cuaca Hari {hari_input.capitalize()}:\n")

    if fitur:
        t, hu, tcc, tp, cuaca_asli = fitur
        prob = prediksi_hujan(t, hu, tcc, tp)
        label = klasifikasi_weather(prob)

        print(f"🧾 Berdasarkan Data BMKG:")
        print(f"- Kondisi: {cuaca_asli}")
        print(f"- Suhu: {t:.1f}°C | Kelembapan: {hu:.1f}% | Tutupan Awan: {tcc:.1f}% | Curah Hujan: {tp:.1f}mm")
        print(f"\n📈 Prediksi Model:")
        print(f"- Probabilitas hujan: {prob:.2f}")
        print(f"- Hasil: {label}\n")
    else:
        print(f"Data BMKG untuk hari {hari_input.capitalize()} belum tersedia.")
        print("🔁 Menggunakan estimasi berdasarkan pola hari sebelumnya...\n")

        if records:
            terakhir = records[-1]
            prob = prediksi_hujan(terakhir["t"], terakhir["hu"], terakhir["tcc"], terakhir["tp"])
            label = klasifikasi_weather(prob)
            print(f"Prediksi estimasi: {label} (Prob. Hujan: {prob:.2f})")
        else:
            print("⚠️ Tidak ada data sama sekali untuk wilayah ini.")
