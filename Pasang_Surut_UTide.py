# Streamlit App Analisa Pasang Surut dengan Prediksi Tanggal di Bawah Latitude

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utide import solve, reconstruct
import math
from sklearn.metrics import r2_score
from datetime import datetime
import io
import zipfile

st.set_page_config(page_title="Aplikasi Analisa Pasang Surut", page_icon="🌊")
st.title("🌊Aplikasi Analisa Pasang Surut")

st.header("1. Upload Data")
st.markdown("""
**Format file Excel yang diupload harus memiliki minimal 2 kolom dengan nama:**

- `Tanggal`: Tanggal dan waktu
- `Elevasi`: Tinggi air (meter)
""")

uploaded_file = st.file_uploader("Pilih file Excel", type=["xlsx"])

latitude = st.number_input("Masukkan nilai latitude lokasi pengamatan (dalam derajat desimal):", 
                           min_value=-90.0, max_value=90.0, value=0.000000, format="%.6f")

st.markdown("### Pilih Rentang Prediksi Pasang Surut")
start_pred = st.date_input("Tanggal Mulai", datetime(2025, 1, 1))
end_pred = st.date_input("Tanggal Akhir", datetime(2025, 6, 30))

interval_options = {
    "6 Jam": "6H",
    "3 Jam": "3H",
    "1 Jam": "1H",
    "30 Menit": "30min",
    "15 Menit": "15min"
}
interval_label = st.selectbox("Pilih Interval Prediksi", list(interval_options.keys()), index=1)
interval = interval_options[interval_label]


if start_pred >= end_pred:
    st.error("Tanggal mulai harus lebih awal dari tanggal akhir.")

run_analysis = st.button("Analisa Pasang Surut")

if uploaded_file is not None and run_analysis:
    df = pd.read_excel(uploaded_file)
    bfill_imputed = df.copy(deep=True)
    bfill_imputed.fillna(method='bfill', inplace=True)
    tanggal = pd.to_datetime(bfill_imputed['Tanggal'])
    sensor = np.array(bfill_imputed['Elevasi'])

    st.header("2. Grafik Pasang Surut Observasi")
    fig_obs, ax_obs = plt.subplots(figsize=(17, 8))
    ax_obs.plot(tanggal, sensor, color='black', linewidth=1)
    ax_obs.set(xlabel='Tanggal', ylabel='Tinggi Air [m]', title='Grafik Pasang Surut')
    ax_obs.grid()
    st.pyplot(fig_obs)

    constituents = ['M2', 'S2', 'K1', 'O1', 'P1', 'K2', 'N2', 'M4', 'MS4']
    decompose_utide = solve(tanggal, sensor, lat=latitude,
                            constit=constituents, nodal=False,
                            trend=False, method='ols', conf_int='linear')

    DatFrame_UTide = pd.DataFrame({
        'Name': decompose_utide['name'],
        'Freq [cph]': decompose_utide['aux']['frq'],
        'Amplitude [m]': decompose_utide['A'],
        'Amp CI [m]': decompose_utide['A_ci'],
        'Phase [°]': decompose_utide['g'],
        'Phase CI [°]': decompose_utide['g_ci']
    })

    st.header("3. Tabel Komponen Harmonik")
    st.dataframe(DatFrame_UTide.style.format({
        'Freq [cph]': '{:.4f}',
        'Amplitude [m]': '{:.4f}',
        'Amp CI [m]': '{:.4f}',
        'Phase [°]': '{:.4f}',
        'Phase CI [°]': '{:.4f}'
    }))

    MSL = np.mean(sensor)
    amp = lambda x: decompose_utide['A'][decompose_utide['name'] == x][0] if x in decompose_utide['name'] else 0
    M2, S2, K1, O1, P1, K2, M4, MS4 = [amp(x) for x in ['M2','S2','K1','O1','P1','K2','M4','MS4']]

    HWS = MSL + M2 + S2 + K1 + O1 + P1 + K2 + M4 + MS4
    LWS = MSL - (M2 + S2 + K1 + O1 + P1 + K2 + M4 + MS4)
    MHWS = MSL + M2 + S2
    MLWS = MSL - (M2 + S2)
    MHWL = MSL + M2 + K1 + O1
    MLWL = MSL - (M2 + K1 + O1)
    HHWL = MSL + M2 + S2 + K1 + O1 + P1 + K2
    LLWL = MSL - (M2 + S2 + K1 + O1 + P1 + K2)
    Tunggang_Pasang = HWS - LWS
    Formzahl = (K1 + O1) / (M2 + S2) if (M2 + S2) != 0 else 0

    if Formzahl > 3.0:
        jenis_pasang_surut = "Pasang Surut Harian Tunggal (Diurnal)"
    elif 1.5 < Formzahl <= 3.0:
        jenis_pasang_surut = "Pasang Surut Campuran Condong ke Harian Tunggal"
    elif 0.25 < Formzahl <= 1.5:
        jenis_pasang_surut = "Pasang Surut Campuran Condong ke Harian Ganda"
    else:
        jenis_pasang_surut = "Pasang Surut Harian Ganda (Semidiurnal)"

    df_elevasi = pd.DataFrame({
        'Parameter': [
            'MSL', 'HWS', 'LWS', 'MHWS', 'MLWS',
            'MHWL', 'MLWL', 'HHWL', 'LLWL', 'Tunggang Pasang',
            'Formzahl', 'Jenis Pasut'
        ],
        'Elevasi (m)': [
            MSL, HWS, LWS, MHWS, MLWS,
            MHWL, MLWL, HHWL, LLWL, Tunggang_Pasang,
            Formzahl, jenis_pasang_surut
        ]
    })

    fig_formzahl, ax_formzahl = plt.subplots(figsize=(6, 4))
    ax_formzahl.bar(["Formzahl"], [Formzahl], color='orange')
    ax_formzahl.set_ylim(0, max(Formzahl + 1, 4))
    ax_formzahl.axhline(3.0, color='red', linestyle='--', label='Diurnal')
    ax_formzahl.axhline(1.5, color='green', linestyle='--', label='Mixed-Diurnal')
    ax_formzahl.axhline(0.25, color='blue', linestyle='--', label='Mixed-Semidiurnal')
    ax_formzahl.set_ylabel("Nilai Formzahl")
    ax_formzahl.legend()
    ax_formzahl.text(0, Formzahl + 0.1, f"{Formzahl:.4f}", ha='center', color='blue', fontsize=12)

    st.header("4. Visualisasi Formzahl dan Klasifikasi Pasang Surut")
    st.pyplot(fig_formzahl)
    st.markdown(f"**Jenis Pasang Surut:** {jenis_pasang_surut}")
    st.markdown("""
    **Kategori Formzahl (F):**

    - F ≤ 0.25 : Pasang surut harian ganda *(Semidiurnal)*  
    - 0.25 < F ≤ 1.50 : Pasang surut campuran condong ke harian ganda *(Mixed, Predominantly Semidiurnal)*  
    - 1.50 < F ≤ 3.00 : Pasang surut campuran condong ke harian tunggal *(Mixed, Predominantly Diurnal)*  
    - F > 3.00 : Pasang surut harian tunggal *(Diurnal)*
    """)
    st.header("5. Tabel Perhitungan Elevasi Penting")
    st.dataframe(df_elevasi.style.format({
        'Elevasi (m)': lambda x: f"{x:.2f}" if isinstance(x, (int,float)) else x
    }))

    fig_elevasi, ax_elevasi = plt.subplots(figsize=(17, 6))
    ax_elevasi.plot(tanggal, sensor, label='Pasang Surut', color='blue')
    ax_elevasi.axhline(MSL, color='orange', linestyle='-', label='MSL')
    ax_elevasi.axhline(HHWL, color='gray', linestyle='--', label='HHWL')
    ax_elevasi.axhline(LLWL, color='gold', linestyle='--', label='LLWL')
    ax_elevasi.axhline(MHWL, color='green', linestyle='--', label='MHWL')
    ax_elevasi.axhline(MLWL, color='lime', linestyle='--', label='MLWL')
    ax_elevasi.axhline(HWS, color='black', linestyle=':', label='HWS')
    ax_elevasi.axhline(LWS, color='brown', linestyle=':', label='LWS')
    ax_elevasi.axhline(MHWS, color='purple', linestyle='-.', label='MHWS')
    ax_elevasi.axhline(MLWS, color='pink', linestyle='-.', label='MLWS')
    ax_elevasi.set_title('Grafik Pasang Surut + Elevasi Penting')
    ax_elevasi.set_ylabel('Tinggi Permukaan Air (m)')
    ax_elevasi.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False)
    ax_elevasi.grid(True)
    st.header("Grafik Pasang Surut dengan Elevasi Penting")
    st.pyplot(fig_elevasi)

    st.header("6. Prediksi Pasang Surut")
    timepred_UTIDE = pd.date_range(start=start_pred, end=end_pred, freq=interval)
    tidepred_UTIDE = reconstruct(timepred_UTIDE, decompose_utide, verbose=True)
    h_out_predutide = tidepred_UTIDE.h
    fig_pred, ax_pred = plt.subplots(figsize=(17, 6))
    ax_pred.plot(timepred_UTIDE, h_out_predutide, color='blue', linewidth=1)
    ax_pred.set(xlabel='Tanggal', ylabel='Tinggi Air [m]', title='Prediksi Tinggi Pasang')
    ax_pred.grid()
    st.pyplot(fig_pred)
    df_prediksi = pd.DataFrame({'Tanggal': timepred_UTIDE, 'Elevasi': h_out_predutide})

    tide_utide = reconstruct(tanggal, decompose_utide, verbose=False)
    pred_utide = tide_utide.h
    residual_utide = sensor - pred_utide
    RMSE_UTIDE = math.sqrt(np.mean(np.square(sensor - pred_utide)))
    R_square = r2_score(sensor, pred_utide)

    fig_obs_pred, ax_obs_pred = plt.subplots(figsize=(10, 4))
    ax_obs_pred.plot(tanggal, sensor, label="Observasi", color="black")
    ax_obs_pred.plot(tanggal, pred_utide, label="Prediksi", color="red")
    ax_obs_pred.set_title("Observasi vs Prediksi")
    ax_obs_pred.set_ylabel("Tinggi Air [m]")
    ax_obs_pred.legend()
    ax_obs_pred.grid(True)

    fig_residual, ax_residual = plt.subplots(figsize=(10, 4))
    ax_residual.plot(tanggal, residual_utide, label="Residual", color="green")
    ax_residual.set_title("Residual")
    ax_residual.set_ylabel("Selisih [m]")
    ax_residual.legend()
    ax_residual.grid(True)

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig_obs_pred)
    with col2:
        st.pyplot(fig_residual)

    st.header("7. Ringkasan Analisis")
    st.markdown(f"""
- **Formzahl**: {Formzahl:.2f}  
- **Jenis Pasang Surut**: {jenis_pasang_surut}  
- **RMSE**: {RMSE_UTIDE:.2f}  
- **R²**: {R_square:.2f}  
- **MSL**: {MSL:.2f} m  
- **HWS**: {HWS:.2f} m  
- **LWS**: {LWS:.2f} m  
- **Tunggang Pasang**: {Tunggang_Pasang:.2f} m  
""")

    st.header("8. Ekspor Hasil")

    def fig_to_bytes(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='jpg')
        buf.seek(0)
        return buf.read()

    with io.BytesIO() as buffer:
        with zipfile.ZipFile(buffer, "w") as zipf:
            zipf.writestr("Grafik_Observasi.jpg", fig_to_bytes(fig_obs))
            zipf.writestr("Grafik_Formzahl.jpg", fig_to_bytes(fig_formzahl))
            zipf.writestr("Grafik_Observasi_vs_Prediksi.jpg", fig_to_bytes(fig_obs_pred))
            zipf.writestr("Grafik_Residual.jpg", fig_to_bytes(fig_residual))
            zipf.writestr("Grafik_Elevasi_Penting.jpg", fig_to_bytes(fig_elevasi))
            zipf.writestr("Grafik_Prediksi.jpg", fig_to_bytes(fig_pred))
        buffer.seek(0)
        st.download_button("📷 Unduh Semua Grafik JPG (ZIP)", data=buffer, file_name="Grafik_Pasang_Surut.zip", mime="application/zip")

    st.download_button("📄 Unduh Komponen Harmonik", DatFrame_UTide.to_csv(index=False), file_name="Komponen_Harmonik.csv", mime="text/csv")
    st.download_button("📄 Unduh Elevasi Penting", df_elevasi.to_csv(index=False), file_name="Elevasi_Penting.csv", mime="text/csv")
    st.download_button("📅 Unduh Data Prediksi", df_prediksi.to_csv(index=False), file_name="Prediksi_Pasang.csv", mime="text/csv")

# Tanda tangan
st.markdown("---")
st.markdown("**by : Moh. Tri Arwadi modified ChatGPT and Blackbox**") 

