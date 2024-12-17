import streamlit as st
import pandas as pd

st.set_page_config(page_title="Klasifikasi Nasabah", page_icon=":money_with_wings:")

# Fungsi untuk menghitung probabilitas prior
def calculate_prior_probabilities(data):
    total_samples = len(data)
    class_counts = {}
    for sample in data:
        label = sample[-1]
        class_counts[label] = class_counts.get(label, 0) + 1
    return {label: count / total_samples for label, count in class_counts.items()}

# Fungsi untuk mengelompokkan nilai numerik ke dalam kategori
def categorize_numerical_value(value, ranges):
    for r in ranges:
        min_val, max_val, label = r
        if min_val <= value <= max_val:
            return label
    # Jika tidak masuk ke rentang manapun
    return ranges[-1][2]

# Fungsi untuk menghitung probabilitas likelihood setiap atribut
def calculate_attribute_probabilities(data, numerical_indices, categorical_indices, class_labels):
    attribute_probabilities = {}

    # Rentang kustom untuk fitur numerik
    custom_ranges = {
        0: [  # Usia
            (17, 22, "17-22"),
            (23, 28, "23-28"),
            (29, 34, "29-34"),
            (35, 40, "35-40"),
            (41, float('inf'), ">40")
        ],
        4: [  # Penghasilan
            (2000000, 2749000, '2.000.000 - 2.749.000'),
            (2750000, 3499000, '2.750.000 - 3.499.000'),
            (3500000, 4249000, '3.500.000 - 4.249.000'),
            (4250000, 4999999, '4.250.000 - 4.999.999'),
            (5000000, float('inf'), '≥ 5.000.000')
        ],
        6: [  # Nilai Pinjaman
            (1000000, 2499999, '1.000.000 - 2.499.999'),
            (2500000, 3999999, '2.500.000 - 3.999.999'),
            (4000000, 5499999, '4.000.000 - 5.499.999'),
            (5500000, 6999999, '5.500.000 - 6.999.999'),
            (7000000, float('inf'), '≥ 7.000.000')
        ],
        7: [  # Tenor
            (3, 4, "3-4 bulan"),
            (5, 6, "5-6 bulan"),
            (7, 8, "7-8 bulan"),
            (9, 10, "9-10 bulan"),
            (11, 12, "11-12 bulan")
        ]
    }

    # Ubah nilai numerik di data menjadi kategori
    processed_data = []
    for sample in data:
        new_sample = sample.copy()
        for idx in numerical_indices:
            val = float(new_sample[idx])
            if idx in custom_ranges:
                new_sample[idx] = categorize_numerical_value(val, custom_ranges[idx])
            else:
                # Jika tidak ada custom range, fallback bisa ditambahkan.
                new_sample[idx] = str(val)
        processed_data.append(new_sample)

    # Hitung probabilitas tiap nilai atribut per kelas
    for index in range(len(processed_data[0]) - 1):
        attribute_probabilities[index] = {}
        for label in class_labels:
            filtered_data = [sample for sample in processed_data if sample[-1] == label]
            total_samples = len(filtered_data)
            value_counts = {}
            for sample in filtered_data:
                value = sample[index]
                value_counts[value] = value_counts.get(value, 0) + 1
            
            attribute_probabilities[index][label] = {
                val: count / total_samples for val, count in value_counts.items()
            }
    
    return attribute_probabilities, processed_data

# Fungsi untuk menghitung probabilitas kelas dari input sample
def naive_bayes_probabilities(sample, prior_probs, attribute_probabilities, class_labels, numerical_indices):
    # Rentang kustom yang sama seperti pada training
    custom_ranges = {
        0: [  # Usia
            (17, 22, "17-22"),
            (23, 28, "23-28"),
            (29, 34, "29-34"),
            (35, 40, "35-40"),
            (41, float('inf'), ">40")
        ],
        4: [  # Penghasilan
            (2000000, 2749000, '2.000.000 - 2.749.000'),
            (2750000, 3499000, '2.750.000 - 3.499.000'),
            (3500000, 4249000, '3.500.000 - 4.249.000'),
            (4250000, 4999999, '4.250.000 - 4.999.999'),
            (5000000, float('inf'), '≥ 5.000.000')
        ],
        6: [  # Nilai Pinjaman
            (1000000, 2499999, '1.000.000 - 2.499.999'),
            (2500000, 3999999, '2.500.000 - 3.999.999'),
            (4000000, 5499999, '4.000.000 - 5.499.999'),
            (5500000, 6999999, '5.500.000 - 6.999.999'),
            (7000000, float('inf'), '≥ 7.000.000')
        ],
        7: [  # Tenor
            (3, 4, "3-4 bulan"),
            (5, 6, "5-6 bulan"),
            (7, 8, "7-8 bulan"),
            (9, 10, "9-10 bulan"),
            (11, 12, "11-12 bulan")
        ]
    }

    # Konversi sample input
    for idx in numerical_indices:
        val = float(sample[idx])
        if idx in custom_ranges:
            sample[idx] = categorize_numerical_value(val, custom_ranges[idx])
        else:
            sample[idx] = str(val)

    probabilities = {}
    for label in class_labels:
        # Mulai dari prior probability
        probability = prior_probs[label]
        # Kalikan dengan likelihood tiap fitur
        for index, value in enumerate(sample):
            value_prob = attribute_probabilities[index][label].get(value, 1e-6)
            probability *= value_prob
        probabilities[label] = probability
    return probabilities

st.title("Klasifikasi Kelayakan Nasabah 💰")

st.write("Petunjuk:")
st.write("1. Upload data training dalam format Excel.")
st.write("2. Program akan memproses data training dengan membagi atribut numerik ke dalam rentang kategori dan menghitung probabilitas prior serta likelihood.")
st.write("3. Masukkan data testing secara manual melalui UI. Tidak perlu upload file test data.")
st.write("4. Program akan menampilkan probabilitas tiap kelas dan kelas dengan probabilitas tertinggi sebagai hasil prediksi.")
st.write("5. Program juga menampilkan hasil probabilitas tiap atribut berdasarkan data training yang diunggah.")

# Upload data training
uploaded_file = st.file_uploader("Upload Data Training Nasabah (Excel)📤", type=["xlsx"])

if uploaded_file is not None:
    # Baca dataset
    dataset = pd.read_excel(uploaded_file)
    st.subheader("Data Training Preview👀")
    st.write(dataset.head())
    
    data = dataset.values.tolist()
    
    # Indeks numerik & kategorikal (sesuaikan dengan dataset)
    numerical_indices = [0, 4, 6, 7]
    categorical_indices = [i for i in range(len(data[0]) - 1) if i not in numerical_indices]
    
    class_labels = list(set(sample[-1] for sample in data))
    
    # Hitung prior probability
    prior_probs = calculate_prior_probabilities(data)
    
    # Hitung attribute probabilities
    attribute_probabilities, processed_data = calculate_attribute_probabilities(
        data, numerical_indices, categorical_indices, class_labels
    )
    
    # Tampilkan hasil probabilitas tiap atribut
    st.subheader("Probabilitas Tiap Atribut Berdasarkan Data Training")
    columns = dataset.columns.tolist()
    
    for i in range(len(columns) - 1):
        st.write(f"**Atribut: {columns[i]}**")
        # Ambil nilai-nilai unik dari atribut ini
        # value_list untuk menata baris
        all_values = set()
        for label in class_labels:
            for val in attribute_probabilities[i][label].keys():
                all_values.add(val)
        all_values = list(all_values)
        
        # Buat dataframe: index = nilai atribut, columns = kelas, values = probabilitas
        df_prob = pd.DataFrame(index=all_values, columns=class_labels)
        for val in all_values:
            for label in class_labels:
                df_prob.loc[val, label] = attribute_probabilities[i][label].get(val, 0.0)
        
        # Tampilkan dataframe
        st.dataframe(df_prob.style.format("{:.2%}"))
        st.write("---")
    
    st.subheader("Input Data Testing")
    st.write("Masukkan data testing di bawah ini:")
    input_prompts = [
        "Usia (numerik)", "Jenis Kelamin (L/P)", "Status Perkawinan", "Profesi",
        "Penghasilan (numerik)", "Status Pinjaman", "Nilai Pinjam (numerik)", "Tenor (numerik)"
    ]
    
    sample_to_predict = []
    for prompt in input_prompts:
        value = st.text_input(prompt)
        sample_to_predict.append(value)
    
    if st.button("Prediksi"):
        if all(v.strip() != "" for v in sample_to_predict):
            # Hitung probabilitas setiap kelas
            probs = naive_bayes_probabilities(sample_to_predict, prior_probs, attribute_probabilities, class_labels, numerical_indices)
            
            # Tampilkan probabilitas setiap kelas
            st.subheader("Probabilitas Setiap Kelas:")
            for cls in class_labels:
                st.write(f"Kelas {cls}: {probs[cls]:.6f}")
            
            # Tentukan kelas dengan probabilitas tertinggi
            predicted_class = max(probs, key=probs.get)
            st.success(f"Kelas yang Diprediksi: {predicted_class}")
        else:
            st.error("Mohon isi semua atribut terlebih dahulu.")

st.markdown("""
<h3 style="text-align: center;">>>> Kelompok 2 <<<</h3>
<p style="text-align: center;">1. Abril Berliando Cahyariata (2301010186)</p>
<p style="text-align: center;">2. Septaro Travian Gadha (23081010270)</p>
<p style="text-align: center;">3. Alvino Dwi Nengku Wijaya (23081010284)</p>
""", unsafe_allow_html=True)
