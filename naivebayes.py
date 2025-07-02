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
    return ranges[-1][2]

# Fungsi untuk menghitung probabilitas atribut (likelihood)
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
            # Pastikan nilai bisa diubah ke float
            val = float(new_sample[idx])
            if idx in custom_ranges:
                new_sample[idx] = categorize_numerical_value(val, custom_ranges[idx])
            else:
                # Jika tidak ada custom range, ubah menjadi string (fallback)
                new_sample[idx] = str(val)
        processed_data.append(new_sample)

    # Hitung probabilitas tiap nilai atribut per kelas
    for index in range(len(processed_data[0]) - 1):
        attribute_probabilities[index] = {}
        for label in class_labels:
            filtered_data = [s for s in processed_data if s[-1] == label]
            total_samples = len(filtered_data)
            value_counts = {}
            for s in filtered_data:
                value = s[index]
                value_counts[value] = value_counts.get(value, 0) + 1
            attribute_probabilities[index][label] = {
                val: count / total_samples for val, count in value_counts.items()
            }

    return attribute_probabilities, processed_data

def naive_bayes_probabilities(sample, prior_probs, attribute_probabilities, class_labels, numerical_indices):
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
        probability = prior_probs[label]
        for index, value in enumerate(sample):
            value_prob = attribute_probabilities[index][label].get(value, 1e-6)  # Laplace smoothing
            probability *= value_prob
        probabilities[label] = probability
    return probabilities

def calculate_accuracy(true_labels, predicted_labels):
    correct = sum(t == p for t, p in zip(true_labels, predicted_labels))
    accuracy = (correct / len(true_labels)) * 100
    return accuracy

st.title("Klasifikasi Kelayakan Nasabah 💰")

st.write("Petunjuk:")
st.write("1. Download dataset https://shorturl.at/lIfEW")
st.write("2. Upload data")
st.write("3. Program akan membagi data menjadi data training (438) dan data testing (146).")
st.write("4. Program akan memproses data training dengan menghitung probabilitas atribut.")
st.write("5. Program akan menampilkan hasil probabilitas atribut, prediksi, serta akurasi data training dan data testing.")
st.write("6. Uji data kamu untuk kelayakan peminjaman!")

uploaded_file = st.file_uploader("Upload Dataset Nasabah (Excel)📤", type=["xlsx"])

if uploaded_file is not None:
    dataset = pd.read_excel(uploaded_file)
    
    # Debug info
    st.write("**Data Shape:**", dataset.shape)
    st.write("**Columns:**", dataset.columns.tolist())
    
    st.subheader("Data Preview👀")
    st.write(dataset)  # Show first 10 rows for verification
    
    data = dataset.values.tolist()
    
    total_rows = len(data)
    if total_rows < 584:
        st.error("Dataset harus memiliki setidaknya 585 baris data (termasuk header).")
    else:
        # Split data into training and testing
        training_data = data[0:438]  # Baris 2 sampai 439 (438 baris)
        testing_data = data[438:585]  # Baris 440 sampai 585 (146 baris)
        
        st.subheader("Data Training (438 Baris)")
        st.write(pd.DataFrame(training_data, columns=dataset.columns))
        
        st.subheader("Data Testing (146 Baris)")
        st.write(pd.DataFrame(testing_data, columns=dataset.columns))

        # Calculate and display prior probabilities for training data
        training_prior_probs = calculate_prior_probabilities(training_data)
        st.subheader("Probabilitas Prior Data Training")
        training_prior_df = pd.DataFrame(list(training_prior_probs.items()), 
                                       columns=['Kelas', 'Probabilitas Prior'])
        st.write(training_prior_df.style.format({'Probabilitas Prior': '{:.2%}'}))
        
        # Definisikan indeks numerik dan kategorikal
        numerical_indices = [0, 4, 6, 7]  # Usia(0), Penghasilan(4), Nilai Pinjam(6), Tenor(7)
        categorical_indices = [i for i in range(len(data[0]) - 1) if i not in numerical_indices]
        
        class_labels = list(set(sample[-1] for sample in training_data))
        prior_probs = calculate_prior_probabilities(training_data)
        
        attribute_probabilities, processed_training_data = calculate_attribute_probabilities(
            training_data, numerical_indices, categorical_indices, class_labels
        )
        
        # Tampilkan hasil probabilitas tiap atribut
        st.subheader("Probabilitas Tiap Atribut Berdasarkan Data Training")
        columns = dataset.columns.tolist()
        
        for i in range(len(columns) - 1):
            st.write(f"**Atribut: {columns[i]}**")
            all_values = set()
            for label in class_labels:
                for val in attribute_probabilities[i][label].keys():
                    all_values.add(val)
            all_values = sorted(all_values)  # Sorting for better readability
            
            df_prob = pd.DataFrame(index=all_values, columns=class_labels)
            for val in all_values:
                for label in class_labels:
                    df_prob.loc[val, label] = attribute_probabilities[i][label].get(val, 0.0)
            
            st.dataframe(df_prob.style.format("{:.2%}"))
            st.write("---")
        
        # Proses data testing
        st.subheader("Hasil Prediksi Data Testing")
        testing_true_labels = [row[-1] for row in testing_data]
        testing_predictions = []
        testing_probs_layak = []
        testing_probs_tidak_layak = []

        for row in testing_data:
            sample_test = row[:-1]
            probs_test = naive_bayes_probabilities(sample_test, prior_probs, attribute_probabilities, class_labels, numerical_indices)
            predicted_class = max(probs_test, key=probs_test.get)
            testing_predictions.append(predicted_class)
            testing_probs_layak.append(f"{probs_test.get('Layak', 0):.10f}")
            testing_probs_tidak_layak.append(f"{probs_test.get('Tidak Layak', 0):.10f}")

        result_df = pd.DataFrame(testing_data, columns=dataset.columns)
        result_df['Probabilitas Layak'] = testing_probs_layak
        result_df['Probabilitas Tidak Layak'] = testing_probs_tidak_layak
        result_df['Prediksi Kelas'] = testing_predictions

        st.write("Hasil Prediksi:")
        st.write(result_df)

        testing_accuracy = calculate_accuracy(testing_true_labels, testing_predictions)
        st.write(f"**Akurasi pada Data Testing:** {testing_accuracy:.2f}%")
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
